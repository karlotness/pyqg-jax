# Copyright Karl Otness
# SPDX-License-Identifier: MIT


import operator
import itertools
import jax
import jax.numpy as jnp
from . import _utils, state as _state


@_utils.register_pytree_node_class_private
class PseudoSpectralKernel:
    def __init__(
        self,
        *,
        nz: int,
        ny: int,
        nx: int,
        rek: float = 0,
        precision: _state.Precision = _state.Precision.SINGLE,
    ):
        # Store small, fundamental properties (others will be computed on demand)
        self.nz = nz
        self.ny = ny
        self.nx = nx
        self.rek = rek
        self.precision = precision
        if self.precision == _state.Precision.SINGLE:
            self._dtype_real = jnp.float32
            self._dtype_complex = jnp.complex64
        elif self.precision == _state.Precision.DOUBLE:
            self._dtype_real = jnp.float64
            self._dtype_complex = jnp.complex128
        else:
            raise ValueError("invalid choice for precision")

    def get_full_state(
        self, state: _state.PseudoSpectralState
    ) -> _state.FullPseudoSpectralState:
        def _empty_real():
            return jnp.zeros((self.nz, self.ny, self.nx), dtype=self._dtype_real)

        def _empty_com():
            return jnp.zeros((self.nz, self.nl, self.nk), dtype=self._dtype_complex)

        self._state_shape_check(state)
        full_state = _state.FullPseudoSpectralState(
            state=state,
            ph=_empty_com(),
            u=_empty_real(),
            v=_empty_real(),
            uq=_empty_real(),
            vq=_empty_real(),
            dqhdt=_empty_com(),
        )
        full_state = self._invert(full_state)
        full_state = self._do_advection(full_state)
        full_state = self._do_friction(full_state)
        return full_state

    def get_updates(
        self, state: _state.PseudoSpectralState
    ) -> _state.PseudoSpectralState:
        """Get updates for time-stepping `state`.

        Parameters
        ----------
        state : PseudoSpectralState
            The state which will be time stepped using the computed updates.

        Returns
        -------
        PseudoSpectralState
            A new state object where each field corresponds to a
            time-stepping *update* to be applied.

        Note
        ----
        The object returned by this function has the same type of
        `state`, but contains *updates*. This is so the time-stepping
        can be done by mapping over the states and updates as JAX
        pytrees with the same structure.

        """
        full_state = self.get_full_state(state)
        return _state.PseudoSpectralState(
            qh=full_state.dqhdt,
        )

    def postprocess_state(
        self, state: _state.PseudoSpectralState
    ) -> _state.PseudoSpectralState:
        """Apply fixed filtering to `state`.

        This function should be called once on each new state after each time step.

        :class:`SteppedModel <pyqg_jax.steppers.SteppedModel>` handles
        this internally.

        Parameters
        ----------
        state : PseudoSpectralState
            The state to be filtered.

        Returns
        -------
        PseudoSpectralState
            The filtered state.
        """
        return state.update(qh=jnp.expand_dims(self.filtr, 0) * state.qh)

    def create_initial_state(self) -> _state.PseudoSpectralState:
        return _state.PseudoSpectralState(
            qh=jnp.zeros((self.nz, self.nl, self.nk), dtype=self._dtype_complex)
        )

    def _state_shape_check(self, state):
        if state.qh.ndim != 3:
            vmap_msg = " (use jax.vmap)" if state.qh.ndim > 3 else ""
            raise ValueError(
                f"state has {state.qh.ndim} dimensions, but should have 3{vmap_msg}."
            )
        correct_shape = (self.nz, self.nl, self.nk)
        if state.qh.shape != correct_shape:
            raise ValueError(
                f"state.qh has wrong shape {state.qh.shape}, should be {correct_shape}"
            )

    @property
    def nl(self):
        return self.ny

    @property
    def nk(self):
        return (self.nx // 2) + 1

    @property
    def kk(self):
        return jnp.zeros((self.nk,), dtype=self._dtype_real)

    @property
    def _ik(self):
        return jnp.zeros((self.nk,), dtype=self._dtype_complex)

    @property
    def ll(self):
        return jnp.zeros((self.nl,), dtype=self._dtype_real)

    @property
    def _il(self):
        return jnp.zeros((self.nl,), dtype=self._dtype_complex)

    @property
    def _k2l2(self):
        return jnp.zeros((self.nl, self.nk), dtype=self._dtype_real)

    # Friction
    @property
    def Ubg(self):
        return jnp.zeros((self.nk,), dtype=self._dtype_real)

    @property
    def filtr(self):
        raise NotImplementedError("define filtr property in subclass")

    @property
    def _ikQy(self):
        raise NotImplementedError("define _ikQy property in subclass")

    def _invert(
        self, state: _state.FullPseudoSpectralState
    ) -> _state.FullPseudoSpectralState:
        # Set ph to zero (skip, recompute fresh from sum below)
        # invert qh to find ph
        ph = self._apply_a_ph(state)
        # calculate spectral velocities
        uh = jnp.negative(jnp.expand_dims(self._il, (0, -1))) * ph
        vh = jnp.expand_dims(self._ik, (0, 1)) * ph
        # Update state values
        return state.update(ph=ph, uh=uh, vh=vh)

    def _do_advection(
        self, state: _state.FullPseudoSpectralState
    ) -> _state.FullPseudoSpectralState:
        # multiply to get advective flux in space
        uq = (state.u + jnp.expand_dims(self.Ubg[: self.nz], (-1, -2))) * state.q
        vq = state.v * state.q
        state = state.update(uq=uq, vq=vq)
        # spectral divergence
        dqhdt = jnp.negative(
            jnp.expand_dims(self._ik, (0, 1)) * state.uqh
            + jnp.expand_dims(self._il, (0, -1)) * state.vqh
            + jnp.expand_dims(self._ikQy[: self.nz], 1) * state.ph
        )
        return state.update(dqhdt=dqhdt)

    def _do_friction(
        self, state: _state.FullPseudoSpectralState
    ) -> _state.FullPseudoSpectralState:
        # Apply Beckman friction to lower layer tendency

        def compute_friction(state):
            k = operator.index(self.nz - 1)
            dqhdt = jnp.concatenate(
                [
                    state.dqhdt[:k],
                    jnp.expand_dims(
                        state.dqhdt[k] + (self.rek * self._k2l2 * state.ph[k]), 0
                    ),
                    state.dqhdt[(k + 1) :],
                ],
                axis=0,
            )
            return state.update(dqhdt=dqhdt)

        return jax.lax.cond(
            self.rek != 0,
            compute_friction,
            lambda state: state,
            state,
        )

    def _apply_a_ph(self, state):
        return jnp.zeros_like(state.ph)

    def _tree_flatten(self):
        static_attributes = (
            "nz",
            "ny",
            "nx",
            "precision",
            "_dtype_real",
            "_dtype_complex",
        )
        child_attributes = ("rek",)
        child_vals = [getattr(self, attr) for attr in child_attributes]
        static_vals = [getattr(self, attr) for attr in static_attributes]
        return child_vals, (child_attributes, static_vals, static_attributes)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        child_attributes, static_vals, static_attributes = aux_data
        obj = cls.__new__(cls)
        for name, val in itertools.chain(
            zip(child_attributes, children),
            zip(static_attributes, static_vals),
        ):
            setattr(obj, name, val)
        return obj
