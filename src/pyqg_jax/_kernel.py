# Copyright 2023 Karl Otness
# SPDX-License-Identifier: MIT


import operator
import abc
import jax
import jax.numpy as jnp
from . import _utils, state as _state


@_utils.register_pytree_class_attrs(
    children=["rek"],
    static_attrs=["nz", "ny", "nx", "precision"],
)
class PseudoSpectralKernel(abc.ABC):
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
        if not isinstance(self.precision, _state.Precision):
            raise ValueError(f"invalid choice for precision {self.precision}")

    def get_full_state(
        self, state: _state.PseudoSpectralState
    ) -> _state.FullPseudoSpectralState:
        def _empty_real():
            return jnp.zeros(self.get_grid().real_state_shape, dtype=self._dtype_real)

        def _empty_com():
            return jnp.zeros(
                self.get_grid().spectral_state_shape, dtype=self._dtype_complex
            )

        self._state_shape_check(state)
        full_state = _state.FullPseudoSpectralState(
            state=state,
            ph=_empty_com(),
            u=_empty_real(),
            v=_empty_real(),
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
            _q_shape=self.get_grid().real_state_shape[-2:],
        )

    def postprocess_state(
        self, state: _state.PseudoSpectralState
    ) -> _state.PseudoSpectralState:
        """Apply fixed filtering to `state`.

        This function should be called once on each new state after each time step.

        :class:`~pyqg_jax.steppers.SteppedModel` handles
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

    def create_initial_state(self, key=None) -> _state.PseudoSpectralState:
        return _state.PseudoSpectralState(
            qh=jnp.zeros(
                self.get_grid().spectral_state_shape, dtype=self._dtype_complex
            ),
            _q_shape=self.get_grid().real_state_shape[-2:],
        )

    @abc.abstractmethod
    def get_grid(self) -> _state.Grid:
        pass

    def _state_shape_check(self, state):
        corr_shape = self.get_grid().spectral_state_shape
        corr_dims = len(corr_shape)
        dims = state.qh.ndim
        if dims != corr_dims:
            vmap_msg = " (use jax.vmap)" if dims > corr_dims else ""
            raise ValueError(
                f"state has {dims} dimensions, but should have {corr_dims}{vmap_msg}"
            )
        if state.qh.shape != corr_shape:
            raise ValueError(
                f"state.qh has wrong shape {state.qh.shape}, should be {corr_shape}"
            )

    @property
    def _dtype_real(self):
        match self.precision:
            case _state.Precision.SINGLE:
                return jnp.float32
            case _state.Precision.DOUBLE:
                return jnp.float64
            case _:
                raise ValueError(f"invalid choice for precision {self.precision}")

    @property
    def _dtype_complex(self):
        match self.precision:
            case _state.Precision.SINGLE:
                return jnp.complex64
            case _state.Precision.DOUBLE:
                return jnp.complex128
            case _:
                raise ValueError(f"invalid choice for precision {self.precision}")

    @property
    def nl(self):
        return self.get_grid().nl

    @property
    def nk(self):
        return self.get_grid().nk

    @property
    @abc.abstractmethod
    def kk(self) -> jax.Array:
        pass

    @property
    def _ik(self):
        return 1j * self.kk

    @property
    @abc.abstractmethod
    def ll(self) -> jax.Array:
        pass

    @property
    def _il(self):
        return 1j * self.ll

    @property
    def _k2l2(self) -> jax.Array:
        return (jnp.expand_dims(self.kk, 0) ** 2) + (jnp.expand_dims(self.ll, -1) ** 2)

    # Friction
    @property
    @abc.abstractmethod
    def Ubg(self) -> jax.Array:
        pass

    @property
    @abc.abstractmethod
    def filtr(self) -> jax.Array:
        pass

    @property
    @abc.abstractmethod
    def Qy(self) -> jax.Array:
        pass

    @property
    def _ikQy(self):
        return 1j * (jnp.expand_dims(self.kk, 0) * jnp.expand_dims(self.Qy, -1))

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
        uqh = _state._generic_rfftn(uq)
        vqh = _state._generic_rfftn(vq)
        # spectral divergence
        dqhdt = jnp.negative(
            jnp.expand_dims(self._ik, (0, 1)) * uqh
            + jnp.expand_dims(self._il, (0, -1)) * vqh
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

    @abc.abstractmethod
    def _apply_a_ph(self, state: _state.FullPseudoSpectralState) -> jax.Array:
        pass

    def __repr__(self):
        return _utils.auto_repr(self)
