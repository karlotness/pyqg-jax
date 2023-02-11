# Copyright Karl Otness
# SPDX-License-Identifier: MIT


import operator
import typing
import itertools
import jax
import jax.numpy as jnp
from . import _utils, state as _state


def _zeros_property(
    shape: typing.Tuple[typing.Union[int, str], ...], dtype
) -> property:
    def get_zeros(self):
        new_shape = []
        for shape_elem in shape:
            if isinstance(shape_elem, str):
                new_shape.append(getattr(self, shape_elem))
            else:
                new_shape.append(shape_elem)
        if dtype == "real":
            true_dtype = self._dtype_real
        elif dtype == "complex":
            true_dtype = self._dtype_complex
        else:
            raise ValueError(f"unsupported dtype lookup {dtype}")
        return jnp.zeros(new_shape, dtype=true_dtype)

    return property(fget=get_zeros)


@_utils.register_pytree_node_class_private
class PseudoSpectralKernel:
    def __init__(
        self,
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
        self._precision = precision
        if self._precision == _state.Precision.SINGLE:
            self._dtype_real = jnp.float32
            self._dtype_complex = jnp.complex64
        elif self._precision == _state.Precision.DOUBLE:
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
        full_state = self.get_full_state(state)
        return _state.PseudoSpectralState(
            qh=full_state.dqhdt,
        )

    def postprocess_state(
        self, state: _state.PseudoSpectralState
    ) -> _state.PseudoSpectralState:
        return state.update(qh=jnp.expand_dims(self.filtr, 0) * state.qh)

    def create_initial_state(self) -> _state.PseudoSpectralState:
        return _state.PseudoSpectralState(
            qh=jnp.zeros((self.nz, self.nl, self.nk), dtype=self._dtype_complex)
        )

    @property
    def nl(self):
        return self.ny

    @property
    def nk(self):
        return (self.nx // 2) + 1

    kk = _zeros_property(("nk",), dtype="real")
    _ik = _zeros_property(("nk",), dtype="complex")
    ll = _zeros_property(("nl",), dtype="real")
    _il = _zeros_property(("nl",), dtype="complex")
    _k2l2 = _zeros_property(("nl", "nk"), dtype="real")

    # Friction
    Ubg = _zeros_property(("nk",), dtype="real")

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
        uh = (-1 * jnp.expand_dims(self._il, (0, -1))) * ph
        vh = jnp.expand_dims(self._ik, (0, 1)) * ph
        # transform to get u and v
        u = _state._generic_irfftn(uh)
        v = _state._generic_irfftn(vh)
        # Update state values
        return state.update(ph=ph, u=u, v=v)

    def _do_advection(
        self, state: _state.FullPseudoSpectralState
    ) -> _state.FullPseudoSpectralState:
        # multiply to get advective flux in space
        uq = (state.u + jnp.expand_dims(self.Ubg[: self.nz], (-1, -2))) * state.q
        vq = state.v * state.q
        # transform to get spectral advective flux
        uqh = _state._generic_rfftn(uq)
        vqh = _state._generic_rfftn(vq)
        # spectral divergence
        dqhdt = -1 * (
            jnp.expand_dims(self._ik, (0, 1)) * uqh
            + jnp.expand_dims(self._il, (0, -1)) * vqh
            + jnp.expand_dims(self._ikQy[: self.nz], 1) * state.ph
        )
        return state.update(uq=uq, vq=vq, dqhdt=dqhdt)

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
        a = jnp.zeros((self.nz, self.nz, self.nl, self.nk), dtype=self._dtype_complex)
        ph = jnp.sum(a * jnp.expand_dims(state.qh, 0), axis=1)
        return ph

    def _tree_flatten(self):
        static_attributes = (
            "nz",
            "ny",
            "nx",
            "_precision",
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
