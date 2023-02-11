# Copyright Karl Otness
# SPDX-License-Identifier: MIT


import jax.numpy as jnp
from . import _kernel, _utils, state


def _grid_xy(nx, ny, L, W, dtype_real):
    x, y = jnp.meshgrid(
        (jnp.arange(0.5, nx, 1.0, dtype=dtype_real) / nx) * L,
        (jnp.arange(0.5, ny, 1.0, dtype=dtype_real) / ny) * W,
    )
    return x, y


def _grid_kl(kk, ll):
    k, l = jnp.meshgrid(kk, ll)
    return k, l


@_utils.register_pytree_node_class_private
class Model(_kernel.PseudoSpectralKernel):
    def __init__(
        self,
        # grid size parameters
        nz=1,
        nx=64,
        ny=None,
        L=1e6,
        W=None,
        # friction parameters
        rek=5.787e-7,
        filterfac=23.6,
        # constants
        f=None,
        g=9.81,
        precision=state.Precision.SINGLE,
    ):
        if ny is None:
            ny = nx
        super().__init__(
            nz=nz,
            ny=ny,
            nx=nx,
            rek=rek,
            precision=precision,
        )

        if W is None:
            W = L
        self.L = L
        self.W = W
        self.filterfac = filterfac
        self.g = g
        self.f = f

    def get_full_state(
        self, state: state.PseudoSpectralState
    ) -> state.FullPseudoSpectralState:
        full_state = super().get_full_state(state)
        full_state = self._do_external_forcing(full_state)
        return full_state

    def _do_external_forcing(
        self, state: state.FullPseudoSpectralState
    ) -> state.FullPseudoSpectralState:
        return state

    @property
    def f2(self):
        if self.f is not None:
            return self.f**2
        else:
            return None

    @property
    def dk(self):
        return 2 * jnp.pi / self.L

    @property
    def dl(self):
        return 2 * jnp.pi / self.W

    @property
    def dx(self):
        return self.L / self.nx

    @property
    def dy(self):
        return self.W / self.ny

    @property
    def M(self):
        return self.nx * self.ny

    @property
    def x(self):
        return _grid_xy(
            nx=self.nx,
            ny=self.ny,
            L=self.L,
            W=self.W,
            dtype_real=self._dtype_real,
        )[0]

    @property
    def y(self):
        return _grid_xy(
            nx=self.nx,
            ny=self.ny,
            L=self.L,
            W=self.W,
            dtype_real=self._dtype_real,
        )[1]

    @property
    def ll(self):
        return self.dl * jnp.append(
            jnp.arange(0.0, self.nx / 2), jnp.arange(-self.nx / 2, 0.0)
        ).astype(self._dtype_real)

    @property
    def _il(self):
        return 1j * self.ll

    @property
    def _k2l2(self):
        return (jnp.expand_dims(self.kk, 0) ** 2) + (jnp.expand_dims(self.ll, -1) ** 2)

    @property
    def kk(self):
        return self.dk * jnp.arange(0.0, self.nk).astype(self._dtype_real)

    @property
    def _ik(self):
        return 1j * self.kk

    @property
    def k(self):
        return _grid_kl(kk=self.kk, ll=self.ll)[0]

    @property
    def l(self):
        return _grid_kl(kk=self.kk, ll=self.ll)[1]

    @property
    def ik(self):
        return 1j * self.k

    @property
    def il(self):
        return 1j * self.l

    @property
    def wv2(self):
        return self.k**2 + self.l**2

    @property
    def wv(self):
        return jnp.sqrt(self.wv2)

    @property
    def wv2i(self):
        return jnp.where((self.wv2 != 0), jnp.power(self.wv2, -1), self.wv2)

    @property
    def filtr(self):
        cphi = 0.65 * jnp.pi
        wvx = jnp.sqrt((self.k * self.dx) ** 2 + (self.l * self.dy) ** 2)
        filtr = jnp.exp(-self.filterfac * (wvx - cphi) ** 4)
        return jnp.where(wvx <= cphi, 1, filtr)

    def _tree_flatten(self):
        super_children, (
            super_attrs,
            super_static_vals,
            super_static_attrs,
        ) = super()._tree_flatten()
        new_attrs = ("L", "W", "filterfac", "g", "f")
        new_children = [getattr(self, name) for name in new_attrs]
        children = [*super_children, *new_children]
        new_attrs = (*super_attrs, *new_attrs)
        return children, (new_attrs, super_static_vals, super_static_attrs)
