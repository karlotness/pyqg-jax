# Copyright Karl Otness
# SPDX-License-Identifier: MIT


import json
import math
import jax
import jax.numpy as jnp
from . import kernel
from .kernel import DTYPE_REAL, DTYPE_COMPLEX


def _grid_xy(nx, ny, L, W):
    x, y = jnp.meshgrid(
        (jnp.arange(0.5, nx, 1.0, dtype=DTYPE_REAL) / nx) * L,
        (jnp.arange(0.5, ny, 1.0, dtype=DTYPE_REAL) / ny) * W
    )
    return x, y

def _grid_kl(kk, ll):
    k, l = jnp.meshgrid(kk, ll)
    return k, l


@jax.tree_util.register_pytree_node_class
class Model(kernel.PseudoSpectralKernel):

    def __init__(
            self,
            # grid size parameters
            nz=1,
            nx=64,
            ny=None,
            L=1e6,
            W=None,
            # timestepping parameters
            dt=7200.0,
            tmax=1576800000.0,
            tavestart=315360000.0,
            taveint=86400.,
            # friction parameters
            rek=5.787e-7,
            filterfac=23.6,
            # constants
            f=None,
            g=9.81,
    ):
        if ny is None:
            ny = nx
        super().__init__(
            nz=nz,
            ny=ny,
            nx=nx,
            dt=dt,
            rek=rek,
        )

        if W is None:
            W = L
        self.L = L
        self.W = W
        self.tmax = tmax
        self.tavestart = tavestart
        self.taveint = taveint
        self.filterfac = filterfac
        self.taveints = math.ceil(self.taveint / self.dt)
        self.g = g
        self.f = f

    @property
    def f2(self):
        return self.f**2

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
        x, _y = _grid_xy(
            nx=self.nx,
            ny=self.ny,
            L=self.L,
            W=self.W,
        )
        return x

    @property
    def y(self):
        _x, y = _grid_xy(
            nx=self.nx,
            ny=self.ny,
            L=self.L,
            W=self.W,
        )
        return y

    @property
    def ll(self):
        return self.dl * jnp.append(jnp.arange(0.0, self.nx / 2), jnp.arange(-self.nx / 2, 0.0)).astype(DTYPE_REAL)

    @property
    def _il(self):
        return 1j * self.ll

    @property
    def _k2l2(self):
        return (jnp.expand_dims(self.kk, 0)**2) + (jnp.expand_dims(self.ll, -1)**2)

    @property
    def kk(self):
        return self.dk * jnp.arange(0.0, self.nk).astype(DTYPE_REAL)

    @property
    def _ik(self):
        return 1j * self.kk

    @property
    def _k2l2(self):
        return (jnp.expand_dims(self.kk, 0)**2) + (jnp.expand_dims(self.ll, -1)**2)

    @property
    def k(self):
        k, _l = _grid_kl(kk=self.kk, ll=self.ll)
        return k

    @property
    def l(self):
        _k, l = _grid_kl(kk=self.kk, ll=self.ll)
        return l

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
        wvx = jnp.sqrt((self.k * self.dx)**2 + (self.l * self.dy)**2)
        filtr = jnp.exp(-self.filterfac * (wvx - cphi)**4)
        return jnp.where(wvx <= cphi, 1, filtr)

    def do_external_forcing(self, state):
        return state

    def step_forward(self, state, uv_param_func=None, q_param_func=None):
        if uv_param_func is not None and q_param_func is not None:
            raise ValueError(f"Can only provide one parameterization function at a time!")
        state = self.invert(state)
        state = self.do_advection(state)
        state = self.do_friction(state)
        state = self.do_external_forcing(state)
        state = self.do_uv_subgrid_parameterization(state, uv_param_func)
        state = self.do_q_subgrid_parameterization(state, q_param_func)
        state = self.forward_timestep(state)
        return state

    def tree_flatten(self):
        attributes = ("nz", "nx", "ny", "L", "W", "dt", "tmax", "tavestart", "taveint", "rek", "filterfac", "f", "g")
        children = [getattr(self, attr) for attr in attributes]
        return children, attributes
