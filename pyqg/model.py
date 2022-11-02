import json
import math
import jax
import jax.numpy as jnp
from . import kernel
from .kernel import DTYPE_REAL, DTYPE_COMPLEX


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
            filtr=None,
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
        self.g = g
        if f:
            self.f = f
            self.f2 = f**2

        # Initialize grid
        self.x, self.y = jnp.meshgrid(
            (jnp.arange(0.5, self.nx, 1.0, dtype=DTYPE_REAL) / self.nx) * self.L,
            (jnp.arange(0.5, self.ny, 1.0, dtype=DTYPE_REAL) / self.ny) * self.W
        )
        self.dk = 2 * jnp.pi / self.L
        self.dl = 2 * jnp.pi / self.W

        self.ll = self.dl * jnp.append(jnp.arange(0.0, self.nx / 2), jnp.arange(-self.nx / 2, 0.0)).astype(DTYPE_REAL)
        self._il = 1j * self.ll
        self._k2l2 = (jnp.expand_dims(self.kk, 0)**2) + (jnp.expand_dims(self.ll, -1)**2)

        self.kk = self.dk * jnp.arange(0.0, self.nk).astype(DTYPE_REAL)
        self._ik = 1j * self.kk
        self._k2l2 = (jnp.expand_dims(self.kk, 0)**2) + (jnp.expand_dims(self.ll, -1)**2)

        self.k, self.l = jnp.meshgrid(self.kk, self.ll)
        self.ik = 1j * self.k
        self.il = 1j * self.l
        self.dx = self.L / self.nx
        self.dy = self.W / self.ny
        self.M = self.nx * self.ny
        self.wv2 = self.k**2 + self.l**2
        self.wv = jnp.sqrt(self.wv2)
        iwv2 = self.wv2 != 0.0
        self.wv2i = jnp.where((self.wv2 != 0), jnp.power(self.wv2, -1), self.wv2)

        # initialize_background
        # NOT IMPLEMENTED

        # initialize forcing
        # NOT IMPLEMENTED

        # initialize filter
        cphi = 0.65 * jnp.pi
        wvx = jnp.sqrt((self.k * self.dx)**2 + (self.l * self.dy)**2)
        filtr = jnp.exp(-self.filterfac * (wvx - cphi)**4)
        self.filtr = jnp.where(wvx <= cphi, 1, filtr)

        # initialize time
        self.taveints = math.ceil(self.taveint / self.dt)

        # initialize inversion matrix
        # NOT IMPLEMENTED

        # initialize diagnostics
        # SKIPPED

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

    def param_json(self):
        params = {
            "nz": self.nz,
            "nx": self.nx,
            "ny": self.ny,
            "L": self.L,
            "W": self.W,
            "dt": self.dt,
            "tmax": self.tmax,
            "tavestart": self.tavestart,
            "taveint": self.taveint,
            "rek": self.rek,
            "filterfac": self.filterfac,
            "g": self.g,
        }
        if hasattr(self, "f"):
            params["f"] = self.f
        return json.dumps(params)

    @classmethod
    def from_param_json(cls, param_str):
        return cls(**json.loads(param_str))
