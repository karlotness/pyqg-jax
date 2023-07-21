# Copyright Karl Otness
# SPDX-License-Identifier: MIT


"""An implementation of :class:`pyqg.BTModel`."""


__all__ = ["BTModel"]


import jax
import jax.numpy as jnp
from . import _model, _utils, state as _state


@_utils.register_pytree_class_attrs(
    children=["beta", "rd", "H", "U"],
    static_attrs=[],
)
class BTModel(_model.Model):
    r"""Single-layer (barotropic) quasigeostrophic model.

    See also :class:`pyqg.BTModel`.

    Parameters
    ----------
    nx : int, optional
        Number of grid points in the `x` direction.

    ny : int, optional
        Number of grid points in the `y` direction. Defaults to `nx`.

    L : float, optional
        Domain length in the `x` direction. Units: :math:`\mathrm{m}`.

    W : float, optional
        Domain length in the `y` direction. Defaults to `L`.
        Units: :math:`\mathrm{m}`.

    rek : float, optional
        Linear drag in lower layer. Units: :math:`\mathrm{sec}^{-1}`.

    filterfac : float, optional
        Amplitude of the spectral spherical filter.

    f : float, optional

    g : float, optional

    beta : float, optional
        Gradient of coriolis parameter. Units:
        :math:`\mathrm{m}^{-1}\ \mathrm{sec}^{-1}`.

    rd : float, optional
        Deformation radius. Units: :math:`\mathrm{m}`.

    H : float, optional

    U : float, optional
        Upper layer flow. Units: :math:`\mathrm{m}\ \mathrm{sec}^{-1}`.

    precision : Precision, optional
        Precision of model computation. Selects dtype of state values.
    """

    def __init__(
        self,
        *,
        # grid size parameters
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
        # Additional model parameters
        beta=0.0,
        rd=0.0,
        H=1.0,
        U=0.0,
        # Precision choice
        precision=_state.Precision.SINGLE,
    ):
        super().__init__(
            nz=1,
            nx=nx,
            ny=ny,
            L=L,
            W=W,
            rek=rek,
            filterfac=filterfac,
            f=f,
            g=g,
            precision=precision,
        )
        self.beta = beta
        if rd is None:
            rd = 0.0
        self.rd = rd
        self.H = H
        self.U = U

    @property
    def kd2(self):
        exp = jnp.where(self.rd != 0, -2, 1)
        return jnp.asarray(self.rd, dtype=self._dtype_real) ** exp

    def create_initial_state(self, key):
        """Create a new initial state with random initialization.

        Parameters
        ----------
        key : jax.random.PRNGKey
            The PRNG used as the random key for initialization.

        Returns
        -------
        PseudoSpectralState
            The new state with random initialization.
        """
        state = super().create_initial_state()
        # initial conditions (pv anomalies)
        q = 1e-3 * jax.random.uniform(
            key, shape=(self.nz, self.ny, self.nx), dtype=self._dtype_real
        )
        return state.update(q=q)

    @property
    def Ubg(self):
        return jnp.full(shape=(1,), fill_value=self.U, dtype=self._dtype_real)

    @property
    def Qy(self):
        return jnp.full(shape=(1,), fill_value=self.beta, dtype=self._dtype_real)

    @property
    def ikQy(self):
        return self.Qy * 1j * self.k

    @property
    def ilQx(self):
        return 0.0

    @property
    def _ikQy(self):
        return 1j * (jnp.expand_dims(self.kk, 0) * jnp.expand_dims(self.Qy, -1))

    def _apply_a_ph(self, state):
        return jnp.negative(state.qh * (self.wv2i + self.kd2))

    def __repr__(self):
        nx_summary = _utils.indent_repr(_utils.summarize_object(self.nx), 2)
        ny_summary = _utils.indent_repr(_utils.summarize_object(self.ny), 2)
        L_summary = _utils.indent_repr(_utils.summarize_object(self.L), 2)
        W_summary = _utils.indent_repr(_utils.summarize_object(self.W), 2)
        rek_summary = _utils.indent_repr(_utils.summarize_object(self.rek), 2)
        filterfac_summary = _utils.indent_repr(
            _utils.summarize_object(self.filterfac), 2
        )
        f_summary = _utils.indent_repr(_utils.summarize_object(self.f), 2)
        g_summary = _utils.indent_repr(_utils.summarize_object(self.g), 2)
        beta_summary = _utils.indent_repr(_utils.summarize_object(self.beta), 2)
        rd_summary = _utils.indent_repr(_utils.summarize_object(self.rd), 2)
        H_summary = _utils.indent_repr(_utils.summarize_object(self.H), 2)
        U_summary = _utils.indent_repr(_utils.summarize_object(self.U), 2)
        precision_summary = self.precision.name
        return f"""\
BTModel(
  nx={nx_summary},
  ny={ny_summary},
  L={L_summary},
  W={W_summary},
  rek={rek_summary},
  filterfac={filterfac_summary},
  f={f_summary},
  g={g_summary},
  beta={beta_summary},
  rd={rd_summary},
  H={H_summary},
  U={U_summary},
  precision={precision_summary},
)"""
