# Copyright 2023 Karl Otness
# SPDX-License-Identifier: MIT


"""An implementation of :class:`pyqg.QGModel`."""

__all__ = ["QGModel"]


import jax
import jax.numpy as jnp
from . import _model, _utils, state as _state


@_utils.register_pytree_class_attrs(
    children=["beta", "rd", "delta", "U1", "U2", "H1"],
    static_attrs=[],
)
class QGModel(_model.Model):
    r"""Two-layer quasi-geostrophic model.

    See also :class:`pyqg.QGModel`.

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
        :math:`\mathrm{sec}^{-1} \mathrm{m}^{-1}`.

    rd : float, optional
        Deformation radius. Units: :math:`\mathrm{m}`.

    delta : float, optional
        Layer thickness ratio :math:`\mathrm{H1}/\mathrm{H2}`. Unitless.

    H1 : float, optional
        Layer thickness

    U1 : float, optional
        Upper layer flow. Units: :math:`\mathrm{m}\ \mathrm{sec}^{-1}`.

    U2 : float, optional
        Lower layer flow. Units: :math:`\mathrm{m}\ \mathrm{sec}^{-1}`.

    precision : Precision, optional
        Precision of model computation. Selects dtype of state values.

    Attributes
    ----------
    Ubg : jax.Array
        The background velocity for this model.

    Note
    ----
    This model internally uses 64-bit floating point values for part
    of its computation *regardless* of the chosen :class:`precision
    <pyqg_jax.state.Precision>`.

    Make sure that JAX has `64-bit precision enabled
    <https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision>`__.
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
        beta=1.5e-11,
        rd=15000.0,
        delta=0.25,
        H1=500,
        U1=0.025,
        U2=0.0,
        # Precision choice
        precision=_state.Precision.SINGLE,
    ):
        super().__init__(
            nz=2,
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
        self.rd = rd
        self.delta = delta
        self.U1 = U1
        self.U2 = U2
        self.H1 = H1

    def create_initial_state(self, key):
        """Create a new initial state with random initialization.

        Parameters
        ----------
        key : jax.random.key
            The PRNG state used as the random key for initialization.

        Returns
        -------
        PseudoSpectralState
            The new state with random initialization.
        """
        state = super().create_initial_state()
        # initial conditions (pv anomalies)
        rng_a, rng_b = jax.random.split(key, num=2)
        q1 = 1e-7 * jax.random.uniform(
            rng_a, shape=(self.ny, self.nx), dtype=self.precision.dtype_real
        ) + 1e-6 * (
            jnp.ones((self.ny, 1), dtype=self.precision.dtype_real)
            * jax.random.uniform(
                rng_b, shape=(1, self.nx), dtype=self.precision.dtype_real
            )
        )
        q2 = jnp.zeros_like(self.x, dtype=self.precision.dtype_real)
        state = state.update(q=jnp.stack([q1, q2], axis=-3))
        return state

    @property
    def U(self):
        return self.U1 - self.U2

    @property
    def Hi(self):
        return jnp.array(
            [self.H1, self.H1 / self.delta], dtype=self.precision.dtype_real
        )

    @property
    def H(self):
        return self.get_grid().H

    @property
    def Ubg(self):
        return jnp.array([self.U1, self.U2], dtype=self.precision.dtype_real)

    @property
    def F1(self):
        return self.rd**-2 / (1 + self.delta)

    @property
    def F2(self):
        return self.delta * self.F1

    @property
    def Qy1(self):
        return self.beta + self.F1 * (self.U1 - self.U2)

    @property
    def Qy2(self):
        return self.beta - self.F2 * (self.U1 - self.U2)

    @property
    def Qy(self):
        return jnp.array([self.Qy1, self.Qy2], dtype=self.precision.dtype_real)

    @property
    def ikQy1(self):
        return self.Qy1 * 1j * self.k

    @property
    def ikQy2(self):
        return self.Qy2 * 1j * self.k

    @property
    def ikQy(self):
        return jnp.stack([self.ikQy1, self.ikQy2], axis=-3)

    @property
    def ilQx(self):
        return 0

    @property
    def del1(self):
        return self.delta / (self.delta + 1)

    @property
    def del2(self):
        return (self.delta + 1) ** -1

    def _apply_a_ph(self, state):
        f64_wv2 = self.wv2.astype(jnp.float64)
        f64_f1 = jnp.float64(self.F1)
        f64_f2 = jnp.float64(self.F2)
        det = f64_wv2 * (f64_wv2 + f64_f1 + f64_f2)
        det1 = jnp.where(det == 0, 1, det)
        qh = state.qh.astype(jnp.complex128)
        ph = jnp.where(
            det == 0,
            0,
            jnp.stack(
                [
                    (-(f64_wv2 + f64_f2) * qh[0] - f64_f1 * qh[1]) / det1,
                    (-f64_f2 * qh[0] - (f64_wv2 + f64_f1) * qh[1]) / det1,
                ]
            ),
        )
        return ph.astype(self.precision.dtype_complex)
