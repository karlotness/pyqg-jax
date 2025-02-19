# Copyright 2023 Karl Otness
# SPDX-License-Identifier: MIT


"""An implementation of :class:`pyqg.QGModel`."""


__all__ = ["QGModel"]


import inspect
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
    <https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision>`__.
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
            rng_a, shape=(self.ny, self.nx), dtype=self._dtype_real
        ) + 1e-6 * (
            jnp.ones((self.ny, 1), dtype=self._dtype_real)
            * jax.random.uniform(rng_b, shape=(1, self.nx), dtype=self._dtype_real)
        )
        q2 = jnp.zeros_like(self.x, dtype=self._dtype_real)
        state = state.update(q=jnp.stack([q1, q2], axis=-3))
        return state

    @property
    def U(self):
        return self.U1 - self.U2

    @property
    def Hi(self):
        return jnp.array([self.H1, self.H1 / self.delta], dtype=self._dtype_real)

    @property
    def H(self):
        return self.get_grid().H

    @property
    def Ubg(self):
        return jnp.array([self.U1, self.U2], dtype=self._dtype_real)

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
        return jnp.array([self.Qy1, self.Qy2], dtype=self._dtype_real)

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
        f64_model = QGModel(
            **(
                {
                    arg: getattr(self, arg)
                    for arg in inspect.signature(QGModel).parameters
                }
                | {"precision": _state.Precision.DOUBLE}
            )
        )
        qh = jnp.moveaxis(state.qh, 0, -1)
        qh_orig_shape = qh.shape
        qh = qh.reshape((-1, 2))
        # Compute inversion matrix
        inv_mat2 = jnp.moveaxis(
            jnp.array(
                [
                    [
                        # 0, 0
                        -(f64_model.wv2 + f64_model.F1),
                        # 0, 1
                        jnp.full_like(f64_model.wv2, f64_model.F1),
                    ],
                    [
                        # 1, 0
                        jnp.full_like(f64_model.wv2, f64_model.F2),
                        # 1, 1
                        -(f64_model.wv2 + f64_model.F2),
                    ],
                ]
            ),
            (0, 1),
            (-2, -1),
        ).reshape((-1, 2, 2))[1:]
        # Solve the system for the tail
        ph_tail = jnp.squeeze(
            jnp.linalg.solve(
                inv_mat2, jnp.expand_dims(qh[1:].astype(f64_model._dtype_complex), -1)
            ).astype(self._dtype_complex),
            -1,
        )
        # Fill zeros for the head
        ph_head = jnp.expand_dims(jnp.zeros_like(qh[0]), 0)
        # Combine and return
        ph = jnp.concatenate([ph_head, ph_tail], axis=0).reshape(qh_orig_shape)
        return jnp.moveaxis(ph, -1, 0)
