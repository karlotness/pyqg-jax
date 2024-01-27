# Copyright Karl Otness
# SPDX-License-Identifier: MIT


"""An implementation of :class:`pyqg.SQGModel`.
"""


__all__ = ["SQGModel"]


import jax
import jax.numpy as jnp
from . import _model, _utils, state as _state


@_utils.register_pytree_class_attrs(
    children=["beta", "Nb", "f_0", "H", "U"],
    static_attrs=[],
)
class SQGModel(_model.Model):
    r"""Surface quasigeostrophic model.

    See also :class:`pyqg.SQGModel`.

    .. versionadded:: 0.7.0

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

    Nb : float, optional
        Buoyancy frequency. Units: :math:`\mathrm{sec}^{-1}`

    f_0: float, optional

    H : float, optional

    U : float, optional
        Background zonal flow. Units: :math:`\mathrm{m}\ \mathrm{sec}^{-1}`.

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
        Nb=1.0,
        f_0=1.0,
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
        self.Nb = Nb
        self.f_0 = f_0
        self.H = H
        self.U = U

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
        return (
            super()
            .create_initial_state()
            .update(
                q=1e-3
                * jax.random.uniform(
                    key, shape=(self.nz, self.ny, self.nx), dtype=self._dtype_real
                ),
            )
        )

    def get_grid(self) -> _state.Grid:
        """Retrieve information on the model grid.

        Returns
        -------
        Grid
            A grid instance with attributes giving information on the
            spatial and spectral model grids.
        """
        Hi = jnp.full(shape=(1,), fill_value=self.H, dtype=self._dtype_real)
        return _state.Grid(
            nz=self.nz,
            ny=self.ny,
            nx=self.nx,
            L=self.L,
            W=self.W,
            Hi=Hi,
        )

    @property
    def Hi(self):
        return jnp.full(shape=(1,), fill_value=self.H, dtype=self._dtype_real)

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
    def Ubg(self):
        return jnp.full(shape=(1,), fill_value=self.U, dtype=self._dtype_real)

    def _apply_a_ph(self, state):
        return (self.f_0 / self.Nb) * jnp.sqrt(self.wv2i) * state.qh
