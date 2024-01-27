# Copyright Karl Otness
# SPDX-License-Identifier: MIT


"""Base model state objects.

In `pyqg-jax` model states are separated into immutable objects to
more closely match the rest of JAX.

Models manipulate instances of :class:`PseudoSpectralState` which
stores only the `q` component from which other model variables are
derived.

To access these variables, the states can be expanded into a
:class:`FullPseudoSpectralState` with the remaining attributes.

.. note::
   When time-stepping in JAX (in particular with
   :func:`jax.lax.scan`), the state objects will have leading time
   dimensions in addition the their normal spatial dimensions. JAX
   will store these in "`structure-of-arrays
   <https://en.wikipedia.org/wiki/AoS_and_SoA>`__" style.

   To slice into states, consider combining
   :func:`jax.tree_util.tree_map` with a :term:`lambda
   <python:lambda>` or a combination of :func:`operator.itemgetter
   <python:operator.itemgetter>` and :class:`slice <python:slice>`.
"""


__all__ = ["Precision", "PseudoSpectralState", "FullPseudoSpectralState", "Grid"]


import enum
import dataclasses
import typing
import operator
import jax
import jax.numpy as jnp
from . import _utils


@enum.unique
class Precision(enum.Enum):
    """Enumeration for model precision levels.

    When constructing a base model, use values of this enumeration to
    select the numerical precision which should be used for the states
    and internal calculations.

    Double precision may be significantly slower, for example on GPUs.

    Attributes
    ----------
    SINGLE
        Single precision.

        Models will use :class:`jax.numpy.float32` and :class:`jax.numpy.complex64`.

    DOUBLE
        Double precision

        Models will use :class:`jax.numpy.float64` and :class:`jax.numpy.complex128`.

        Ensure that JAX has `64-bit precision enabled
        <https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision>`__.
    """

    SINGLE = enum.auto()
    DOUBLE = enum.auto()


def _generic_rfftn(a):
    return jnp.fft.rfftn(a, axes=(-2, -1))


def _generic_irfftn(a):
    return jnp.fft.irfftn(a, axes=(-2, -1))


@_utils.register_pytree_dataclass
@dataclasses.dataclass(frozen=True)
class PseudoSpectralState:
    """Core state evolved by a model instance.

    This is the innermost state type evolved by the models. This state
    can be expanded into a :class:`FullPseudoSpectralState` by calling
    methods on the models.

    Warning
    -------
    You should not construct this class yourself. Instead, you should
    retrieve instances from a model and if desired :meth:`update`
    their attributes.

    Attributes
    ----------
    q : jax.Array
        Potential vorticity in real space.

        This entry has shape :pycode:`(nz, ny, nx)`

    qh : jax.Array
        Potential vorticity in spectral space.

        This is the spectral form of :attr:`q`.
    """

    qh: jnp.ndarray

    @property
    def q(self) -> jnp.ndarray:
        return _generic_irfftn(self.qh)

    @typing.overload
    def update(self, *, q: jnp.ndarray) -> "PseudoSpectralState": ...

    @typing.overload
    def update(self, *, qh: jnp.ndarray) -> "PseudoSpectralState": ...

    def update(self, **kwargs) -> "PseudoSpectralState":
        """Replace the value stored in this state.

        This function produces a *new* state object, containing the
        replacement value.

        The keyword arguments may be either `q` or `qh` (not both),
        allowing the replacement value to be provided in spectral form
        if desired.

        The object this method is called on is not modified.

        Parameters
        ----------
        q : jax.Array
            Replacement value for :attr:`q`

        qh : jax.Array
            Replacement value for :attr:`qh`

        Returns
        -------
        PseudoSpectralState
            A copy of this object with the specified values replaced.

        Raises
        ------
        ValueError
            If the shape of the replacement does not match the
            existing shape or if duplicate updates are specified.

        TypeError
            If the dtype of the replacement does not match the existing type.
        """
        if len(kwargs) > 1:
            raise ValueError("duplicate updates for q (specified q and qh)")
        if "q" in kwargs:
            new_qh = _generic_rfftn(kwargs["q"])
        else:
            new_qh = kwargs["qh"]
        # Check that shape and dtypes match
        if self.qh.shape != new_qh.shape:
            raise ValueError("found mismatched shapes for q")
        if self.qh.dtype != new_qh.dtype:
            raise TypeError("found mismatched dtypes for q")
        return dataclasses.replace(self, qh=new_qh)

    def __repr__(self):
        qh_summary = _utils.summarize_object(self.qh)
        return f"PseudoSpectralState(qh={qh_summary})"


@_utils.register_pytree_dataclass
@dataclasses.dataclass(frozen=True)
class FullPseudoSpectralState:
    """Full state including calculated values expanded by a model.

    This is an expanded form of :class:`PseudoSpectralState` which
    includes additional attributes calculated as part of running one
    of the models.

    Warning
    -------
    You should not construct this class yourself. Instead, you should
    retrieve instances from a model and if desired :meth:`update`
    their attributes.

    Attributes
    ----------
    state : PseudoSpectralState
        Inner, partial state providing values for :attr:`q` and
        :attr:`qh`.

    q : jax.Array
        Potential vorticity in real space.

        Pass-through accessor for :attr:`state.q <PseudoSpectralState.q>`

    qh : jax.Array
        Potential vorticity in spectral space.

        Pass-through accessor for :attr:`state.qh <PseudoSpectralState.qh>`

    p : jax.Array
        Streamfunction in real space.

    ph : jax.Array
        Streamfunction in spectral space.

    u : jax.Array
        Zonal velocity anomaly in real space.

    uh : jax.Array
        Zonal velocity anomaly in spectral space.

    v : jax.Array
        Meridional velocity anomaly in real space.

    vh : jax.Array
        Meridional velocity anomaly in spectral space.

    dqhdt : jax.Array
        Spectral derivative with respect to time for :attr:`qh`.

        This value is the update applied to the model when time
        stepping.

    dqdt : jax.Array
        Real space version of :attr:`dqhdt`.

    Notes
    -----
    .. versionchanged:: 0.7.0
       Removed attributes `uq`, `vq`, `uqh`, and `vqh`.
    """

    state: PseudoSpectralState
    ph: jnp.ndarray
    u: jnp.ndarray
    v: jnp.ndarray
    dqhdt: jnp.ndarray

    @property
    def qh(self) -> jnp.ndarray:
        return self.state.qh

    @property
    def q(self) -> jnp.ndarray:
        return self.state.q

    @property
    def p(self) -> jnp.ndarray:
        return _generic_irfftn(self.ph)

    @property
    def uh(self) -> jnp.ndarray:
        return _generic_rfftn(self.u)

    @property
    def vh(self) -> jnp.ndarray:
        return _generic_rfftn(self.v)

    @property
    def dqdt(self) -> jnp.ndarray:
        return _generic_irfftn(self.dqhdt)

    def update(self, **kwargs) -> "FullPseudoSpectralState":
        """Replace values stored in this state.

        This function produces a *new* state object, with specified
        attributes replaced.

        The keyword arguments may specify any of this class's
        attributes *except* :attr:`state`, but must not apply multiple
        updates to the same attribute. That is, modifying both the
        spectral and real space values at the same time is not
        allowed.

        The object this method is called on is not modified.

        Parameters
        ----------
        q : jax.Array
            Replacement value for :attr:`q`. This also updates :attr:`state`.

        qh : jax.Array
            Replacement value for :attr:`qh`. This also updates :attr:`state`.

        p : jax.Array
            Replacement value for :attr:`p`.

        ph : jax.Array
            Replacement value for :attr:`ph`.

        u : jax.Array
            Replacement value for :attr:`u`.

        uh : jax.Array
            Replacement value for :attr:`uh`.

        v : jax.Array
            Replacement value for :attr:`v`.

        vh : jax.Array
            Replacement value for :attr:`vh`.

        dqhdt : jax.Array
            Replacement value for :attr:`dqhdt`.

        dqdt : jax.Array
            Replacement value for :attr:`dqdt`.

        Returns
        -------
        FullPseudoSpectralState
            A copy of this object with the specified values replaced.

        Raises
        ------
        ValueError
            If the shape of the replacement does not match the
            existing shape or if duplicate updates are specified.

        TypeError
            If the dtype of the replacement does not match the existing type.
        """
        new_values = {}
        if "state" in kwargs:
            raise ValueError(
                "do not update attribute 'state' directly, update individual fields"
            )
        for name, new_val in kwargs.items():
            # Check that shapes and dtypes match
            current_sd = jax.eval_shape(operator.attrgetter(name), self)
            if getattr(current_sd, "shape", None) != getattr(new_val, "shape", None):
                raise ValueError(f"found mismatched shapes for {name}")
            if getattr(current_sd, "dtype", None) != getattr(new_val, "dtype", None):
                raise TypeError(f"found mismatched dtypes for {name}")
            if name in {"q", "qh"}:
                # Special handling for q and qh, make spectral and assign to state
                new_val = self.state.update(**{name: new_val})
                name = "state"
            elif name in {"uh", "vh"}:
                # Handle other spectral names, store as non-spectral
                new_val = _generic_irfftn(new_val)
                name = name[:-1]
            elif name == "p":
                new_val = _generic_rfftn(new_val)
                name = "ph"
            elif name == "dqdt":
                new_val = _generic_rfftn(new_val)
                name = "dqhdt"
            # Check that we don't have duplicate destinations
            if name in new_values:
                raise ValueError(f"duplicate updates for {name}")
            # Set up the actual replacement
            new_values[name] = new_val
        # Produce new object with processed values
        return dataclasses.replace(self, **new_values)

    def __repr__(self):
        state_summary = _utils.indent_repr(_utils.summarize_object(self.state), 2)
        ph_summary = _utils.summarize_object(self.ph)
        u_summary = _utils.summarize_object(self.u)
        v_summary = _utils.summarize_object(self.v)
        dqhdt_summary = _utils.summarize_object(self.dqhdt)
        return f"""\
FullPseudoSpectralState(
  state={state_summary},
  ph={ph_summary},
  u={u_summary},
  v={v_summary},
  dqhdt={dqhdt_summary},
)"""


def _precision_to_real_dtype(
    precision: typing.Union[Precision, jnp.dtype], /
) -> jnp.dtype:
    if isinstance(precision, Precision):
        if precision == Precision.SINGLE:
            return jnp.float32
        elif precision == Precision.DOUBLE:
            return jnp.float64
        else:
            raise ValueError(f"unsupported precision {precision}")
    return precision


@_utils.register_pytree_class_attrs(
    children=["L", "W", "Hi"],
    static_attrs=["nz", "ny", "nx"],
)
class Grid:
    """Information on the spatial grid used by a model.

    The models in this package use an `Arakawa A-grid
    <https://en.wikipedia.org/wiki/Arakawa_grids#Arakawa_A-grid>`__
    for real space grids. This class also provides information on the
    shapes of arrays storing real and spectral values and the
    distances along each grid edge.

    Warning
    -------
    You should not construct this class yourself. Instead, you should
    retrieve instances from a model.

    Attributes
    ----------
    real_state_shape : tuple[int, int, int]
        Tuple specifying the shape of arrays for real space variables.

    spectral_state_shape : tuple[int, int, int]
        Tuple specifying the shape of arrays for spectral variables.

    nx : int
        Number of grid points in the x direction.

    ny : int
        Number of grid points in the y direction.

    nz : int
        Number of grid points in the z direction.

    L : float
        Domain length in the x direction.

    W : float
        Domain length in the y direction.

    H : float
        Domain length in the z direction.

        In most cases this may actually be a JAX float scalar or
        tracer.

    Hi : jax.Array
        The length of each layer in the z direction.

        This is a vector of length :attr:`nz` and whose entries sum to
        :attr:`H`.

    nk : int
        Number of spectral grid points in the k direction.

    nl : int
        Number of spectral grid points in the l direction.

    dx : float
        Space between grid points in the x direction.

    dy : float
        Space between grid points in the y direction.

    dk : float
        Spectral spacing in the k direction.

    dl : float
        Spectral spacing in the l direction.
    """

    def __init__(
        self,
        *,
        nz: int,
        ny: int,
        nx: int,
        L: float,
        W: float,
        Hi: jax.Array,
    ):
        self.nz = nz
        self.ny = ny
        self.nx = nx
        self.L = L
        self.W = W
        self.Hi = Hi
        if jnp.ndim(self.Hi) != 1:
            raise ValueError(
                f"Hi must be a 1D sequence, but had {jnp.ndim(self.Hi)} dimensions"
            )
        if jnp.shape(self.Hi)[0] != self.nz:
            raise ValueError(
                f"Hi must match nz ({self.nz}), but had {jnp.shape(self.Hi)[0]} entries"
            )

    @property
    def real_state_shape(self) -> tuple[int, int, int]:
        return (self.nz, self.ny, self.nx)

    @property
    def spectral_state_shape(self) -> tuple[int, int, int]:
        return (self.nz, self.nl, self.nk)

    @property
    def H(self) -> jax.Array:
        return self.Hi.sum()

    @property
    def nl(self) -> int:
        return self.ny

    @property
    def nk(self) -> int:
        return (self.nx // 2) + 1

    @property
    def dx(self):
        return self.L / self.nx

    @property
    def dy(self):
        return self.W / self.ny

    @property
    def dl(self):
        return 2 * jnp.pi / self.W

    @property
    def dk(self):
        return 2 * jnp.pi / self.L

    def get_kappa(self, dtype=Precision.SINGLE):
        """Information on the wavenumber at each spectral grid point.

        Parameters
        ----------
        precision : Precision, optional
            Precision for the wavenumber calculations.

        Returns
        -------
        jax.Array
            A two-dimensional grid of wavenumber values at the
            specified precision.

            These values have the shape of a spectral state (see
            :attr:`spectral_state_shape`) without the leading
            :attr:`nz` dimension.
        """
        real_dtype = _precision_to_real_dtype(dtype)
        k, l = jnp.meshgrid(
            jnp.fft.rfftfreq(
                self.nx, d=(self.L / (2 * jnp.pi * self.nx)), dtype=real_dtype
            ),
            jnp.fft.fftfreq(
                self.ny, d=(self.W / (2 * jnp.pi * self.ny)), dtype=real_dtype
            ),
        )
        return jnp.sqrt(k**2 + l**2)

    def __repr__(self):
        return _utils.auto_repr(self)
