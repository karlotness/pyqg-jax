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


__all__ = ["Precision", "PseudoSpectralState", "FullPseudoSpectralState"]


import enum
import dataclasses
import typing
import jax.numpy as jnp
from . import _utils


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
    def update(self, *, q: jnp.ndarray) -> "PseudoSpectralState":
        ...

    @typing.overload
    def update(self, *, qh: jnp.ndarray) -> "PseudoSpectralState":
        ...

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

    uq : jax.Array

    uqh : jax.Array

    vq : jax.Array

    vqh : jax.Array

    dqhdt : jax.Array
        Spectral derivative with respect to time for :attr:`qh`.

        This value is the update applied to the model when time
        stepping.
    """

    state: PseudoSpectralState
    ph: jnp.ndarray
    u: jnp.ndarray
    v: jnp.ndarray
    uq: jnp.ndarray
    vq: jnp.ndarray
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
    def uqh(self) -> jnp.ndarray:
        return _generic_rfftn(self.uq)

    @property
    def vqh(self) -> jnp.ndarray:
        return _generic_rfftn(self.vq)

    def update(self, **kwargs) -> "FullPseudoSpectralState":
        """Replace values stored in this state.

        This function produces a *new* state object, with specified
        attributes replaced.

        The keyword arguments may specify any of this class's
        attributes, but must not apply multiple updates to the same
        attribute. That is, modifying both the spectral and real space
        values at the same time is not allowed.

        The object this method is called on is not modified.

        Parameters
        ----------
        state : PseudoSpectralState
            Replacement value for :attr:`state`

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

        uq : jax.Array
            Replacement value for :attr:`uq`.

        uqh : jax.Array
            Replacement value for :attr:`uqh`.

        vq : jax.Array
            Replacement value for :attr:`vq`.

        vqh : jax.Array
            Replacement value for :attr:`vqh`.

        dqhdt : jax.Array
            Replacement value for :attr:`dqhdt`.

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
            if getattr(getattr(self, name), "shape", None) != getattr(
                new_val, "shape", None
            ):
                raise ValueError(f"found mismatched shapes for {name}")
            if getattr(getattr(self, name), "dtype", None) != getattr(
                new_val, "dtype", None
            ):
                raise TypeError(f"found mismatched dtypes for {name}")
            if name in {"q", "qh"}:
                # Special handling for q and qh, make spectral and assign to state
                new_val = self.state.update(**{name: new_val})
                name = "state"
            elif name in {"uh", "vh", "uqh", "vqh"}:
                # Handle other spectral names, store as non-spectral
                new_val = _generic_irfftn(new_val)
                name = name[:-1]
            elif name == "p":
                new_val = _generic_rfftn(new_val)
                name = "ph"
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
        uq_summary = _utils.summarize_object(self.uq)
        vq_summary = _utils.summarize_object(self.vq)
        dqhdt_summary = _utils.summarize_object(self.dqhdt)
        return f"""\
FullPseudoSpectralState(
  state={state_summary},
  ph={ph_summary},
  u={u_summary},
  v={v_summary},
  uq={uq_summary},
  vq={vq_summary},
  dqhdt={dqhdt_summary},
)"""
