# Copyright Karl Otness
# SPDX-License-Identifier: MIT


__all__ = ["Precision", "PseudoSpectralState", "FullPseudoSpectralState"]


import enum
import dataclasses
import typing
import jax.numpy as jnp
from . import _utils


class Precision(enum.Enum):
    SINGLE = enum.auto()
    DOUBLE = enum.auto()


def _generic_rfftn(a):
    return jnp.fft.rfftn(a, axes=(-2, -1))


def _generic_irfftn(a):
    return jnp.fft.irfftn(a, axes=(-2, -1))


def _add_fft_properties(fields):
    def make_getter(field):
        def getter(self):
            return _generic_rfftn(getattr(self, field))

        return getter

    def add_properties(cls):
        for field in fields:
            setattr(
                cls,
                f"{field}h",
                property(fget=make_getter(field)),
            )
        return cls

    return add_properties


@_utils.register_pytree_dataclass
@dataclasses.dataclass(frozen=True)
class PseudoSpectralState:
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
            raise ValueError("found mismatched dtypes for q")
        return dataclasses.replace(self, qh=new_qh)

    def __repr__(self):
        q_summary = _utils.summarize_object(self.q)
        return f"PseudoSpectralState(q={q_summary})"


@_utils.register_pytree_dataclass
@dataclasses.dataclass(frozen=True)
@_add_fft_properties(["u", "v", "uq", "vq"])
class FullPseudoSpectralState:
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

    def update(self, **kwargs) -> "FullPseudoSpectralState":
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
                raise ValueError(f"found mismatched dtypes for {name}")
            if name in {"q", "qh"}:
                # Special handling for q and qh, make spectral and assign to state
                new_val = self.state.update(**{name: new_val})
                name = "state"
            elif name in {"uh", "vh", "uqh", "vqh"}:
                # Handle other spectral names, store as non-spectral
                new_val = _generic_irfftn(new_val)
                name = name[:-1]
            # Check that we don't have duplicate destinations
            if name in new_values:
                raise ValueError(f"duplicate updates for {name}")
            # Set up the actual replacement
            new_values[name] = new_val
        # Produce new object with processed values
        return dataclasses.replace(self, **new_values)

    def __repr__(self):
        q_summary = _utils.summarize_object(self.q)
        ph_summary = _utils.summarize_object(self.ph)
        u_summary = _utils.summarize_object(self.u)
        v_summary = _utils.summarize_object(self.v)
        uq_summary = _utils.summarize_object(self.uq)
        vq_summary = _utils.summarize_object(self.vq)
        dqhdt_summary = _utils.summarize_object(self.dqhdt)
        return f"""\
FullPseudoSpectralState(
  q={q_summary},
  ph={ph_summary},
  u={u_summary},
  v={v_summary},
  uq={uq_summary},
  vq={vq_summary},
  dqhdt={dqhdt_summary},
)"""
