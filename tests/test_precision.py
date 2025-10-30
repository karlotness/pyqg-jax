# Copyright 2025 Karl Otness
# SPDX-License-Identifier: MIT


import warnings
import pytest
import jax.numpy as jnp
import pyqg_jax
from pyqg_jax._utils import array_real_dtype


@pytest.mark.parametrize("val", list(pyqg_jax.state.Precision))
def test_values_have_proper_type(val):
    assert type(val) is pyqg_jax.state.Precision


@pytest.mark.parametrize("val", list(pyqg_jax.state.Precision))
def test_values_are_int(val):
    assert isinstance(val.value, int)


@pytest.mark.parametrize("val", list(pyqg_jax.state.Precision))
def test_dtypes_are_instances(val):
    assert isinstance(val.dtype_real, jnp.dtype)
    assert isinstance(val.dtype_complex, jnp.dtype)


@pytest.mark.parametrize("val", list(pyqg_jax.state.Precision))
def test_real_dtypes_are_real(val):
    assert jnp.issubdtype(val.dtype_real, jnp.floating)


@pytest.mark.parametrize("val", list(pyqg_jax.state.Precision))
def test_complex_dtypes_are_complex(val):
    assert jnp.issubdtype(val.dtype_complex, jnp.complexfloating)


@pytest.mark.parametrize("val", list(pyqg_jax.state.Precision))
def test_complex_and_real_dtypes_match(val):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        complex_sample = jnp.ones(1, dtype=val.dtype_complex)
        real_sample = jnp.ones(1, dtype=val.dtype_real)
    assert array_real_dtype(complex_sample) == real_sample.dtype
