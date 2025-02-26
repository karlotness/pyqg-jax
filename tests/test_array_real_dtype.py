# Copyright 2025 Karl Otness
# SPDX-License-Identifier: MIT


import warnings
import jax
import jax.numpy as jnp
import pytest
from pyqg_jax._utils import array_real_dtype


@pytest.mark.parametrize(
    "func, f32_dt, f64_dt",
    [
        (bool, "bool", "bool"),
        (int, "int32", "int64"),
        (jnp.int8, "int8", "int8"),
        (jnp.int16, "int16", "int16"),
        (jnp.int32, "int32", "int32"),
        (jnp.int64, "int32", "int64"),
        (jnp.uint8, "uint8", "uint8"),
        (jnp.uint16, "uint16", "uint16"),
        (jnp.uint32, "uint32", "uint32"),
        (jnp.uint64, "uint32", "uint64"),
        (float, "float32", "float64"),
        (jnp.float32, "float32", "float32"),
        (jnp.float64, "float32", "float64"),
        (jnp.complex64, "float32", "float32"),
        (jnp.complex128, "float32", "float64"),
    ],
)
def test_real_dtype(func, f32_dt, f64_dt):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        val = func(1)
    expected_dtype = jnp.dtype(f64_dt if jax.config.jax_enable_x64 else f32_dt)
    assert array_real_dtype(val) == expected_dtype
