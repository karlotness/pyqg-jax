# Copyright Karl Otness
# SPDX-License-Identifier: MIT


import pytest
import jax
import jax.numpy as jnp
import pyqg_jax


def test_contains_spectral_leaf():
    state = pyqg_jax.state.PseudoSpectralState(
        qh=jnp.zeros((2, 16, 9), dtype=jnp.complex64)
    )
    leaves = jax.tree_util.tree_leaves(state)
    assert len(leaves) == 1
    assert leaves[0].shape == (2, 16, 9)
    assert leaves[0].dtype == jnp.complex64


def test_converts_to_spatial():
    state = pyqg_jax.state.PseudoSpectralState(
        qh=jnp.zeros((2, 16, 9), dtype=jnp.complex64)
    )
    assert state.q.dtype == jnp.float32
    assert state.q.shape == (2, 16, 16)
    assert jnp.allclose(state.q, 0)


def test_update_qh():
    state = pyqg_jax.state.PseudoSpectralState(
        qh=jnp.zeros((2, 16, 9), dtype=jnp.complex64)
    )
    new_state = state.update(qh=jnp.ones((2, 16, 9), dtype=jnp.complex64))
    assert jnp.allclose(state.qh, 0)
    assert jnp.allclose(new_state.qh, 1)


def test_update_q():
    state = pyqg_jax.state.PseudoSpectralState(
        qh=jnp.zeros((2, 16, 9), dtype=jnp.complex64)
    )
    new_state = state.update(q=jnp.ones((2, 16, 16), dtype=jnp.float32))
    assert jnp.allclose(state.qh, 0)
    assert not jnp.allclose(new_state.qh, 0)
    assert jnp.allclose(new_state.q, 1)


def test_update_rejects_duplicate_updates():
    state = pyqg_jax.state.PseudoSpectralState(
        qh=jnp.zeros((2, 16, 9), dtype=jnp.complex64)
    )
    with pytest.raises(ValueError, match="duplicate"):
        state.update(
            q=jnp.ones((2, 16, 16), dtype=jnp.float32),
            qh=jnp.zeros((2, 16, 9), dtype=jnp.complex64),
        )


@pytest.mark.parametrize("update_name", ["q", "qh"])
def test_update_rejects_wrong_shape(update_name):
    state = pyqg_jax.state.PseudoSpectralState(
        qh=jnp.zeros((2, 16, 9), dtype=jnp.complex64)
    )
    update_args = {
        update_name: jnp.ones(
            (2, 3, 4), dtype=jnp.complex64 if "h" in update_name else jnp.float32
        )
    }
    with pytest.raises(ValueError, match="shape"):
        _ = state.update(**update_args)


@pytest.mark.parametrize("update_name", ["q", "qh"])
def test_update_rejects_wrong_dtype(update_name):
    state = pyqg_jax.state.PseudoSpectralState(
        qh=jnp.zeros((2, 16, 9), dtype=jnp.complex64)
    )
    update_args = {
        update_name: jnp.ones(
            (2, 16, 9) if "h" in update_name else (2, 16, 16),
            dtype=jnp.complex128 if "h" in update_name else jnp.float64,
        )
    }
    with pytest.raises(ValueError, match="dtype"):
        _ = state.update(**update_args)
