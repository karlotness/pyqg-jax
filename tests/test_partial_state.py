# Copyright 2023 Karl Otness
# SPDX-License-Identifier: MIT


import re
import pytest
import jax
import jax.numpy as jnp
import pyqg_jax


def test_contains_spectral_leaf():
    state = pyqg_jax.state.PseudoSpectralState(
        qh=jnp.zeros((2, 16, 9), dtype=jnp.complex64),
        _q_shape=(16, 16),
    )
    leaves = jax.tree_util.tree_leaves(state)
    assert len(leaves) == 1
    assert leaves[0].shape == (2, 16, 9)
    assert leaves[0].dtype == jnp.complex64


def test_converts_to_spatial():
    state = pyqg_jax.state.PseudoSpectralState(
        qh=jnp.zeros((2, 16, 9), dtype=jnp.complex64),
        _q_shape=(16, 16),
    )
    assert state.q.dtype == jnp.float32
    assert state.q.shape == (2, 16, 16)
    assert jnp.allclose(state.q, 0)


def test_update_nothing():
    state = pyqg_jax.state.PseudoSpectralState(
        qh=jnp.zeros((2, 16, 9), dtype=jnp.complex64),
        _q_shape=(16, 16),
    )
    new_state = state.update()
    assert new_state is not state
    assert new_state.qh is state.qh
    assert new_state._q_shape is state._q_shape


def test_update_qh():
    state = pyqg_jax.state.PseudoSpectralState(
        qh=jnp.zeros((2, 16, 9), dtype=jnp.complex64),
        _q_shape=(16, 16),
    )
    new_state = state.update(qh=jnp.ones((2, 16, 9), dtype=jnp.complex64))
    assert jnp.allclose(state.qh, 0)
    assert jnp.allclose(new_state.qh, 1)


def test_update_q():
    state = pyqg_jax.state.PseudoSpectralState(
        qh=jnp.zeros((2, 16, 9), dtype=jnp.complex64),
        _q_shape=(16, 16),
    )
    new_state = state.update(q=jnp.ones((2, 16, 16), dtype=jnp.float32))
    assert jnp.allclose(state.qh, 0)
    assert not jnp.allclose(new_state.qh, 0)
    assert jnp.allclose(new_state.q, 1)


def test_update_rejects_duplicate_updates():
    state = pyqg_jax.state.PseudoSpectralState(
        qh=jnp.zeros((2, 16, 9), dtype=jnp.complex64),
        _q_shape=(16, 16),
    )
    with pytest.raises(ValueError, match="duplicate"):
        state.update(
            q=jnp.ones((2, 16, 16), dtype=jnp.float32),
            qh=jnp.zeros((2, 16, 9), dtype=jnp.complex64),
        )


@pytest.mark.parametrize("update_name", ["q", "qh"])
def test_update_rejects_wrong_shape(update_name):
    state = pyqg_jax.state.PseudoSpectralState(
        qh=jnp.zeros((2, 16, 9), dtype=jnp.complex64),
        _q_shape=(16, 16),
    )
    update_args = {
        update_name: jnp.ones(
            (2, 3, 4), dtype=jnp.complex64 if "h" in update_name else jnp.float32
        )
    }
    with pytest.raises(ValueError, match="shape") as exc_info:
        _ = state.update(**update_args)
    msg = exc_info.value.args[0]
    assert re.search(rf"\b{update_name}\b", msg)
    assert "(2, 3, 4)" in msg
    assert f"{getattr(state, update_name).shape}" in msg


@pytest.mark.parametrize("update_name", ["q", "qh"])
def test_update_rejects_wrong_dtype(update_name):
    state = pyqg_jax.state.PseudoSpectralState(
        qh=jnp.zeros((2, 16, 9), dtype=jnp.complex64),
        _q_shape=(16, 16),
    )
    update_args = {
        update_name: jnp.ones(
            (2, 16, 9) if "h" in update_name else (2, 16, 16),
            dtype=jnp.complex128 if "h" in update_name else jnp.float64,
        )
    }
    with pytest.raises(TypeError, match="dtype") as exc_info:
        _ = state.update(**update_args)
    msg = exc_info.value.args[0]
    assert re.search(rf"\b{update_name}\b", msg)
    assert f"{next(iter(update_args.values())).dtype}" in msg
    assert f"{getattr(state, update_name).dtype}" in msg


@pytest.mark.parametrize("extra_args", [("argx",), ("argx", "argy")])
def test_update_rejects_extra_args(extra_args):
    state = pyqg_jax.state.PseudoSpectralState(
        qh=jnp.zeros((2, 16, 9), dtype=jnp.complex64),
        _q_shape=(16, 16),
    )
    update_args = dict.fromkeys(extra_args, state.qh)
    with pytest.raises(ValueError, match="unknown") as exc_info:
        _ = state.update(**update_args)
    msg = exc_info.value.args[0]
    for arg in extra_args:
        assert arg in msg
    assert re.search(r"attributes\b" if len(extra_args) > 1 else r"attribute\b", msg)
