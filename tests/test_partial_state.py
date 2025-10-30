# Copyright 2023 Karl Otness
# SPDX-License-Identifier: MIT


import re
import pytest
import jax
import jax.numpy as jnp
import pyqg_jax


@pytest.fixture
def partial_state():
    def make_partial_state():
        model = pyqg_jax.qg_model.QGModel(
            nx=16,
            ny=16,
            rek=0,
            precision=pyqg_jax.state.Precision.SINGLE,
        )
        return model.create_initial_state(key=jax.random.key(0))

    return jax.tree_util.tree_map(
        lambda sd: jnp.zeros(sd.shape, dtype=sd.dtype),
        jax.eval_shape(make_partial_state),
    )


def test_contains_spectral_leaf(partial_state):
    leaves = jax.tree_util.tree_leaves(partial_state)
    assert len(leaves) == 1
    assert leaves[0].shape == (2, 16, 9)
    assert leaves[0].dtype == jnp.complex64


def test_converts_to_spatial(partial_state):
    assert partial_state.q.dtype == jnp.float32
    assert partial_state.q.shape == (2, 16, 16)
    assert jnp.allclose(partial_state.q, 0)


def test_update_nothing(partial_state):
    new_state = partial_state.update()
    assert new_state is not partial_state
    assert new_state.qh is partial_state.qh
    assert new_state._q_shape is partial_state._q_shape


def test_update_qh(partial_state):
    new_state = partial_state.update(qh=jnp.ones((2, 16, 9), dtype=jnp.complex64))
    assert jnp.allclose(partial_state.qh, 0)
    assert jnp.allclose(new_state.qh, 1)


def test_update_q(partial_state):
    new_state = partial_state.update(q=jnp.ones((2, 16, 16), dtype=jnp.float32))
    assert jnp.allclose(partial_state.qh, 0)
    assert not jnp.allclose(new_state.qh, 0)
    assert jnp.allclose(new_state.q, 1)


def test_update_rejects_duplicate_updates(partial_state):
    with pytest.raises(ValueError, match="duplicate"):
        partial_state.update(
            q=jnp.ones((2, 16, 16), dtype=jnp.float32),
            qh=jnp.zeros((2, 16, 9), dtype=jnp.complex64),
        )


@pytest.mark.parametrize("update_name", ["q", "qh"])
def test_update_rejects_wrong_shape(partial_state, update_name):
    update_args = {
        update_name: jnp.ones(
            (2, 3, 4), dtype=jnp.complex64 if "h" in update_name else jnp.float32
        )
    }
    with pytest.raises(ValueError, match="shape") as exc_info:
        _ = partial_state.update(**update_args)
    msg = exc_info.value.args[0]
    assert re.search(rf"\b{update_name}\b", msg)
    assert "(2, 3, 4)" in msg
    assert f"{getattr(partial_state, update_name).shape}" in msg


@pytest.mark.parametrize("update_name", ["q", "qh"])
def test_update_rejects_wrong_dtype(partial_state, update_name):
    update_args = {
        update_name: jnp.ones(
            (2, 16, 9) if "h" in update_name else (2, 16, 16),
            dtype=jnp.complex128 if "h" in update_name else jnp.float64,
        )
    }
    with pytest.raises(TypeError, match="dtype") as exc_info:
        _ = partial_state.update(**update_args)
    msg = exc_info.value.args[0]
    assert re.search(rf"\b{update_name}\b", msg)
    assert f"{next(iter(update_args.values())).dtype}" in msg
    assert f"{getattr(partial_state, update_name).dtype}" in msg


@pytest.mark.parametrize("extra_args", [("argx",), ("argx", "argy")])
def test_update_rejects_extra_args(partial_state, extra_args):
    update_args = dict.fromkeys(extra_args, partial_state.qh)
    with pytest.raises(ValueError, match="unknown") as exc_info:
        _ = partial_state.update(**update_args)
    msg = exc_info.value.args[0]
    for arg in extra_args:
        assert arg in msg
    assert re.search(r"attributes\b" if len(extra_args) > 1 else r"attribute\b", msg)
