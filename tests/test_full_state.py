# Copyright Karl Otness
# SPDX-License-Identifier: MIT


import pytest
import jax.numpy as jnp
import pyqg_jax


@pytest.fixture
def full_state():
    model = pyqg_jax.qg_model.QGModel(
        nx=16,
        ny=16,
        rek=0,
        precision=pyqg_jax.state.Precision.SINGLE,
    )
    small_state = pyqg_jax.state.PseudoSpectralState(
        qh=jnp.zeros((2, 16, 9), dtype=jnp.complex64),
        _q_shape=(16, 16),
    )
    return model.get_full_state(small_state)


def test_full_state_forwards_to_partial(full_state):
    assert full_state.qh is full_state.state.qh
    assert jnp.allclose(full_state.q, full_state.state.q)


@pytest.mark.parametrize("name", ["q", "qh"])
def test_full_state_forwards_update_to_partial(full_state, name):
    new_state = full_state.update(**{name: jnp.ones_like(getattr(full_state, name))})
    assert not jnp.allclose(full_state.qh, new_state.qh)
    assert new_state.qh is new_state.state.qh
    assert jnp.allclose(new_state.q, new_state.state.q)


def test_full_state_update_rejects_state(full_state):
    with pytest.raises(ValueError):
        _ = full_state.update(state=full_state.state)


@pytest.mark.parametrize("name", ["ph", "p", "u", "v", "dqhdt", "dqdt", "uh", "vh"])
def test_full_state_update(full_state, name):
    new_val = jnp.ones_like(getattr(full_state, name))
    new_state = full_state.update(**{name: new_val})
    assert jnp.allclose(getattr(new_state, name), 1)
    if name not in {"p", "uh", "vh", "uqh", "vqh", "dqdt"}:
        assert getattr(new_state, name) is new_val


@pytest.mark.parametrize("name", ["ph", "p", "u", "v", "dqhdt", "dqdt", "uh", "vh"])
def test_full_state_update_rejects_wrong_shape(full_state, name):
    new_val = jnp.ones_like(getattr(full_state, name)[..., 1:])
    with pytest.raises(ValueError, match="shape"):
        _ = full_state.update(**{name: new_val})


@pytest.mark.parametrize("name", ["ph", "p", "u", "v", "dqhdt", "dqdt", "uh", "vh"])
def test_full_state_update_rejects_wrong_dtype(full_state, name):
    if name.endswith("h") or name == "dqhdt":
        dtype = jnp.float32
    else:
        dtype = jnp.complex64
    new_val = jnp.ones_like(getattr(full_state, name), dtype=dtype)
    with pytest.raises(TypeError, match="dtype"):
        _ = full_state.update(**{name: new_val})


@pytest.mark.parametrize("name", ["p", "u", "v", "dqdt"])
def test_full_state_update_rejects_duplicate_updates(full_state, name):
    if name == "dqdt":
        spectral_name = "dqhdt"
    else:
        spectral_name = f"{name}h"
    new_vals = {
        name: getattr(full_state, name),
        spectral_name: getattr(full_state, spectral_name),
    }
    with pytest.raises(ValueError, match="duplicate"):
        _ = full_state.update(**new_vals)
