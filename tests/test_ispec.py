import pytest
import jax
import jax.numpy as jnp
import pyqg_jax
import pyqg_jax._spectral


@pytest.mark.parametrize("nx,ny", [(32, 32), (16, 16), (16, 32), (15, 13)])
@pytest.mark.parametrize("L,W", [(1e6, 1e6), (1e5, 1e6), (1e6, 1e5)])
def test_ispec_grid(nx, ny, L, W):
    jax_model = pyqg_jax.qg_model.QGModel(nx=nx, ny=ny, L=L, W=W)
    iso_k, keep = jax.jit(pyqg_jax.diagnostics.ispec_grid)(jax_model.get_grid())
    assert keep < iso_k.shape[0]


def test_ispec_zero():
    jax_model = pyqg_jax.qg_model.QGModel(nx=16, L=1e6, W=1e6)
    test_sd = jax.eval_shape(
        lambda s: jnp.abs(jax_model.create_initial_state(s).qh), jax.random.key(0)
    )
    test_val = jnp.zeros(test_sd.shape, dtype=test_sd.dtype)
    result = jax.jit(pyqg_jax.diagnostics.calc_ispec)(test_val, jax_model.get_grid())
    assert jnp.allclose(result, 0)
