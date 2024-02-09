import math
import warnings
import pytest
import jax
import jax.numpy as jnp
import numpy as np
import pyqg
import pyqg.diagnostic_tools
import pyqg_jax
import pyqg_jax._spectral


@pytest.mark.parametrize("nx,ny", [(32, 32), (16, 16)])
@pytest.mark.parametrize("L,W", [(1e6, 1e6), (1e5, 1e6), (1e6, 1e5)])
@pytest.mark.parametrize("truncate", [True, False])
def test_get_plot_kr(nx, ny, L, W, truncate):
    jax_model = pyqg_jax.qg_model.QGModel(
        nx=nx, ny=ny, L=L, W=W, precision=pyqg_jax.state.Precision.DOUBLE
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        orig_model = pyqg.QGModel(nx=nx, ny=ny, L=L, W=W, log_level=0)
    # Compute JAX kr values
    jax_raw_kr, jax_keep = jax.jit(pyqg_jax._spectral.get_plot_kr)(
        jax_model.get_grid(), truncate=truncate
    )
    jax_kr = np.asarray(jax_raw_kr[:jax_keep])
    # Compute PyQG baselines
    test_sd = jax.eval_shape(
        lambda s: jnp.abs(jax_model.create_initial_state(s).qh), jax.random.key(0)
    )
    test_val = jnp.zeros(test_sd.shape[1:], dtype=test_sd.dtype)
    kr, _ = pyqg.diagnostic_tools.calc_ispec(
        orig_model, np.asarray(test_val), truncate=truncate
    )
    assert jax_kr.shape == kr.shape
    assert np.allclose(jax_kr, kr)


@pytest.mark.parametrize("nx,ny", [(32, 32), (16, 16)])
@pytest.mark.parametrize("L,W", [(1e3, 1e3), (1e5, 1e3)])
@pytest.mark.parametrize("averaging", [True, False])
@pytest.mark.parametrize("truncate", [True, False])
def test_calc_ispec(nx, ny, L, W, averaging, truncate):
    jax_model = pyqg_jax.qg_model.QGModel(
        nx=nx, ny=ny, L=L, W=W, precision=pyqg_jax.state.Precision.DOUBLE
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        orig_model = pyqg.QGModel(nx=nx, ny=ny, L=L, W=W, log_level=0)
    # Compute JAX kr values
    test_sd = jax.eval_shape(
        lambda s: jnp.abs(jax_model.create_initial_state(s).qh), jax.random.key(0)
    )
    test_val = jnp.linspace(
        0, 1, num=int(math.prod(test_sd.shape)), dtype=test_sd.dtype
    ).reshape(test_sd.shape)
    test_val = jnp.ones(test_sd.shape, dtype=test_sd.dtype)
    _, jax_keep = jax.jit(pyqg_jax._spectral.get_plot_kr)(
        jax_model.get_grid(), truncate=truncate
    )
    jax_raw_ispec = jax.jit(pyqg_jax._spectral.calc_ispec)(
        jnp.expand_dims(test_val[0], 0),
        grid=jax_model.get_grid(),
        averaging=averaging,
        truncate=truncate,
    )
    jax_ispec = jnp.squeeze(jax_raw_ispec[:, :jax_keep], axis=0)
    # Compute PyQG baselines
    _, orig_spec = pyqg.diagnostic_tools.calc_ispec(
        orig_model, np.asarray(test_val[0]), averaging=averaging, truncate=truncate
    )
    relerr = jnp.abs(jax_ispec - orig_spec) / orig_spec
    assert jax_ispec.shape == orig_spec.shape
    assert jnp.all(relerr < 1 / 3)
