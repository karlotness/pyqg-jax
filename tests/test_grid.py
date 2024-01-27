import pytest
import jax
import jax.numpy as jnp
import pyqg_jax


def test_tree_flatten_roundtrip():
    Hi = jnp.array([0.1, 0.2, 0.3])
    grid = pyqg_jax.state.Grid(
        nz=3,
        ny=6,
        nx=7,
        L=10.0,
        W=15.0,
        Hi=Hi,
    )
    leaves, treedef = jax.tree_util.tree_flatten(grid)
    restored_grid = jax.tree_util.tree_unflatten(treedef, leaves)
    assert vars(grid).keys() == vars(restored_grid).keys()
    assert all(getattr(grid, k) is getattr(restored_grid, k) for k in vars(grid))


@pytest.mark.parametrize(
    "precision", [pyqg_jax.state.Precision.SINGLE, pyqg_jax.state.Precision.DOUBLE]
)
def test_kappa_shape_dtype(precision):
    Hi = jnp.array([0.1, 0.2, 0.3])
    grid = pyqg_jax.state.Grid(
        nz=3,
        ny=6,
        nx=7,
        L=10.0,
        W=15.0,
        Hi=Hi,
    )
    kappa = grid.get_kappa(dtype=precision)
    assert kappa.shape == grid.spectral_state_shape[1:]
    assert kappa.dtype == (
        jnp.dtype(jnp.float64)
        if precision == pyqg_jax.state.Precision.DOUBLE
        else jnp.dtype(jnp.float32)
    )


@pytest.mark.parametrize("precision", [jnp.float32, jnp.float64])
def test_kappa_direct_dtype(precision):
    Hi = jnp.array([0.1, 0.2, 0.3])
    grid = pyqg_jax.state.Grid(
        nz=3,
        ny=6,
        nx=7,
        L=10.0,
        W=15.0,
        Hi=Hi,
    )
    kappa = grid.get_kappa(precision)
    assert kappa.shape == grid.spectral_state_shape[1:]
    assert kappa.dtype == jnp.dtype(precision)
