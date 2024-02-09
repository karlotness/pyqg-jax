import pytest
import jax
import jax.numpy as jnp
import pyqg_jax


@pytest.mark.parametrize(
    "precision", [pyqg_jax.state.Precision.SINGLE, pyqg_jax.state.Precision.DOUBLE]
)
@pytest.mark.parametrize("nx,ny", [(16, 16), (17, 15)])
def test_basic_use(precision, nx, ny):
    stepped_model = pyqg_jax.steppers.SteppedModel(
        model=pyqg_jax.qg_model.QGModel(nx=nx, ny=ny, precision=precision),
        stepper=pyqg_jax.steppers.AB3Stepper(dt=14400.0),
    )
    init_state = stepped_model.create_initial_state(jax.random.key(0))
    num_steps = 5

    def step_model(carry, _x):
        next_step = stepped_model.step_model(carry)
        full_state = stepped_model.get_full_state(next_step)
        ke_val = pyqg_jax.diagnostics.ke_spec_vals(
            full_state=full_state,
            grid=stepped_model.model.get_grid(),
        )
        return next_step, ke_val

    @jax.jit
    def do_jax_steps(init_state):
        _, traj_ke_vals = jax.lax.scan(
            step_model,
            init_state,
            None,
            length=num_steps,
        )
        ispec = pyqg_jax.diagnostics.calc_ispec(
            jnp.mean(traj_ke_vals, axis=0), stepped_model.model.get_grid()
        )
        kr, keep = pyqg_jax.diagnostics.ispec_grid(stepped_model.model.get_grid())
        return ispec, kr, keep

    ispec, kr, keep = do_jax_steps(init_state)
    expected_type = jnp.dtype(
        jnp.float32 if precision == pyqg_jax.state.Precision.SINGLE else jnp.float64
    )
    expected_shape = (stepped_model.model.nz, max(nx // 2, ny // 2))
    assert keep < ispec.shape[-1]
    assert kr.shape == ispec.shape[-1:]
    assert ispec.shape == expected_shape
    assert ispec.dtype == expected_type
