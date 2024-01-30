import pytest
import jax
import jax.numpy as jnp
import pyqg_jax


@pytest.mark.parametrize(
    "precision", [pyqg_jax.state.Precision.SINGLE, pyqg_jax.state.Precision.DOUBLE]
)
@pytest.mark.parametrize("shape", [(32, 32), (17, 15)])
def test_basic_use(precision, shape):
    stepped_model = pyqg_jax.steppers.SteppedModel(
        model=pyqg_jax.qg_model.QGModel(
            nx=shape[0],
            ny=shape[1],
            precision=precision,
        ),
        stepper=pyqg_jax.steppers.AB3Stepper(dt=14400.0),
    )
    init_state = stepped_model.create_initial_state(jax.random.key(0))
    num_steps = 5

    def step_model(carry, _x):
        next_step = stepped_model.step_model(carry)
        full_state = stepped_model.get_full_state(next_step)
        total_ke = pyqg_jax.diagnostics.total_ke(
            full_state, stepped_model.model.get_grid()
        )
        return next_step, total_ke

    @jax.jit
    def do_jax_steps(init_state):
        _, traj_ke = jax.lax.scan(
            step_model,
            init_state,
            None,
            length=num_steps,
        )
        return traj_ke

    traj_ke = do_jax_steps(init_state)
    expected_type = jnp.dtype(
        jnp.float32 if precision == pyqg_jax.state.Precision.SINGLE else jnp.float64
    )
    assert traj_ke.ndim == 1
    assert traj_ke.shape[0] == num_steps
    assert traj_ke.dtype == expected_type
