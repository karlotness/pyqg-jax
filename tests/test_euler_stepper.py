import math
import pytest
import numpy as np
import jax
import jax.numpy as jnp
import pyqg_jax


@pytest.mark.parametrize("dt", [0.25, 1.0, 2.0])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_state_initialization(dt, dtype):
    stepper = pyqg_jax.steppers.EulerStepper(dt=dt)
    state_a = pyqg_jax.state.PseudoSpectralState(qh=jnp.ones(5, dtype=dtype))
    state_b = pyqg_jax.state.PseudoSpectralState(qh=jnp.arange(5, dtype=dtype))
    wrapped_state = stepper.initialize_stepper_state(state_a)
    updated_state = stepper.apply_updates(wrapped_state, state_b)
    diff = updated_state.state.qh - wrapped_state.state.qh
    assert isinstance(wrapped_state, pyqg_jax.steppers.StepperState)
    assert isinstance(updated_state, pyqg_jax.steppers.StepperState)
    assert updated_state.state.qh.dtype == jnp.dtype(dtype)
    assert np.allclose(diff, state_b.qh * dt)


@pytest.mark.parametrize(
    "precision", [pyqg_jax.state.Precision.SINGLE, pyqg_jax.state.Precision.DOUBLE]
)
def test_step_qh(precision):
    num_steps = 5
    dt = 3600
    model = pyqg_jax.steppers.SteppedModel(
        model=pyqg_jax.qg_model.QGModel(
            nx=16, ny=16, rek=5.787e-7, delta=0.25, beta=1.5e-11, precision=precision
        ),
        stepper=pyqg_jax.steppers.EulerStepper(dt=dt),
    )

    @jax.jit
    def do_jax_steps(model, init_state):
        final_state, _ = jax.lax.scan(
            lambda carry, _: (
                model.step_model(carry),
                None,
            ),
            init_state,
            None,
            length=num_steps,
        )
        return final_state

    final_state = do_jax_steps(model, model.create_initial_state(jax.random.key(0)))
    assert final_state.tc == num_steps
    assert math.isclose(final_state.t.item(), num_steps * dt)
    assert final_state.state.qh.dtype == jnp.dtype(model.model._dtype_complex)
