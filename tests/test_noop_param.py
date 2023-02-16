import jax
import jax.numpy as jnp
import pyqg_jax


def test_match_final_step():
    jax_model = pyqg_jax.qg_model.QGModel(
        nx=64, precision=pyqg_jax.state.Precision.SINGLE
    )
    param_model = pyqg_jax.parameterizations.noop.apply_parameterization(jax_model)
    start_state = jax_model.create_initial_state(jax.random.PRNGKey(0))
    param_start_state = param_model.initialize_param_state(start_state)
    dt = 3600
    num_steps = 1000

    @jax.jit
    def do_jax_steps(model, init_state):
        stepper = pyqg_jax.steppers.AB3Stepper(dt=dt)
        stepped_model = pyqg_jax.steppers.SteppedModel(model=model, stepper=stepper)
        final_state, _ = jax.lax.scan(
            lambda carry, _: (
                stepped_model.step_model(carry),
                None,
            ),
            stepped_model.initialize_stepper_state(init_state),
            None,
            length=num_steps,
        )
        return final_state

    final_jax_state = do_jax_steps(jax_model, start_state)
    final_param_state = do_jax_steps(param_model, param_start_state)
    assert jnp.all(final_jax_state.state.q == final_param_state.state.model_state.q)
