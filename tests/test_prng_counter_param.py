import jax
import pyqg_jax


def test_match_final_step():
    jax_model = pyqg_jax.parameterizations.ParametrizedModel(
        model=pyqg_jax.qg_model.QGModel(
            nx=64, precision=pyqg_jax.state.Precision.DOUBLE
        ),
        param_func=lambda state, param_aux, model: (
            state,
            (jax.random.split(param_aux[0], 2)[1], param_aux[1] + 1),
        ),
        init_param_aux_func=lambda state, model: (jax.random.PRNGKey(0), 0),
    )
    start_jax_state = jax_model.create_initial_state(jax.random.PRNGKey(0))
    dt = 3600
    num_steps = 1000

    @jax.jit
    def do_jax_steps(init_state):
        stepper = pyqg_jax.steppers.AB3Stepper(dt=dt)
        stepped_model = pyqg_jax.steppers.SteppedModel(model=jax_model, stepper=stepper)
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

    final_jax_state = do_jax_steps(start_jax_state)
    assert final_jax_state.state.param_aux.value[1] == num_steps
