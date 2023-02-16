import warnings
import math
import numpy as np
import jax
import jax.numpy as jnp
import pyqg
import pyqg_jax


def test_match_final_step():
    jax_model = pyqg_jax.parameterizations.zannabolton2020.apply_parameterization(
        pyqg_jax.qg_model.QGModel(nx=64, precision=pyqg_jax.state.Precision.DOUBLE)
    )
    start_jax_state = jax_model.create_initial_state(jax.random.PRNGKey(0))
    dt = 3600
    num_steps = 1000
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        orig_model = pyqg.QGModel(
            nx=64,
            log_level=0,
            dt=dt,
            tmax=dt * num_steps,
            twrite=num_steps + 10,
            parameterization=pyqg.parameterizations.ZannaBolton2020(),
        )
    orig_model.q = np.asarray(start_jax_state.model_state.q).copy().astype(np.float64)

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
    orig_model.run()
    assert orig_model.tc == final_jax_state.tc
    assert math.isclose(orig_model.t, final_jax_state.t)
    assert jnp.allclose(orig_model.q, final_jax_state.state.model_state.q)
    abserr = jnp.abs(orig_model.q - final_jax_state.state.model_state.q)
    relerr = abserr / jnp.abs(orig_model.q)
    assert jnp.all(relerr < 1e-10)
