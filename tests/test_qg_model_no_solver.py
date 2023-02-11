import warnings
import math
import numpy as np
import jax
import jax.numpy as jnp
import pyqg
import pyqg_jax


EDDY_ARGS = {
    "rek": 5.787e-7,
    "delta": 0.25,
    "beta": 1.5e-11,
}


class QGModelNoSolver(pyqg_jax.qg_model.QGModel):
    def _compute_nosolver_inversion_matrix(self):
        # Inverse determinant
        det_inv = self.wv2 * (self.wv2 + self.F1 + self.F2)
        det_inv = jnp.where(
            det_inv != 0,
            det_inv**-1,
            0,
        )
        return jnp.stack(
            [
                jnp.stack(
                    [-(self.wv2 + self.F2) * det_inv, -self.F1 * det_inv], axis=0
                ),
                jnp.stack(
                    [-self.F2 * det_inv, -(self.wv2 + self.F1) * det_inv], axis=0
                ),
            ],
            axis=0,
        ).astype(self._dtype_complex)

    def _apply_a_ph(self, state):
        # Return ph from a * qh
        a = self._compute_nosolver_inversion_matrix()
        return jnp.einsum("baji,aji->bji", a, state.qh)


def test_compute_inversion_matrix():
    jax_model = QGModelNoSolver(precision=pyqg_jax.state.Precision.DOUBLE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        orig_model = pyqg.QGModel(log_level=0)
    np_jax_a = np.asarray(jax_model._compute_nosolver_inversion_matrix())
    assert orig_model.a.shape == np_jax_a.shape
    assert orig_model.a.dtype == np_jax_a.dtype
    assert np.allclose(orig_model.a, np_jax_a)


def test_final_step_matches():
    jax_model = QGModelNoSolver(precision=pyqg_jax.state.Precision.DOUBLE, **EDDY_ARGS)
    start_jax_state = jax_model.create_initial_state(jax.random.PRNGKey(0))
    dt = 3600
    num_steps = 500
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        orig_model = pyqg.QGModel(
            log_level=0,
            dt=dt,
            tmax=dt * num_steps,
            twrite=num_steps + 10,
            **EDDY_ARGS,
        )
    orig_model.q = np.asarray(start_jax_state.q).copy().astype(np.float64)

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
    abserr = jnp.abs(orig_model.q - final_jax_state.state.q)
    relerr = abserr / jnp.abs(orig_model.q)
    assert jnp.all(relerr < 1e-10)
