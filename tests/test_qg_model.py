import math
import warnings
import pytest
import pyqg
import jax
import jax.numpy as jnp
import numpy as np
import pyqg_jax


@pytest.mark.parametrize(
    "param",
    [
        "nz",
        "ny",
        "nx",
        "rek",
        "L",
        "W",
        "filterfac",
        "f",
        "g",
        "beta",
        "rd",
        "delta",
        "H1",
        "U1",
        "U2",
        "Hi",
        "kk",
        "_ik",
        "ll",
        "_il",
        "_k2l2",
        "filtr",
        "Ubg",
        "Qy",
        "_ikQy",
        "nl",
        "nk",
        "dk",
        "dl",
        "dx",
        "dy",
        "M",
        "x",
        "y",
        "k",
        "l",
        "ik",
        "il",
        "wv2",
        "wv",
        "wv2i",
    ],
)
def test_default_parameters_match(param):
    jax_model = pyqg_jax.qg_model.QGModel(precision=pyqg_jax.state.Precision.DOUBLE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        orig_model = pyqg.QGModel(log_level=0)
    jax_param = getattr(jax_model, param)
    if param == "f":
        assert jax_param is None
        assert not hasattr(orig_model, "f")
        return
    elif param == "H1":
        orig_param = orig_model.Hi[0]
    else:
        orig_param = getattr(orig_model, param)
    if isinstance(orig_param, np.ndarray):
        np_jax_param = np.asarray(jax_param)
        assert jax_param.shape == orig_param.shape
        assert np_jax_param.dtype == orig_param.dtype
        assert np.allclose(np_jax_param, orig_param)
    elif isinstance(orig_param, float):
        assert math.isclose(orig_param, float(jax_param))
    elif isinstance(orig_param, int):
        assert jax_param == orig_param
    else:
        orig_param = np.asarray(orig_param)
        np_jax_param = np.asarray(jax_param)
        assert jax_param.shape == orig_param.shape
        assert np_jax_param.dtype == orig_param.dtype
        assert np.allclose(np_jax_param, orig_param)


def test_jit_scan():
    num_steps = 5

    @jax.jit
    def scan_steps(model, stepper, init_state):
        final_state, _ = jax.lax.scan(
            lambda carry, _x: (
                stepper.apply_updates(carry, model.get_updates(carry.state)),
                None,
            ),
            stepper.initialize_stepper_state(init_state),
            None,
            length=num_steps,
        )
        return final_state

    model = pyqg_jax.qg_model.QGModel()
    stepper = pyqg_jax.steppers.AB3Stepper(dt=3600)
    init_state = model.create_initial_state(jax.random.PRNGKey(0))
    jit_final_state = scan_steps(model=model, stepper=stepper, init_state=init_state)
    manual_final_state = stepper.initialize_stepper_state(init_state)
    for _ in range(num_steps):
        manual_final_state = stepper.apply_updates(
            manual_final_state, model.get_updates(manual_final_state.state)
        )
    assert jit_final_state.tc == manual_final_state.tc
    assert math.isclose(jit_final_state.t, manual_final_state.t)
    assert jnp.allclose(jit_final_state.state.q, manual_final_state.state.q)
