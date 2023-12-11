import math
import warnings
import pytest
import pyqg
import jax
import jax.numpy as jnp
import numpy as np
import pyqg_jax


BT_PARAMS = {
    "L": 2 * np.pi,
    "nx": 64,
    "beta": 0.0,
    "H": 1.0,
    "rek": 0.0,
    "rd": None,
}


@pytest.mark.parametrize(
    "param",
    [
        # Model attributes
        "nz",
        "nx",
        "ny",
        "L",
        "W",
        "rek",
        "filterfac",
        "f",
        "g",
        "beta",
        "rd",
        "H",
        "U",
        # Properties from kernel
        "nl",
        "nk",
        "kk",
        "_ik",
        "ll",
        "_il",
        "_k2l2",
        "Ubg",
        "filtr",
        "_ikQy",
        # Properties from model
        "f2",
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
        # Properties from bt model
        "kd2",
        "Qy",
        "ikQy",
        "ilQx",
    ],
)
def test_default_parameters_match(param):
    jax_model = pyqg_jax.bt_model.BTModel(precision=pyqg_jax.state.Precision.DOUBLE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        orig_model = pyqg.BTModel(log_level=0)
    jax_param = getattr(jax_model, param)
    if param in {"f", "f2"}:
        assert jax_param is None
        assert not hasattr(orig_model, param)
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


@pytest.mark.parametrize(
    "precision", [pyqg_jax.state.Precision.SINGLE, pyqg_jax.state.Precision.DOUBLE]
)
def test_match_final_step(precision):
    jax_model = pyqg_jax.bt_model.BTModel(precision=precision, **BT_PARAMS)
    start_jax_state = jax_model.create_initial_state(jax.random.key(0))
    dt = 0.0025
    num_steps = 1000
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        orig_model = pyqg.BTModel(
            log_level=0,
            dt=dt,
            tmax=dt * num_steps,
            twrite=num_steps + 10,
            **BT_PARAMS,
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
            length=num_steps + 1,
        )
        return final_state

    final_jax_state = do_jax_steps(start_jax_state)
    orig_model.run()
    assert orig_model.tc == final_jax_state.tc
    abserr = jnp.abs(orig_model.q - final_jax_state.state.q)
    relerr = abserr / jnp.abs(orig_model.q)
    assert jnp.all(
        relerr < (0.025 if precision == pyqg_jax.state.Precision.SINGLE else 1e-10)
    )


def test_tree_flatten_roundtrip():
    model = pyqg_jax.bt_model.BTModel(**BT_PARAMS)
    leaves, treedef = jax.tree_util.tree_flatten(model)
    restored_model = jax.tree_util.tree_unflatten(treedef, leaves)
    assert vars(restored_model) == vars(model)
