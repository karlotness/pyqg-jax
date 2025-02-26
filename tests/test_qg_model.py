# Copyright 2023 Karl Otness
# SPDX-License-Identifier: MIT


import math
import warnings
import itertools
import pytest
import jax
import jax.numpy as jnp
import numpy as np
import pyqg_jax


EDDY_ARGS = {
    "rek": 5.787e-7,
    "delta": 0.25,
    "beta": 1.5e-11,
}


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
    pyqg = pytest.importorskip("pyqg")
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


@pytest.mark.parametrize(
    "precision", [pyqg_jax.state.Precision.SINGLE, pyqg_jax.state.Precision.DOUBLE]
)
def test_match_final_step(precision):
    pyqg = pytest.importorskip("pyqg")
    jax_model = pyqg_jax.qg_model.QGModel(precision=precision, **EDDY_ARGS)
    start_jax_state = jax_model.create_initial_state(jax.random.key(0))
    dt = 3600
    num_steps = 1000
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
    assert jnp.allclose(orig_model.q, final_jax_state.state.q)
    abserr = jnp.abs(orig_model.q - final_jax_state.state.q)
    relerr = abserr / jnp.abs(orig_model.q)
    assert jnp.all(
        relerr < (1e-2 if precision == pyqg_jax.state.Precision.SINGLE else 1e-10)
    )


@pytest.mark.parametrize("nx,ny", [(16, 32), (17, 15)])
def test_rectangular_stepping(nx, ny):
    jax_model = pyqg_jax.steppers.SteppedModel(
        model=pyqg_jax.qg_model.QGModel(
            nx=nx,
            ny=ny,
            **EDDY_ARGS,
        ),
        stepper=pyqg_jax.steppers.AB3Stepper(dt=3600.0),
    )
    init_state = jax_model.create_initial_state(jax.random.key(0))

    @jax.jit
    def do_jax_steps(init_state):
        final_state, _ = jax.lax.scan(
            lambda carry, _: (
                jax_model.step_model(carry),
                None,
            ),
            init_state,
            None,
            length=3,
        )
        return final_state.state.q

    final_state = do_jax_steps(init_state)
    assert jnp.all(jnp.isfinite(final_state))


def test_tree_flatten_roundtrip():
    model = pyqg_jax.qg_model.QGModel(**EDDY_ARGS)
    leaves, treedef = jax.tree_util.tree_flatten(model)
    restored_model = jax.tree_util.tree_unflatten(treedef, leaves)
    assert vars(restored_model) == vars(model)


@pytest.mark.skipif(
    not jax.config.jax_enable_x64, reason="need float64 enabled to test dtype conflict"
)
def test_steps_with_non_weak_dtype_dt():
    jax_model = pyqg_jax.qg_model.QGModel(
        precision=pyqg_jax.state.Precision.SINGLE, **EDDY_ARGS
    )
    start_jax_state = jax_model.create_initial_state(jax.random.key(0))
    dt = 3600
    num_steps = 1000

    def make_stepper(dt):
        @jax.jit
        def do_jax_steps(init_state):
            stepper = pyqg_jax.steppers.AB3Stepper(dt=dt)
            stepped_model = pyqg_jax.steppers.SteppedModel(
                model=jax_model, stepper=stepper
            )
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

        return do_jax_steps

    final_steps = [
        make_stepper(dtv)(start_jax_state)
        for dtv in (dt, float(dt), jnp.float32(dt), jnp.float64(dt))
    ]
    assert final_steps[0].state.q.dtype == jnp.dtype(jnp.float32)
    for state_a, state_b in itertools.pairwise(final_steps):
        assert state_a.tc == state_b.tc
        assert math.isclose(state_a.t, state_b.t)
        assert state_a.state.q.dtype == state_b.state.q.dtype
        assert jnp.allclose(state_a.state.q, state_b.state.q)
        abserr = jnp.abs(state_a.state.q - state_b.state.q)
        relerr = abserr / jnp.abs(state_a.state.q)
        assert jnp.all(relerr < 1e-5)
