# Copyright 2024 Karl Otness
# SPDX-License-Identifier: MIT


import warnings
import pytest
import numpy as np
import jax
import jax.numpy as jnp
import pyqg_jax


EDDY_ARGS = {
    "rek": 5.787e-7,
    "delta": 0.25,
    "beta": 1.5e-11,
}


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


def test_match_orig():
    pyqg = pytest.importorskip("pyqg")
    dt = 3600 * 5
    num_steps = 10
    jax_model = pyqg_jax.qg_model.QGModel(
        precision=pyqg_jax.state.Precision.DOUBLE,
        **EDDY_ARGS,
    )

    @jax.jit
    def compute_jax_ke(q):
        dummy_jax_state = jax_model.create_initial_state(jax.random.key(0))
        jax_state = dummy_jax_state.update(q=q)
        full_state = jax_model.get_full_state(jax_state)
        return jnp.max(pyqg_jax.diagnostics.total_ke(full_state, jax_model.get_grid()))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        orig_model = pyqg.QGModel(
            log_level=0,
            dt=dt,
            tmax=dt * num_steps,
            twrite=1,
            tavestart=0,
            taveint=dt,
            **EDDY_ARGS,
        )
    orig_model.q = np.asarray(
        jax_model.create_initial_state(jax.random.key(123)).q
    ).astype(np.float64)
    orig_kes = []
    jax_kes = []
    for _ in range(num_steps):
        orig_model._step_forward()
        orig_kes.append(orig_model._calc_ke())
        jax_kes.append(compute_jax_ke(orig_model.q.copy()))
    # Convert to NumPy array and slice to handle offsets
    # Offset because original PyQG uses ph value from the *previous* q
    # step while we use the value computed from the current *q* value
    orig_kes = np.array(orig_kes)[1:]
    jax_kes = np.array(jax_kes)[:-1]
    assert np.allclose(orig_kes, jax_kes)
