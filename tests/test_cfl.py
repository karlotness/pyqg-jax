# Copyright 2024 Karl Otness
# SPDX-License-Identifier: MIT


import pytest
import jax
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
        cfl = pyqg_jax.diagnostics.cfl(
            full_state=full_state,
            grid=stepped_model.model.get_grid(),
            ubg=stepped_model.model.Ubg,
            dt=stepped_model.stepper.dt,
        )
        return next_step, cfl

    @jax.jit
    def do_jax_steps(init_state):
        _, traj_cfl = jax.lax.scan(
            step_model,
            init_state,
            None,
            length=num_steps,
        )
        return traj_cfl

    traj_cfl = do_jax_steps(init_state)
    expected_type = precision.dtype_real
    expected_shape = (num_steps, *stepped_model.model.get_grid().real_state_shape)
    assert traj_cfl.shape == expected_shape
    assert traj_cfl.dtype == expected_type
