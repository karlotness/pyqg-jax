# Copyright 2023 Karl Otness
# SPDX-License-Identifier: MIT


import math
import itertools
import pytest
import numpy as np
import jax
import jax.numpy as jnp
import pyqg_jax


@pytest.mark.parametrize("dt", [0.25, 1.0, 2.0])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_state_initialization(dt, dtype):
    stepper = pyqg_jax.steppers.EulerStepper(dt=dt)
    state_a = pyqg_jax.state.PseudoSpectralState(
        qh=jnp.ones((5, 3), dtype=dtype), _q_shape=(5, 4)
    )
    state_b = pyqg_jax.state.PseudoSpectralState(
        qh=jnp.arange(15, dtype=dtype).reshape((5, 3)), _q_shape=(5, 4)
    )
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
    assert final_state.state.qh.dtype == jnp.dtype(model.model.precision.dtype_complex)


@pytest.mark.skipif(
    not jax.config.jax_enable_x64, reason="need float64 enabled to test dtype conflict"
)
def test_steps_with_non_weak_dtype_dt():
    dt = 3600
    num_steps = 1000
    base_model = pyqg_jax.qg_model.QGModel(
        nx=16,
        ny=16,
        rek=5.787e-7,
        delta=0.25,
        beta=1.5e-11,
        precision=pyqg_jax.state.Precision.SINGLE,
    )
    init_state = base_model.create_initial_state(jax.random.key(0))

    def make_stepper(dt):
        @jax.jit
        def do_jax_steps(init_state):
            model = pyqg_jax.steppers.SteppedModel(
                model=base_model,
                stepper=pyqg_jax.steppers.EulerStepper(dt=dt),
            )
            final_state, _ = jax.lax.scan(
                lambda carry, _: (
                    model.step_model(carry),
                    None,
                ),
                model.initialize_stepper_state(init_state),
                None,
                length=num_steps,
            )
            return final_state

        return do_jax_steps

    final_steps = [
        make_stepper(dtv)(init_state)
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
