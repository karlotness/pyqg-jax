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
@pytest.mark.parametrize("nx,ny", [(16, 16), (17, 15)])
def test_basic_use(precision, nx, ny):
    stepped_model = pyqg_jax.steppers.SteppedModel(
        model=pyqg_jax.qg_model.QGModel(nx=nx, ny=ny, precision=precision),
        stepper=pyqg_jax.steppers.AB3Stepper(dt=14400.0),
    )
    init_state = stepped_model.create_initial_state(jax.random.key(0))
    num_steps = 5

    def step_model(carry, _x):
        next_step = stepped_model.step_model(carry)
        full_state = stepped_model.get_full_state(next_step)
        ke_val = pyqg_jax.diagnostics.ke_spec_vals(
            full_state=full_state,
            grid=stepped_model.model.get_grid(),
        )
        return next_step, ke_val

    @jax.jit
    def do_jax_steps(init_state):
        _, traj_ke_vals = jax.lax.scan(
            step_model,
            init_state,
            None,
            length=num_steps,
        )
        ispec = pyqg_jax.diagnostics.calc_ispec(
            jnp.mean(traj_ke_vals, axis=0), stepped_model.model.get_grid()
        )
        kr, keep = pyqg_jax.diagnostics.ispec_grid(stepped_model.model.get_grid())
        return ispec, kr, keep

    ispec, kr, keep = do_jax_steps(init_state)
    expected_type = precision.dtype_real
    expected_shape = (stepped_model.model.nz, max(nx // 2, ny // 2))
    assert keep < ispec.shape[-1]
    assert kr.shape == ispec.shape[-1:]
    assert ispec.shape == expected_shape
    assert ispec.dtype == expected_type


def test_match_orig():
    pyqg = pytest.importorskip("pyqg")
    pyqg_diagnostic_tools = pytest.importorskip("pyqg.diagnostic_tools")
    dt = 3600 * 5
    num_steps = 10
    jax_model = pyqg_jax.qg_model.QGModel(
        precision=pyqg_jax.state.Precision.DOUBLE,
        **EDDY_ARGS,
    )

    @jax.jit
    def compute_jax_ke_vals(raw_qs):
        ke_spec_vals = jax.vmap(
            lambda rq: pyqg_jax.diagnostics.ke_spec_vals(
                jax_model.get_full_state(
                    jax_model.create_initial_state(jax.random.key(0)).update(q=rq)
                ),
                jax_model.get_grid(),
            )
        )(raw_qs)
        return jnp.mean(ke_spec_vals, axis=0)

    @jax.jit
    def compute_jax_ke_spec(raw_qs):
        ispec = pyqg_jax.diagnostics.calc_ispec(
            compute_jax_ke_vals(raw_qs), jax_model.get_grid()
        )
        iso_k, keep = pyqg_jax.diagnostics.ispec_grid(jax_model.get_grid())
        return ispec, iso_k, keep

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        orig_model = pyqg.QGModel(
            log_level=0,
            dt=dt,
            tmax=dt * num_steps,
            twrite=1,
            tavestart=dt,
            taveint=dt,
            **EDDY_ARGS,
        )
    orig_model.q = np.asarray(
        jax_model.create_initial_state(jax.random.key(123)).q
    ).astype(np.float64)
    orig_states = []
    for _ in range(num_steps):
        orig_model._step_forward()
        orig_states.append(orig_model.q.copy())
    orig_states = np.asarray(orig_states)[:-1]
    jax_ke_vals = compute_jax_ke_vals(orig_states)
    jax_kespec, jax_kr, keep = compute_jax_ke_spec(orig_states)
    jax_kr = jax_kr[:keep]
    jax_kespec = jax_kespec[:, :keep]
    orig_ke_vals = orig_model.get_diagnostic("KEspec")
    assert jax_ke_vals.shape == orig_ke_vals.shape
    assert np.allclose(jax_ke_vals, orig_ke_vals)
    for level in range(jax_model.nz):
        orig_kr, orig_kespec = pyqg_diagnostic_tools.calc_ispec(
            orig_model, orig_ke_vals[level]
        )
        assert jax_kr.shape == orig_kr.shape
        assert np.allclose(orig_kr, jax_kr)
        assert orig_kespec.shape == jax_kespec.shape[1:]
        assert np.allclose(jax_kespec[level], orig_kespec, rtol=0.15)
