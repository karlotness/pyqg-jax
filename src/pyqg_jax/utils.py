# Copyright Karl Otness
# SPDX-License-Identifier: MIT


import dataclasses
import jax
from .kernel import PseudoSpectralKernelState


def make_basic_rollout(model, num_steps):

    def do_steps(carry_state, _y):
        next_state = model.step_forward(carry_state)
        return next_state, carry_state

    def gen_traj(first_step):
        _last_step, steps = jax.lax.scan(do_steps, first_step, None, length=num_steps)
        return steps

    return gen_traj


def make_gen_traj(model, num_steps):

    rollout_func = make_basic_rollout(model, num_steps)

    def gen_traj(rng):
        init = model.create_initial_state(rng)
        return rollout_func(init)

    return gen_traj


def slice_kernel_state(state, slicer):
    data_fields = frozenset(f.name for f in dataclasses.fields(PseudoSpectralKernelState))
    return PseudoSpectralKernelState(**{k: getattr(state, k)[slicer] for k in data_fields})
