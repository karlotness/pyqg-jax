# Copyright Karl Otness
# SPDX-License-Identifier: MIT


import functools
import jax.numpy as jnp
from .. import state as _state


def uv_parameterization(param_func):
    @functools.wraps(param_func)
    def wrapped_uv_param(full_state, param_aux, model, *args, **kwargs):
        (du, dv), param_aux = param_func(full_state, param_aux, model, *args, **kwargs)
        duh = _state._generic_rfftn(du)
        dvh = _state._generic_rfftn(dv)
        dqhdt = (
            full_state.dqhdt
            + ((-1 * jnp.expand_dims(model._il, (0, -1))) * duh)
            + (jnp.expand_dims(model._ik, (0, 1)) * dvh)
        )
        return full_state.update(dqhdt=dqhdt), param_aux

    return wrapped_uv_param


def q_parameterization(param_func):
    @functools.wraps(param_func)
    def wrapped_q_param(full_state, param_aux, model, *args, **kwargs):
        dq, param_aux = param_func(full_state, param_aux, model, *args, **kwargs)
        dqh = _state._generic_rfftn(dq)
        dqhdt = full_state.dqhdt + dqh
        return full_state.update(dqhdt=dqhdt)

    return wrapped_q_param
