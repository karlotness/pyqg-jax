# Copyright Karl Otness
# SPDX-License-Identifier: MIT


import functools
import jax.numpy as jnp
from .. import state as _state


def uv_parameterization(param_func):
    @functools.wraps(param_func)
    def wrapped_uv_param(state, param_aux, model, *args, **kwargs):
        (du, dv), param_aux = param_func(state, param_aux, model, *args, **kwargs)
        duh = _state._generic_rfftn(du)
        dvh = _state._generic_rfftn(dv)
        updates = model.get_updates(state)
        dqhdt = (
            updates.qh
            + ((-1 * jnp.expand_dims(model._il, (0, -1))) * duh)
            + (jnp.expand_dims(model._ik, (0, 1)) * dvh)
        )
        return updates.update(qh=dqhdt), param_aux

    return wrapped_uv_param


def q_parameterization(param_func):
    @functools.wraps(param_func)
    def wrapped_q_param(state, param_aux, model, *args, **kwargs):
        dq, param_aux = param_func(state, param_aux, model, *args, **kwargs)
        dqh = _state._generic_rfftn(dq)
        updates = model.get_updates(state)
        dqhdt = updates.qh + dqh
        return updates.update(qh=dqhdt)

    return wrapped_q_param
