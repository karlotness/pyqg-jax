# Copyright Karl Otness, pyqg developers
# SPDX-License-Identifier: MIT


# Smagorinsky 1963: https://doi.org/10.1175/1520-0493(1963)091%3C0099:GCEWTP%3E2.3.CO;2


__all__ = [
    "apply_parameterization",
    "param_func",
    "init_param_aux_func",
]


import functools
import jax.numpy as jnp
from . import _defs, _parametrized_model
from .. import state as _state


def apply_parameterization(model, constant=0.1):
    return _parametrized_model.ParametrizedModel(
        model=model,
        param_func=functools.partial(param_func, constant=constant),
        init_param_aux_func=init_param_aux_func,
    )


@_defs.uv_parameterization
def param_func(state, param_aux, model, constant=0.1):
    full_state = model.get_full_state(state)
    uh = full_state.uh
    vh = full_state.vh
    Sxx = _state._generic_irfftn(uh * model.ik)
    Syy = _state._generic_irfftn(vh * model.il)
    Sxy = 0.5 * _state._generic_irfftn(uh * model.il + vh * model.ik)
    nu = (constant * model.dx) ** 2 * jnp.sqrt(2 * (Sxx**2 + Syy**2 + 2 * Sxy**2))
    nu_Sxxh = _state._generic_rfftn(nu * Sxx)
    nu_Sxyh = _state._generic_rfftn(nu * Sxy)
    nu_Syyh = _state._generic_rfftn(nu * Syy)
    du = 2 * (
        _state._generic_irfftn(nu_Sxxh * model.ik)
        + _state._generic_irfftn(nu_Sxyh * model.il)
    )
    dv = 2 * (
        _state._generic_irfftn(nu_Sxyh * model.ik)
        + _state._generic_irfftn(nu_Syyh * model.il)
    )
    return (du, dv), None


def init_param_aux_func(state, model):
    return None
