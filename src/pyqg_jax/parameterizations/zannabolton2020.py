# Copyright Karl Otness, pyqg developers
# SPDX-License-Identifier: MIT


# Zanna and Bolton 2020: https://doi.org/10.1029/2020GL088376


__all__ = [
    "apply_parameterization",
    "param_func",
    "init_param_aux_func",
]


import functools
from . import _defs, _parametrized_model
from .. import state as _state


def apply_parameterization(model, kappa=-46761284):
    return _parametrized_model.ParametrizedModel(
        model=model,
        param_func=functools.partial(param_func, kappa=kappa),
        init_param_aux_func=init_param_aux_func,
    )


@_defs.uv_parameterization
def param_func(state, param_aux, model, kappa=-46761284):
    full_state = model.get_full_state(state)
    uh = full_state.uh
    vh = full_state.vh
    vx = _state._generic_irfftn(vh * model.ik)
    vy = _state._generic_irfftn(vh * model.il)
    ux = _state._generic_irfftn(uh * model.ik)
    uy = _state._generic_irfftn(uh * model.il)
    rel_vort = vx - uy
    shearing = vx + uy
    stretching = ux - vy
    rv_stretch = _state._generic_rfftn(rel_vort * stretching)
    rv_shear = _state._generic_rfftn(rel_vort * shearing)
    sum_sqs = _state._generic_rfftn(rel_vort**2 + shearing**2 + stretching**2) / 2
    du = kappa * _state._generic_irfftn(
        model.ik * (sum_sqs - rv_shear) + model.il * rv_stretch
    )
    dv = kappa * _state._generic_irfftn(
        model.il * (sum_sqs + rv_shear) + model.ik * rv_stretch
    )
    return (du, dv), None


def init_param_aux_func(state, model):
    return None
