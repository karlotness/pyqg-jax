# Copyright 2023 Karl Otness, pyqg developers
# SPDX-License-Identifier: MIT


"""Parameterization as in `Zanna and Bolton (2020) <https://doi.org/10.1029/2020GL088376>`__."""


__all__ = [
    "apply_parameterization",
    "param_func",
    "init_param_aux_func",
]


import jax
from . import _defs, _parameterized_model
from .. import state as _state


def apply_parameterization(model, *, kappa=-46761284):
    r"""Apply the Zanna-Bolton parameterization to `model`.

    See also: :class:`pyqg.parameterizations.ZannaBolton2020`

    Parameters
    ----------
    model
        The inner model to wrap in the parameterization.

    kappa : float, optional
        Scaling constant with units :math:`\mathrm{m}^{-2}`.

    Returns
    -------
    ParameterizedModel
        `model` wrapped in the parameterization.
    """
    return _parameterized_model.ParameterizedModel(
        model=model,
        param_func=jax.tree_util.Partial(param_func, kappa=kappa),
        init_param_aux_func=init_param_aux_func,
    )


@_defs.uv_parameterization
def param_func(state, param_aux, model, *, kappa=-46761284):
    full_state = model.get_full_state(state)
    uh = full_state.uh
    vh = full_state.vh
    vx = _state._generic_irfftn(vh * model.ik, shape=model.get_grid().real_state_shape)
    vy = _state._generic_irfftn(vh * model.il, shape=model.get_grid().real_state_shape)
    ux = _state._generic_irfftn(uh * model.ik, shape=model.get_grid().real_state_shape)
    uy = _state._generic_irfftn(uh * model.il, shape=model.get_grid().real_state_shape)
    rel_vort = vx - uy
    shearing = vx + uy
    stretching = ux - vy
    rv_stretch = _state._generic_rfftn(rel_vort * stretching)
    rv_shear = _state._generic_rfftn(rel_vort * shearing)
    sum_sqs = _state._generic_rfftn(rel_vort**2 + shearing**2 + stretching**2) / 2
    du = kappa * _state._generic_irfftn(
        model.ik * (sum_sqs - rv_shear) + model.il * rv_stretch,
        shape=model.get_grid().real_state_shape,
    )
    dv = kappa * _state._generic_irfftn(
        model.il * (sum_sqs + rv_shear) + model.ik * rv_stretch,
        shape=model.get_grid().real_state_shape,
    )
    return (du, dv), None


def init_param_aux_func(state, model):
    return None
