# Copyright Karl Otness, pyqg developers
# SPDX-License-Identifier: MIT


"""Parameterization as in `Smagorinsky (1963) <https://doi.org/10.1175/1520-0493(1963)091%3C0099:GCEWTP%3E2.3.CO;2>`__."""


__all__ = [
    "apply_parameterization",
    "param_func",
    "init_param_aux_func",
]


import jax
import jax.numpy as jnp
from . import _defs, _parameterized_model
from .. import state as _state


def apply_parameterization(model, *, constant=0.1):
    """Apply the Smagorinsky parameterization to `model`.

    See also: :class:`pyqg.parameterizations.Smagorinsky`

    Parameters
    ----------
    model
        The inner model to wrap in the parameterization.

    constant : float, optional
        Smagorinsky constant

    Returns
    -------
    ParameterizedModel
        `model` wrapped in the parameterization.
    """
    return _parameterized_model.ParameterizedModel(
        model=model,
        param_func=jax.tree_util.Partial(param_func, constant=constant),
        init_param_aux_func=init_param_aux_func,
    )


@_defs.uv_parameterization
def param_func(state, param_aux, model, *, constant=0.1):
    full_state = model.get_full_state(state)
    uh = full_state.uh
    vh = full_state.vh
    Sxx = _state._generic_irfftn(uh * model.ik, shape=model.get_grid().real_state_shape)
    Syy = _state._generic_irfftn(vh * model.il, shape=model.get_grid().real_state_shape)
    Sxy = 0.5 * _state._generic_irfftn(
        uh * model.il + vh * model.ik, shape=model.get_grid().real_state_shape
    )
    nu = (constant * model.dx) ** 2 * jnp.sqrt(2 * (Sxx**2 + Syy**2 + 2 * Sxy**2))
    nu_Sxxh = _state._generic_rfftn(nu * Sxx)
    nu_Sxyh = _state._generic_rfftn(nu * Sxy)
    nu_Syyh = _state._generic_rfftn(nu * Syy)
    du = 2 * (
        _state._generic_irfftn(
            nu_Sxxh * model.ik, shape=model.get_grid().real_state_shape
        )
        + _state._generic_irfftn(
            nu_Sxyh * model.il, shape=model.get_grid().real_state_shape
        )
    )
    dv = 2 * (
        _state._generic_irfftn(
            nu_Sxyh * model.ik, shape=model.get_grid().real_state_shape
        )
        + _state._generic_irfftn(
            nu_Syyh * model.il, shape=model.get_grid().real_state_shape
        )
    )
    return (du, dv), None


def init_param_aux_func(state, model):
    return None
