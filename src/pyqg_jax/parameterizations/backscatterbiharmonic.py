# Copyright Karl Otness, pyqg developers
# SPDX-License-Identifier: MIT


"""Parameterization as in
`Jansen and Held (2014) <https://doi.org/10.1016/j.ocemod.2014.06.002>`__
and
`Jansen et al. (2015) <https://doi.org/10.1016/j.ocemod.2015.05.007>`__.
"""


__all__ = [
    "apply_parameterization",
    "param_func",
    "init_param_aux_func",
]


import jax
import jax.numpy as jnp
from . import _defs, _parameterized_model
from .. import state as _state


def apply_parameterization(model, *, smag_constant=0.08, back_constant=0.99, eps=1e-32):
    """Apply the Backscatter Biharmonic parameterization to `model`.

    See also: :class:`pyqg.parameterizations.BackscatterBiharmonic`

    Parameters
    ----------
    model
        The inner model to wrap in the parameterization.

    smag_constant : float, optional
        :mod:`Smagorinsky <pyqg_jax.parameterizations.smagorinsky>`
        constant.

    back_constant : float, optional
        Backscatter constant.

    eps : float, optional
        Small constant to add to denominators to avoid division by
        zero errors.

    Returns
    -------
    ParameterizedModel
        `model` wrapped in the parameterization.
    """
    return _parameterized_model.ParameterizedModel(
        model=model,
        param_func=jax.tree_util.Partial(
            param_func,
            smag_constant=smag_constant,
            back_constant=back_constant,
            eps=eps,
        ),
        init_param_aux_func=init_param_aux_func,
    )


def _smagorinsky_just_viscosity(full_state, model, constant):
    uh = full_state.uh
    vh = full_state.vh
    Sxx = _state._generic_irfftn(uh * model.ik)
    Syy = _state._generic_irfftn(vh * model.il)
    Sxy = 0.5 * _state._generic_irfftn(uh * model.il + vh * model.ik)
    nu = (constant * model.dx) ** 2 * jnp.sqrt(2 * (Sxx**2 + Syy**2 + 2 * Sxy**2))
    return nu


@_defs.q_parameterization
def param_func(
    state, param_aux, model, *, smag_constant=0.08, back_constant=0.99, eps=1e-32
):
    full_state = model.get_full_state(state)
    lap = model.ik**2 + model.il**2
    psi = _state._generic_irfftn(full_state.ph)
    lap_lap_psi = _state._generic_irfftn(lap**2 * full_state.ph)
    dissipation = -_state._generic_irfftn(
        lap
        * _state._generic_rfftn(
            lap_lap_psi
            * model.dx**2
            * _smagorinsky_just_viscosity(
                full_state=full_state, model=model, constant=smag_constant
            )
        )
    )
    backscatter = (
        -back_constant
        * lap_lap_psi
        * (
            (jnp.sum(model.Hi * jnp.mean(psi * dissipation, axis=(-1, -2))))
            / (jnp.sum(model.Hi * jnp.mean(psi * lap_lap_psi, axis=(-1, -2))) + eps)
        )
    )
    dq = dissipation + backscatter
    return dq, None


def init_param_aux_func(state, model):
    return None
