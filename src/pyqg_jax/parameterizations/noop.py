# Copyright 2023 Karl Otness
# SPDX-License-Identifier: MIT


"""A no-op parameterization (does nothing).

This parameterization wraps the model, but otherwise leaves it
unmodified.

The only use case for this is if you need your model consistently
wrapped into a :class:`ParameterizedModel
<pyqg_jax.parameterizations.ParameterizedModel>` but don't want to
change their behavior.

If you don't *need* a parameterization or the wrapping, just skip
applying a parameterization and use the inner model directly.
"""

__all__ = [
    "apply_parameterization",
    "param_func",
    "init_param_aux_func",
]


from . import _parameterized_model


def apply_parameterization(model):
    """Apply the no-op parameterization to `model`.

    .. versionadded:: 0.3.0

    Parameters
    ----------
    model
        The inner model to wrap in the parameterization.

    Returns
    -------
    ParameterizedModel
        `model` wrapped in the parameterization.
    """
    return _parameterized_model.ParameterizedModel(
        model=model,
        param_func=param_func,
        init_param_aux_func=init_param_aux_func,
    )


def param_func(state, param_aux, model):
    return model.get_updates(state), None


def init_param_aux_func(state, model):
    return None
