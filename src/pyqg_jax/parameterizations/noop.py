# Copyright Karl Otness
# SPDX-License-Identifier: MIT


# A no-op parameterization (does nothing)
# Parameterizations are optional, so the only use-case for this is if
# you need your models consistently wrapped with a parameterizations,
# but don't want updates. If you don't need that, just skip applying
# the parameterization.


__all__ = [
    "apply_parameterization",
    "param_func",
    "init_param_aux_func",
]


from . import _parameterized_model


def apply_parameterization(model):
    return _parameterized_model.ParameterizedModel(
        model=model,
        param_func=param_func,
        init_param_aux_func=init_param_aux_func,
    )


def param_func(state, param_aux, model):
    return model.get_updates(state), None


def init_param_aux_func(state, model):
    return None
