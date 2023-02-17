# Copyright Karl Otness
# SPDX-License-Identifier: MIT


__all__ = ["ParametrizedModelState", "ParametrizedModel"]


import dataclasses
import itertools
import typing
from .. import state as _state, _utils, steppers as _steppers


@_utils.register_pytree_dataclass
@dataclasses.dataclass(frozen=True)
class ParametrizedModelState:
    model_state: typing.Union[
        _state.PseudoSpectralState, _state.FullPseudoSpectralState
    ]
    param_aux: _steppers.NoStepValue

    def __repr__(self):
        model_state_summary = _utils.indent_repr(
            _utils.summarize_object(self.model_state), 2
        )
        param_aux_summary = _utils.indent_repr(
            _utils.summarize_object(self.param_aux), 2
        )
        return f"""\
ParametrizedModelState(
  model_state={model_state_summary},
  param_aux={param_aux_summary},
)"""


def _init_none(init_state, model):
    return None


@_utils.register_pytree_node_class_private
class ParametrizedModel:
    def __init__(self, model, param_func, init_param_aux_func=None):
        # param_func(full_state, param_aux, model) -> full_state, param_aux
        # init_param_aux_func(model_state, model) -> param_aux
        # param_aux (often None) is used to carry parameterization state
        # between time steps, for example: a JAX PRNGKey, if needed
        self.model = model
        self.param_func = param_func
        if init_param_aux_func is None:
            self.init_param_aux_func = _init_none
        else:
            self.init_param_aux_func = init_param_aux_func

    def get_full_state(self, state):
        return self.model.get_full_state(state.model_state)

    def get_updates(self, state):
        param_updates, new_param_aux = self.param_func(
            state.model_state, state.param_aux.value, self.model
        )
        return ParametrizedModelState(
            model_state=param_updates,
            param_aux=_steppers.NoStepValue(new_param_aux),
        )

    def postprocess_state(self, state):
        return ParametrizedModelState(
            model_state=self.model.postprocess_state(state.model_state),
            param_aux=state.param_aux,
        )

    def create_initial_state(self, key):
        return self.initialize_param_state(self.model.create_initial_state(key=key))

    def initialize_param_state(self, state):
        init_param_state = self.init_param_aux_func(state, self.model)
        return ParametrizedModelState(
            model_state=state,
            param_aux=_steppers.NoStepValue(init_param_state),
        )

    def _tree_flatten(self):
        static_attributes = ("param_func", "init_param_aux_func")
        child_attributes = ("model",)
        child_vals = [getattr(self, attr) for attr in child_attributes]
        static_vals = [getattr(self, attr) for attr in static_attributes]
        return child_vals, (child_attributes, static_vals, static_attributes)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        child_attributes, static_vals, static_attributes = aux_data
        obj = cls.__new__(cls)
        for name, val in itertools.chain(
            zip(child_attributes, children),
            zip(static_attributes, static_vals),
        ):
            setattr(obj, name, val)
        return obj

    def __repr__(self):
        model_summary = _utils.indent_repr(_utils.summarize_object(self.model), 2)
        param_func_summary = _utils.indent_repr(
            _utils.summarize_object(self.param_func), 2
        )
        init_param_aux_func_summary = _utils.indent_repr(
            _utils.summarize_object(self.init_param_aux_func), 2
        )
        return f"""\
ParametrizedModel(
  model={model_summary},
  param_func={param_func_summary},
  init_param_aux_func={init_param_aux_func_summary},
)"""
