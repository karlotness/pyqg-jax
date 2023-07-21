# Copyright Karl Otness
# SPDX-License-Identifier: MIT


__all__ = ["ParameterizedModelState", "ParameterizedModel"]


import dataclasses
from .. import state as _state, _utils, steppers as _steppers


@_utils.register_pytree_dataclass
@dataclasses.dataclass(frozen=True)
class ParameterizedModelState:
    """Wrapped model state for parameterized models.

    Warning
    -------
    You should not construct this class yourself. Instead, you should
    obtain instances from :class:`ParameterizedModel`.

    Attributes
    ----------
    model_state : PseudoSpectralState
        The inner model state. The actual types depends on the inner
        model, but this is likely to be
        :class:`FullPseudoSpectralState
        <pyqg_jax.state.PseudoSpectralState>`.

    param_aux : NoStepValue
        The auxiliary state for the parameterization. This is an
        arbitrary object time-stepped by the parameterization itself.
        It will be wrapped in a :class:`NoStepValue
        <pyqg_jax.steppers.NoStepValue>` to shield it from the time
        steppers.
    """

    model_state: _state.PseudoSpectralState
    param_aux: _steppers.NoStepValue

    def __repr__(self):
        model_state_summary = _utils.indent_repr(
            _utils.summarize_object(self.model_state), 2
        )
        param_aux_summary = _utils.indent_repr(
            _utils.summarize_object(self.param_aux), 2
        )
        return f"""\
ParameterizedModelState(
  model_state={model_state_summary},
  param_aux={param_aux_summary},
)"""


def _init_none(init_state, model):
    return None


@_utils.register_pytree_class_attrs(
    children=["model"],
    static_attrs=["param_func", "init_param_aux_func"],
)
class ParameterizedModel:
    """A model wrapped in a user-specified parameterization.

    Parameters
    ----------
    model
        The inner core model to wrap in the parameterization.

    param_func : function
        The function implementing the parameterization. Will be called
        by :meth:`get_updates` to compute time-stepping updates.

    init_param_aux_func : function, optional
        The function used to initialize the parameterization's
        auxiliary state. Defaults to a function initializing the state
        to :pycode:`None`.

    Attributes
    ----------
    model
        The inner model wrapped in the parameterization.

    param_func : function
        The user-specified parameterization function.
        Takes arguments :pycode:`(model_state, param_aux, model)`.

    init_param_aux_func : function
        Function used to initialize the auxiliary state.
        Takes arguments :pycode:`(model_state, model)`.
    """

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
        """Expand a wrapped partial state into an *unwrapped* full
        state.

        This function defers to :attr:`model` to compute the full
        state.

        Parameters
        ----------
        state : ParameterizedModelState
            The wrapped, parameterized state to be expanded.

        Returns
        -------
        FullPseudoSpectralState
            The expanded state. The real type depends on
            :attr:`model`, but is likely to be
            :class:`FullPseudoSpectralState
            <pyqg_jax.state.FullPseudoSpectralState>`.
        """
        return self.model.get_full_state(state.model_state)

    def get_updates(self, state):
        """Get updates for time-stepping `state`.

        `state` is a wrapped, partial :attr:`model` state. This
        function returns updates for time-stepping.

        This function makes use of :attr:`param_func`, applying the
        parameterization to the updates.

        Parameters
        ----------
        state : ParameterizedModelState
            The state which will be time stepped using the computed
            updates.

        Returns
        -------
        ParameterizedModelState
            A new state object where each field corresponds to a
            time-stepping *update* to be applied.

        Note
        ----
        The object returned by this function has the same type of
        `state`, but contains *updates*. This is so the time-stepping
        can be done by mapping over the states and updates as JAX
        pytrees with the same structure.
        """
        param_updates, new_param_aux = self.param_func(
            state.model_state, state.param_aux.value, self.model
        )
        return ParameterizedModelState(
            model_state=param_updates,
            param_aux=_steppers.NoStepValue(new_param_aux),
        )

    def postprocess_state(self, state):
        """Apply fixed filtering to `state`.

        This function should be called once on each new state after
        each time step.

        :class:`SteppedModel <pyqg_jax.steppers.SteppedModel>` handles
        this internally.

        This function defers to :attr:`model` for the post-processing.

        Parameters
        ----------
        state : ParameterizedModelState
            The wrapped state to be filtered.

        Returns
        -------
        ParameterizedModelState
            The wrapped filtered state.
        """
        return ParameterizedModelState(
            model_state=self.model.postprocess_state(state.model_state),
            param_aux=state.param_aux,
        )

    def create_initial_state(self, key, *args, **kwargs):
        """Create a new wrapped initial state with random
        initialization.

        This function defers to :attr:`model` to initialize the inner
        state and makes use of :attr:`init_param_aux_func` to
        initialize the parameterization's auxiliary state.

        Parameters
        ----------
        key : jax.random.PRNGKey
            The PRNG used as the random key for initialization.

        *args
            Arbitrary additional arguments for :attr:`init_param_aux_func`

        **kwargs
            Arbitrary additional arguments for :attr:`init_param_aux_func`

        Returns
        -------
        ParameterizedModelState
            The new wrapped state with random initialization.
        """
        return self.initialize_param_state(
            self.model.create_initial_state(key=key), *args, **kwargs
        )

    def initialize_param_state(self, state, *args, **kwargs):
        """Wrap an existing state from :attr:`model` in a
        :class:`ParameterizedModelState`.

        This function takes an existing inner model state and wraps it
        so that it can be used with the parameterized model.

        This function uses of :attr:`init_param_aux_func` to
        initialize the parameterization's auxiliary state.

        Parameters
        ----------
        state
            The inner model state to wrap. The type depends on
            :attr:`model` but is likely to be
            :class:`PseudoSpectralState
            <pyqg_jax.state.PseudoSpectralState>`.

        *args
            Arbitrary additional arguments for :attr:`init_param_aux_func`

        **kwargs
            Arbitrary additional arguments for :attr:`init_param_aux_func`

        Returns
        -------
        ParameterizedModelState
            A wrapped copy of `state`.
        """
        init_param_state = self.init_param_aux_func(state, self.model, *args, **kwargs)
        return ParameterizedModelState(
            model_state=state,
            param_aux=_steppers.NoStepValue(init_param_state),
        )

    def __repr__(self):
        model_summary = _utils.indent_repr(_utils.summarize_object(self.model), 2)
        param_func_summary = _utils.indent_repr(
            _utils.summarize_object(self.param_func), 2
        )
        init_param_aux_func_summary = _utils.indent_repr(
            _utils.summarize_object(self.init_param_aux_func), 2
        )
        return f"""\
ParameterizedModel(
  model={model_summary},
  param_func={param_func_summary},
  init_param_aux_func={init_param_aux_func_summary},
)"""
