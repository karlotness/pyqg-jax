# Copyright Karl Otness
# SPDX-License-Identifier: MIT


"""Time-stepping schemes and utilities for using them to update model states.

To step a model through time, use :class:`SteppedModel`.

:class:`AB3Stepper` implements the same time-stepping scheme used in
PyQG.
"""


__all__ = [
    "SteppedModel",
    "AB3Stepper",
    "AB3State",
    "NoStepValue",
    "EulerStepper",
    "StepperState",
]


import typing
import functools
import dataclasses
import abc
import jax
import jax.numpy as jnp
from . import _utils


P = typing.TypeVar("P")


@_utils.register_pytree_dataclass
@dataclasses.dataclass(frozen=True)
class StepperState(typing.Generic[P]):
    """Model state wrapped for time-stepping

    Warning
    -------
    You should not construct this class yourself. Instead, you should
    obtain instances from your chosen time stepper, or
    :class:`SteppedModel`.

    Attributes
    ----------

    state : PseudoSpectralState or ParameterizedModelState
        The inner state from the model being stepped forward. The
        actual type of `state` depends on the model being stepped.

    t : jax.numpy.float32
        The current model time

    tc : jax.numpy.uint32
        The current model timestep
    """

    state: P
    t: jnp.float32
    tc: jnp.uint32

    def update(self, **kwargs):
        """Replace values stored in this state.

        This function produces a *new* state object, containing the
        replacement values.

        The keyword arguments may be any of `state`, `t`, or `tc`.

        The object this method is called on is not modified.

        Parameters
        ----------
        state : PseudoSpectralState or ParameterizedModelState, optional
            Replacement value for :attr:`state`.

        t : jax.numpy.float32, optional
            Replacement value for :attr:`t`.
            The current model time

        tc : jax.numpy.uint32, optional
            Replacement value for :attr:`tc`.

        Returns
        -------
        StepperState
            A copy of this object with the specified values replaced.
        """
        # Check that only valid updates are applied
        if not kwargs.keys() <= {"state", "t", "tc"}:
            raise ValueError("invalid state updates, can only update state, t, and tc")
        # Perform the update
        return dataclasses.replace(self, **kwargs)

    def __repr__(self):
        class_name = type(self).__name__
        state_summary = _utils.indent_repr(_utils.summarize_object(self.state), 2)
        t_summary = _utils.summarize_object(self.t)
        tc_summary = _utils.summarize_object(self.tc)
        return f"""\
{class_name}(
  t={t_summary},
  tc={tc_summary},
  state={state_summary},
)"""


@dataclasses.dataclass
class Stepper(abc.ABC):
    dt: float

    def initialize_stepper_state(self, state):
        """Wrap an existing `state` from a model in a
        :class:`StepperState` to prepare it for time stepping.

        This initializes a new :class:`StepperState` from a time of
        :pycode:`0`.

        Parameters
        ----------
        state
            The model state to wrap.

        Returns
        -------
        StepperState
            The wrapped state. Note this will be a subclass of
            :class:`StepperState` appropriate for this time stepper.
        """
        return StepperState(
            state=state,
            t=jnp.float32(0),
            tc=jnp.uint32(0),
        )

    @abc.abstractmethod
    def apply_updates(self, stepper_state, updates):
        pass

    def __repr__(self):
        class_name = type(self).__name__
        dt_summary = _utils.summarize_object(self.dt)
        return f"{class_name}(dt={dt_summary})"


@_utils.register_pytree_class_attrs(
    children=["model", "stepper"],
    static_attrs=[],
)
class SteppedModel:
    """Combine an inner model with a time stepper.

    This class simplifies the process of stepping a base model through
    time by handling the interactions between the model and time
    stepper.

    Parameters
    ----------
    model
        The inner model to step through time.

    stepper
        The time stepper applying the updates to the model each step.

    Attributes
    ----------
    model
        The inner model being stepped.

    stepper
        The time stepper used to apply the updates.
    """

    def __init__(self, model, stepper):
        self.model = model
        self.stepper = stepper

    def create_initial_state(self, key, *args, **kwargs):
        """Create a new wrapped initial state with random
        initialization.

        This function defers to :attr:`model` to initialize the inner
        state, passing it any additional arguments. Then wraps it by
        calling `initialize_stepper_state` on :attr:`stepper`.

        Parameters
        ----------
        key : jax.random.key
            The PRNG state used as the random key for initialization.

        *args
            Arbitrary additional arguments for the model's
            initialization function.

        **kwargs
            Arbitrary additional arguments for the model's
            initialization function.

        Returns
        -------
        StepperState
            The new wrapped state with random initialization.
        """
        model_state = self.model.create_initial_state(key, *args, **kwargs)
        return self.initialize_stepper_state(model_state)

    def initialize_stepper_state(self, state, /):
        """Wrap an existing state from :attr:`model` in a
        :class:`StepperState`, preparing it for time stepping.

        This function takes an existing inner model state and wraps it
        so that it can be stepped through time by :attr:`stepper`.

        This function defers to :attr:`stepper` to perform the
        wrapping.

        Parameters
        ----------
        state
            The inner model state to wrap. The type depends on
            :attr:`model` but is likely to be
            :class:`PseudoSpectralState
            <pyqg_jax.state.PseudoSpectralState>` or
            :class:`ParameterizedModelState
            <pyqg_jax.state.ParameterizedModelState>`.

        Returns
        -------
        StepperState
            A wrapped copy of `state`.
        """
        return self.stepper.initialize_stepper_state(state)

    def step_model(self, stepper_state, /):
        """Update a state by computing the next time step.

        This method handles the interaction between :attr:`model` and
        :attr:`stepper`, including post-processing/filtering.

        To take multiple steps over time combine this method with
        :func:`jax.lax.scan`.

        Parameters
        ----------
        stepper_state : StepperState
            The wrapped state to step forward in time.

        Returns
        -------
        StepperState
            The updated wrapped state, a new object.
        """
        new_stepper_state = self.stepper.apply_updates(
            stepper_state,
            self.model.get_updates(stepper_state.state),
        )
        postprocessed_state = self.model.postprocess_state(new_stepper_state.state)
        return new_stepper_state.update(state=postprocessed_state)

    def get_full_state(self, stepper_state, /):
        """Expand a wrapped partial state into an *unwrapped* full
        state.

        This function defers to :attr:`model` to compute the full
        state.

        Parameters
        ----------
        stepper_state : StepperState
            The wrapped state to be expanded.

        Returns
        -------
        FullPseudoSpectralState
            The expanded state. The real type depends on
            :attr:`model`, but is likely to be
            :class:`FullPseudoSpectralState
            <pyqg_jax.state.FullPseudoSpectralState>`.
        """
        return self.model.get_full_state(stepper_state.state)

    def __repr__(self):
        model_summary = _utils.indent_repr(_utils.summarize_object(self.model), 2)
        stepper_summary = _utils.indent_repr(_utils.summarize_object(self.stepper), 2)
        return f"""\
SteppedModel(
  model={model_summary},
  stepper={stepper_summary},
)"""


def _wrap_nostep_update(func):
    @functools.wraps(func)
    def wrapper(leaf, update, *args, **kwargs):
        if isinstance(update, NoStepValue):
            return update
        return func(leaf, update, *args, **kwargs)

    return wrapper


def _nostep_tree_map(func, tree, *rest):
    return jax.tree_util.tree_map(
        _wrap_nostep_update(func),
        tree,
        *rest,
        is_leaf=(lambda l: isinstance(l, NoStepValue)),
    )


def _dummy_step_init(state):
    def leaf_map(leaf):
        if isinstance(leaf, NoStepValue):
            return NoStepValue(None)
        return jnp.zeros_like(leaf)

    return jax.tree_util.tree_map(
        leaf_map, state, is_leaf=(lambda l: isinstance(l, NoStepValue))
    )


def _map_state_remove_nostep(state):
    def leaf_map(leaf):
        if isinstance(leaf, NoStepValue):
            return NoStepValue(None)
        return leaf

    return jax.tree_util.tree_map(
        leaf_map, state, is_leaf=(lambda l: isinstance(l, NoStepValue))
    )


@_utils.register_pytree_dataclass
@dataclasses.dataclass(repr=False)
class EulerStepper(Stepper):
    """Forward Euler (first-order) stepper.

    Parameters
    ----------
    dt : float
        Numerical time step

    Attributes
    ----------
    dt : float
        Numerical time step
    """

    def apply_updates(self, stepper_state, updates):
        """Apply `updates` to the existing `stepper_state` producing
        the next step in time.

        `updates` should be provided by the model that produced
        :attr:`StepperState.state`.

        Parameters
        ----------
        stepper_state : StepperState
            The time-stepper wrapped state to be updated.

        updates : PseudoSpectralState or ParameterizedModelState
            The *unwrapped* updates to apply. The actual type of
            `updates` depends on the model being stepped.

        Returns
        -------
        StepperState
            The updated, wrapped state at the next time step.

        Note
        ----
        This method does not apply post-processing to the updated
        state.
        """
        new_state = _nostep_tree_map(
            (lambda v, u: v + (self.dt * u)),
            stepper_state.state,
            updates,
        )
        new_t = stepper_state.t + jnp.float32(self.dt)
        new_tc = stepper_state.tc + 1
        return StepperState(
            state=new_state,
            t=new_t,
            tc=new_tc,
        )


@_utils.register_pytree_dataclass
@dataclasses.dataclass(frozen=True, repr=False)
class AB3State(StepperState[P]):
    _ablevel: jnp.uint8
    _updates: tuple[P, P]


@_utils.register_pytree_dataclass
@dataclasses.dataclass(repr=False)
class AB3Stepper(Stepper):
    """Third order Adams-Bashforth stepper.

    This is the same time stepping scheme as used in PyQG.

    This time-stepper bootstraps using lower order Adams-Bashforth
    schemes for the first two steps.

    Parameters
    ----------
    dt : float
        Numerical time step

    Attributes
    ----------
    dt : float
        Numerical time step
    """

    def initialize_stepper_state(self, state: P) -> AB3State[P]:
        """Wrap an existing `state` from a model in a
        :class:`StepperState` to prepare it for time stepping.

        This initializes a new :class:`StepperState` from a time of
        :pycode:`0`.

        Parameters
        ----------
        state
            The model state to wrap.

        Returns
        -------
        StepperState
            The wrapped state. Note this will be a subclass of
            :class:`StepperState` appropriate for this time stepper.
        """
        base_state = super().initialize_stepper_state(state)
        dummy_update: P = _dummy_step_init(state)
        return AB3State(
            state=base_state.state,
            t=base_state.t,
            tc=base_state.tc,
            _ablevel=jnp.uint8(0),
            _updates=(dummy_update, dummy_update),
        )

    def apply_updates(
        self,
        stepper_state: AB3State[P],
        updates: P,
    ) -> AB3State[P]:
        """Apply `updates` to the existing `stepper_state` producing
        the next step in time.

        `updates` should be provided by the model that produced
        :attr:`StepperState.state`.

        Parameters
        ----------
        stepper_state : StepperState
            The time-stepper wrapped state to be updated.

        updates : PseudoSpectralState or ParameterizedModelState
            The *unwrapped* updates to apply. The actual type of
            `updates` depends on the model being stepped.

        Returns
        -------
        StepperState
            The updated, wrapped state at the next time step.

        Note
        ----
        This method does not apply post-processing to the updated
        state.
        """
        new_ablevel, dt1, dt2, dt3 = jax.lax.switch(
            stepper_state._ablevel,
            [
                lambda: (jnp.uint8(1), 1.0 * self.dt, 0.0, 0.0),
                lambda: (jnp.uint8(2), 1.5 * self.dt, -0.5 * self.dt, 0.0),
                lambda: (
                    jnp.uint8(2),
                    (23 / 12) * self.dt,
                    (-16 / 12) * self.dt,
                    (5 / 12) * self.dt,
                ),
            ],
        )
        updates_p, updates_pp = stepper_state._updates
        new_state = _nostep_tree_map(
            (lambda v, u, u_p, u_pp: v + (dt1 * u) + (dt2 * u_p) + (dt3 * u_pp)),
            stepper_state.state,
            updates,
            updates_p,
            updates_pp,
        )
        new_t = stepper_state.t + jnp.float32(self.dt)
        new_tc = stepper_state.tc + 1
        new_updates = (_map_state_remove_nostep(updates), updates_p)
        return AB3State(
            state=new_state,
            t=new_t,
            tc=new_tc,
            _ablevel=new_ablevel,
            _updates=new_updates,
        )


@_utils.register_pytree_dataclass
@dataclasses.dataclass
class NoStepValue(typing.Generic[P]):
    """Shields contents from the provided time-steppers.

    When a time-stepper encounters a value wrapped in this class, it
    will skip its normal stepping computations and directly use the
    value from the updates. This allows a user to manually update an
    auxiliary value outside the normal time-stepping.

    For example, :func:`jax.random.key` values should not be
    time-stepped normally. Wrapping them in this class and manually
    :func:`updating them <jax.random.split>` can accomplish this.

    This class is used as part of :class:`ParameterizedModelState
    <pyqg_jax.parameterizations.ParameterizedModelState>`.

    Parameters
    ----------
    value : object
        The inner value to wrap. This can be an arbitrary JAX PyTree.

    Attributes
    ----------
    value
        The internal, wrapped value
    """

    value: P

    def __repr__(self):
        value_summary = _utils.summarize_object(self.value)
        if "\n" not in value_summary:
            # Single line summary
            return f"NoStepValue(value={value_summary})"
        else:
            value_summary = _utils.indent_repr(value_summary, 2)
        return f"""\
NoStepValue(
  value={value_summary}
)"""
