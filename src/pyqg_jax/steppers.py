# Copyright Karl Otness
# SPDX-License-Identifier: MIT


__all__ = ["SteppedModel", "AB3Stepper", "AB3State"]


import typing
import jax
import jax.numpy as jnp
import jaxtyping
from . import _utils


P = typing.TypeVar("P", bound=jaxtyping.PyTree)


@_utils.register_pytree_node_class_private
class StepperState(typing.Generic[P]):
    def __init__(self, state: P, t: float, tc: int):
        self.state = state
        self.t = jnp.float32(t)
        self.tc = jnp.uint32(tc)

    def _tree_flatten(self):
        attr_names = ("state", "t", "tc")
        return [getattr(self, name) for name in attr_names], attr_names

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        for name, val in zip(aux_data, children):
            setattr(obj, name, val)
        return obj

    def update(self, **kwargs):
        # Check that only valid updates are applied
        if not kwargs.keys() <= {"state", "t", "tc"}:
            raise ValueError("invalid state updates, can only update state, t, and tc")
        # Perform the update
        children, attr_names = self._tree_flatten()
        attr_dict = {k: v for k, v in zip(attr_names, children)}
        attr_dict.update(kwargs)
        return self._tree_unflatten(attr_names, [attr_dict[k] for k in attr_names])


S = typing.TypeVar("S", bound=StepperState)


class Stepper:
    def __init__(self, dt: float):
        self.dt = float(dt)

    def initialize_stepper_state(self, state):
        raise NotImplementedError("implement in a subclass")

    def apply_updates(self, stepper_state, updates, postprocess_state=None):
        raise NotImplementedError("implement in a subclass")


@_utils.register_pytree_node_class_private
class SteppedModel:
    def __init__(self, model, stepper):
        self.model = model
        self.stepper = stepper

    def create_initial_state(self, key):
        model_state = self.model.create_initial_state(key=key)
        return self.initialize_stepper_state(model_state)

    def initialize_stepper_state(self, state, /):
        return self.stepper.initialize_stepper_state(state)

    def step_model(self, stepper_state, /):
        return self.stepper.apply_updates(
            stepper_state,
            self.model.get_updates(stepper_state.state),
            postprocess_state=self.model.postprocess_state,
        )

    def get_full_state(self, stepper_state, /):
        return self.model.get_full_state(stepper_state.state)

    def _tree_flatten(self):
        return (self.model, self.stepper), None

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        model, stepper = children
        obj = cls.__new__(cls)
        obj.model = model
        obj.stepper = stepper
        return obj


@_utils.register_pytree_node_class_private
class AB3State(StepperState[P]):
    def __init__(
        self, state: P, t: float, tc: int, ablevel: int, updates: typing.Tuple[P, P]
    ):
        super().__init__(state=state, t=t, tc=tc)
        self._ablevel: int = jnp.uint8(ablevel)
        self._updates: typing.Tuple[P, P] = updates

    def _tree_flatten(self):
        super_children, super_attrs = super()._tree_flatten()
        attr_names = (*super_attrs, "_ablevel", "_updates")
        children = [*super_children, self._ablevel, self._updates]
        return children, attr_names


@_utils.register_pytree_node_class_private
class AB3Stepper(Stepper):
    def __init__(self, dt: float):
        super().__init__(dt=dt)

    def initialize_stepper_state(self, state: P) -> AB3State[P]:
        dummy_update: P = jax.tree_util.tree_map(jnp.zeros_like, state)
        return AB3State(
            state=state,
            t=jnp.float32(0),
            tc=jnp.uint32(0),
            ablevel=jnp.uint8(0),
            updates=(dummy_update, dummy_update),
        )

    def apply_updates(
        self,
        stepper_state: AB3State[P],
        updates: P,
        postprocess_state: typing.Optional[typing.Callable[[P], P]] = None,
    ) -> AB3State[P]:
        new_ablevel, dt1, dt2, dt3 = jax.lax.switch(
            stepper_state._ablevel,
            [
                lambda: (jnp.uint8(1), self.dt, 0.0, 0.0),
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
        new_state = jax.tree_util.tree_map(
            (lambda v, u, u_p, u_pp: v + (dt1 * u) + (dt2 * u_p) + (dt3 * u_pp)),
            stepper_state.state,
            updates,
            updates_p,
            updates_pp,
        )
        if postprocess_state is not None:
            new_state = postprocess_state(new_state)
        new_t = stepper_state.t + jnp.float32(self.dt)
        new_tc = stepper_state.tc + 1
        new_updates = (updates, updates_p)
        return AB3State(
            state=new_state,
            t=new_t,
            tc=new_tc,
            ablevel=new_ablevel,
            updates=new_updates,
        )

    def _tree_flatten(self):
        return (self.dt,), None

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj.dt = children[0]
        return obj
