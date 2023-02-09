# Copyright Karl Otness
# SPDX-License-Identifier: MIT


__all__ = ["AB3Stepper", "AB3State"]


import dataclasses
import typing
import types
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
        return cls(**dict(zip(aux_data, children)))


@_utils.register_pytree_node_class_private
class AB3State(StepperState[P]):
    def __init__(self, state: P, t: float, tc: int, ablevel: int, updates: typing.Tuple[P, P]):
        super().__init__(state=state, t=t, tc=tc)
        self._ablevel: int = jnp.uint8(0) if ablevel is None else ablevel
        self._updates: typing.Tuple[P, P] = updates

    def _tree_flatten(self):
        super_children, super_attrs = super()._tree_flatten()
        attr_names = ["ablevel", "updates"]
        attr_names.extend(super_attrs)
        children = [self._ablevel, self._updates]
        children.extend(super_children)
        return children, tuple(attr_names)


@_utils.register_pytree_node_class_private
class AB3Stepper:
    def __init__(self, dt: float):
        self.dt = dt

    def initialize_stepper_state(self, state: P) -> AB3State[P]:
        dummy_update: P = jax.tree_util.tree_map(jnp.zeros_like, state)
        return AB3State(
            state=state,
            t=jnp.float32(0),
            tc=jnp.uint32(0),
            ablevel=jnp.uint8(0),
            updates=(dummy_update, dummy_update),
        )

    def apply_updates(self, stepper_state: AB3State[P], updates: P) -> AB3State[P]:
        new_ablevel, dt1, dt2, dt3 = jax.lax.switch(
            stepper_state._ablevel,
            [
                lambda: (jnp.uint8(1), self.dt, 0.0, 0.0),
                lambda: (jnp.uint8(2), 1.5 * self.dt, -0.5 * self.dt, 0.0),
                lambda: (jnp.uint8(2), (23 / 12) * self.dt, (-16 / 12) * self.dt, (5 / 12) * self.dt),
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
        new_t = stepper_state.t + self.dt
        new_tc = stepper_state.tc + 1
        new_updates = (updates, updates_p)
        return AB3State(
            state=new_state,
            t=new_t,
            tc=new_tc,
            ablevel=new_ablevel,
            updates=new_updates
        )

    def _tree_flatten(self):
        return (self.dt, ), None

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(dt=children[0])
