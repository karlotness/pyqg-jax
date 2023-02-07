# Copyright Karl Otness
# SPDX-License-Identifier: MIT


import typing
import operator
import dataclasses
import jax
import jaxtyping


Children = typing.TypeVar("Children", bound=jaxtyping.PyTree)
AuxData = typing.TypeVar("AuxData")


class _PyTreePrivateProtocol(typing.Protocol[Children, AuxData]):
    def _tree_flatten(self) -> typing.Tuple[Children, AuxData]:
        ...

    @classmethod
    def _tree_unflatten(
            cls: typing.Type[typing.Self],
            aux_data: AuxData,
            children: Children
    ) -> typing.Self:
        ...


C = typing.TypeVar("C", bound=typing.Type[_PyTreePrivateProtocol])
Class = typing.TypeVar("Class", bound=typing.Type)


def register_pytree_node_class_private(cls: C) -> C:
    jax.tree_util.register_pytree_node(
        cls,
        operator.methodcaller("_tree_flatten"),
        cls._tree_unflatten,
    )
    return cls


def register_pytree_dataclass(cls: Class) -> Class:
    fields = tuple(f.name for f in dataclasses.fields(cls))

    def flatten(obj):
        return [getattr(obj, name) for name in fields], fields

    def unflatten(aux_data, flat_contents):
        return cls(**dict(zip(aux_data, flat_contents)))

    jax.tree_util.register_pytree_node(cls, flatten, unflatten)
    return cls
