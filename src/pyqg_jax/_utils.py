# Copyright Karl Otness
# SPDX-License-Identifier: MIT


import typing
import functools
import textwrap
import types
import operator
import itertools
import dataclasses
import jax
import jaxtyping


def summarize_object(obj):
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        return summarize_array(obj)
    elif isinstance(obj, functools.partial):
        return summarize_partial(obj)
    elif isinstance(obj, types.FunctionType):
        return summarize_function(obj)
    else:
        return repr(obj)


def summarize_function(func):
    try:
        func_name = str(func.__qualname__)
        func_module = str(func.__module__)
    except AttributeError:
        return repr(func)
    if func_module == "builtins":
        return f"<function {func_name}>"
    return f"<function {func_module}.{func_name}>"


def summarize_partial(partial):
    func = summarize_object(partial.func)
    args = (summarize_object(arg) for arg in partial.args)
    kwargs = (
        f"{name}={summarize_object(value)}" for name, value in partial.keywords.items()
    )
    contents = ", ".join(itertools.chain([func], args, kwargs))
    return f"functools.partial({contents})"


def summarize_array(arr):
    dtype = (
        str(arr.dtype.name)
        .replace("float", "f")
        .replace("uint", "u")
        .replace("int", "i")
        .replace("complex", "c")
    )
    shape = ",".join(str(d) for d in arr.shape)
    return f"{dtype}[{shape}]"


def indent_repr(text, spaces):
    indent_str = " " * spaces
    indented = textwrap.indent(text, indent_str)
    if indented.startswith(indent_str):
        return indented[spaces:]
    return indented


Children = typing.TypeVar("Children", bound=jaxtyping.PyTree)
AuxData = typing.TypeVar("AuxData")


class _PyTreePrivateProtocol(typing.Protocol[Children, AuxData]):
    def _tree_flatten(self) -> typing.Tuple[Children, AuxData]:
        ...

    @classmethod
    def _tree_unflatten(cls, aux_data: AuxData, children: Children):
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
