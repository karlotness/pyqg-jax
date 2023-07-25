# Copyright Karl Otness
# SPDX-License-Identifier: MIT


import typing
import functools
import textwrap
import types
import itertools
import dataclasses
import inspect
import weakref
import jax


def summarize_object(obj: object) -> str:
    if isinstance(obj, jax.Array) or (hasattr(obj, "shape") and hasattr(obj, "dtype")):
        return summarize_array(obj)
    elif isinstance(obj, functools.partial):
        return summarize_partial(obj)
    elif isinstance(obj, types.FunctionType):
        return summarize_function(obj)
    elif isinstance(obj, (tuple, list, set)):
        return summarize_sequence(obj)
    elif isinstance(obj, dict):
        return summarize_dict(obj)
    else:
        return repr(obj)


def summarize_function(func: types.FunctionType) -> str:
    try:
        func_name = str(func.__qualname__)
        func_module = str(func.__module__)
    except AttributeError:
        return repr(func)
    if func_module == "builtins":
        return f"<function {func_name}>"
    return f"<function {func_module}.{func_name}>"


def summarize_partial(partial: functools.partial) -> str:
    func = summarize_object(partial.func)
    args = (summarize_object(arg) for arg in partial.args)
    kwargs = (
        f"{name}={summarize_object(value)}" for name, value in partial.keywords.items()
    )
    contents = ", ".join(itertools.chain([func], args, kwargs))
    cls = type(partial)
    qualname = f"{cls.__module__!s}.{cls.__qualname__!s}"
    return f"{qualname}({contents})"


def summarize_array(arr: jax.Array) -> str:
    dtype = (
        str(arr.dtype.name)
        .replace("float", "f")
        .replace("uint", "u")
        .replace("int", "i")
        .replace("complex", "c")
    )
    shape = ",".join(str(d) for d in arr.shape)
    return f"{dtype}[{shape}]"


class ReprDummy:
    def __init__(self, rep_str: str):
        self.rep_str = rep_str

    def __repr__(self) -> str:
        return self.rep_str


def summarize_sequence(
    seq: typing.Union[tuple[object], list[object], set[object]]
) -> str:
    return repr(type(seq)(ReprDummy(summarize_object(o)) for o in seq))


def summarize_dict(dt: dict[object, object]) -> str:
    return repr(
        {
            ReprDummy(summarize_object(k)): ReprDummy(summarize_object(v))
            for k, v in dt.items()
        }
    )


def indent_repr(text: str, spaces: int) -> str:
    indent_str = " " * spaces
    indented = textwrap.indent(text, indent_str)
    if indented.startswith(indent_str):
        return indented[spaces:]
    return indented


pytree_class_attrs_registry = weakref.WeakKeyDictionary()


def register_pytree_class_attrs(children, static_attrs):
    children = tuple(children)
    static_attrs = tuple(static_attrs)

    def do_registration(cls):
        pytree_class_attrs_registry[cls] = (children, static_attrs)
        # Combine recursively
        mro = inspect.getmro(cls)
        cls_children = set()
        cls_static = set()
        for c in mro:
            c_children, c_static = pytree_class_attrs_registry.get(c, ((), ()))
            cls_children.update(c_children)
            cls_static.update(c_static)
        if not cls_children.isdisjoint(cls_static):
            raise ValueError("Recursive static and dynamic attributes overlap")
        cls_children = tuple(cls_children)
        cls_static = tuple(cls_static)

        def flatten_with_keys(obj):
            key_children = [
                (jax.tree_util.GetAttrKey(name), getattr(obj, name))
                for name in cls_children
            ]
            if cls_static:
                aux = tuple(getattr(obj, name) for name in cls_static)
            else:
                aux = None
            return key_children, aux

        def flatten(obj):
            flatkeys, aux = flatten_with_keys(obj)
            return [c for _, c in flatkeys], aux

        def unflatten(aux_data, children):
            obj = cls.__new__(cls)
            if aux_data is None:
                aux_data = ()
            for name, val in itertools.chain(
                zip(cls_children, children),
                zip(cls_static, aux_data),
            ):
                setattr(obj, name, val)
            return obj

        jax.tree_util.register_pytree_with_keys(
            cls, flatten_with_keys, unflatten, flatten
        )
        return cls

    return do_registration


def register_pytree_dataclass(cls):
    fields = tuple(f.name for f in dataclasses.fields(cls))

    def flatten_with_keys(obj):
        return [
            (jax.tree_util.GetAttrKey(name), getattr(obj, name)) for name in fields
        ], None

    def flatten(obj):
        flatkeys, aux = flatten_with_keys(obj)
        return [c for _, c in flatkeys], aux

    def unflatten(aux_data, flat_contents):
        return cls(**dict(zip(fields, flat_contents)))

    jax.tree_util.register_pytree_with_keys(cls, flatten_with_keys, unflatten, flatten)
    return cls
