# Copyright 2023 Karl Otness
# SPDX-License-Identifier: MIT


import functools
import textwrap
import types
import itertools
import dataclasses
import inspect
import weakref
import importlib
import typing
import jax
import jax.numpy as jnp


def find_array_like_types() -> tuple[type, ...]:
    array_like_types = [jax.Array, jax.ShapeDtypeStruct]
    try:
        np_mod = importlib.import_module("numpy")
        array_like_types.append(np_mod.ndarray)
    except Exception:
        pass
    return tuple(array_like_types)


array_like_types: typing.Final[tuple[type, ...]] = find_array_like_types()


def summarize_object(obj: object) -> str:
    if isinstance(obj, array_like_types):
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
    if cls == jax.tree_util.Partial:
        qualname = "jax.tree_util.Partial"
    else:
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


@dataclasses.dataclass(frozen=True)
class ReprDummy:
    obj: object

    def __repr__(self):
        return summarize_object(self.obj)


def summarize_sequence(seq: tuple[object] | list[object] | set[object]) -> str:
    return repr(type(seq)(ReprDummy(o) for o in seq))


def summarize_dict(dt: dict[object, object]) -> str:
    return repr({ReprDummy(k): ReprDummy(v) for k, v in dt.items()})


def indent_repr(text: str, spaces: int) -> str:
    indent_str = " " * spaces
    indented = textwrap.indent(text, indent_str)
    if indented.startswith(indent_str):
        return indented[spaces:]
    return indented


def auto_repr(obj: object) -> str:
    parts = [f"{type(obj).__name__}("]
    for attr in inspect.signature(type(obj)).parameters:
        parts.append(
            f"  {attr}={indent_repr(summarize_object(getattr(obj, attr)), 2)},"
        )
    parts.append(")")
    return "\n".join(parts)


@dataclasses.dataclass(frozen=True)
class AttrGetter:
    # Like operator.attrgetter but supports weak references
    attr: str

    def __call__(self, obj):
        return getattr(obj, self.attr)


pytree_class_attrs_registry = weakref.WeakKeyDictionary()


def register_pytree_class_attrs(children, static_attrs):
    children = tuple(children)
    static_attrs = tuple(static_attrs)

    def do_registration(cls):
        pytree_class_attrs_registry[cls] = (children, static_attrs)
        # Combine recursively
        cls_children = set()
        cls_static = set()
        for c in inspect.getmro(cls):
            c_children, c_static = pytree_class_attrs_registry.get(c, ((), ()))
            cls_children.update(c_children)
            cls_static.update(c_static)
        if not cls_children.isdisjoint(cls_static):
            raise ValueError("recursive static and dynamic attributes overlap")
        cls_child_fields = tuple(cls_children)
        cls_static_fields = tuple(cls_static)

        def flatten_with_keys(obj):
            key_children = [
                (jax.tree_util.GetAttrKey(name), getattr(obj, name))
                for name in cls_child_fields
            ]
            if cls_static_fields:
                aux = tuple(getattr(obj, name) for name in cls_static_fields)
            else:
                aux = None
            return key_children, aux

        def flatten(obj):
            flatkeys, aux = flatten_with_keys(obj)
            return [c for _, c in flatkeys], aux

        def unflatten(aux_data, children):
            obj = cls.__new__(cls)
            for name, val in itertools.chain(
                zip(cls_child_fields, children, strict=True),
                zip(cls_static_fields, aux_data or (), strict=True),
            ):
                setattr(obj, name, val)
            return obj

        jax.tree_util.register_pytree_with_keys(
            cls, flatten_with_keys, unflatten, flatten
        )
        return cls

    return do_registration


def register_pytree_dataclass(cls):
    cls_fields = []
    cls_static_fields = []
    for field in dataclasses.fields(cls):
        if not field.init:
            continue
        if field.metadata.get("pyqg_jax", {}).get("static", False):
            cls_static_fields.append(field.name)
        else:
            cls_fields.append(field.name)
    fields = tuple(cls_fields)
    static_fields = tuple(cls_static_fields)

    def flatten_with_keys(obj):
        if static_fields:
            aux = tuple(getattr(obj, name) for name in static_fields)
        else:
            aux = None
        return [
            (jax.tree_util.GetAttrKey(name), getattr(obj, name)) for name in fields
        ], aux

    def flatten(obj):
        flatkeys, aux = flatten_with_keys(obj)
        return [c for _, c in flatkeys], aux

    def unflatten(aux_data, flat_contents):
        return cls(
            **dict(
                itertools.chain(
                    zip(fields, flat_contents, strict=True),
                    zip(static_fields, aux_data or (), strict=True),
                )
            )
        )

    jax.tree_util.register_pytree_with_keys(cls, flatten_with_keys, unflatten, flatten)
    return cls


def array_real_dtype(arr):
    return jax.eval_shape(jnp.real, arr).dtype
