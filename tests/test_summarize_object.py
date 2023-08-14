import warnings
import functools
import numpy as np
import re
import ast
import jax.config
import jax.numpy as jnp
import pytest
from pyqg_jax._utils import summarize_object


@pytest.mark.parametrize(
    "dtype, jax_dtype_key, np_dtype_key",
    [
        (jnp.float32, "f32", "f32"),
        (jnp.complex64, "c64", "c64"),
        (jnp.int16, "i16", "i16"),
        (jnp.uint8, "u8", "u8"),
        (jnp.float64, "f64" if jax.config.jax_enable_x64 else "f32", "f64"),
        (jnp.complex128, "c128" if jax.config.jax_enable_x64 else "c64", "c128"),
    ],
)
@pytest.mark.parametrize("shape", [(3,), 5, (4, 5, 6), (8, 0), (0,), 0, ()])
def test_summarize_array(dtype, jax_dtype_key, np_dtype_key, shape):
    with warnings.catch_warnings():
        if not jax.config.jax_enable_x64:
            warnings.simplefilter("ignore", category=UserWarning)
        jax_arr = jnp.zeros(shape, dtype=dtype)
    np_arr = np.zeros(shape, dtype=dtype)
    try:
        shape_str = ",".join(map(str, shape))
    except TypeError:
        shape_str = str(shape)
    jax_summary = summarize_object(jax_arr)
    np_summary = summarize_object(np_arr)
    assert jax_summary == np_summary or jax_dtype_key != np_dtype_key
    assert jax_summary == f"{jax_dtype_key}[{shape_str}]"
    assert np_summary == f"{np_dtype_key}[{shape_str}]"


@pytest.mark.parametrize("cls", [functools.partial, jax.tree_util.Partial])
def test_partial(cls):
    def test_function_name():
        pass

    obj = cls(test_function_name, "value1", kwarg2="value2")
    rgx = (
        rf"^.+?{re.escape(cls.__name__)}"
        r"\(<function .+test_function_name>, "
        rf"{re.escape(repr('value1'))}, "
        rf"kwarg2={re.escape(repr('value2'))}\)$"
    )
    assert re.match(rgx, summarize_object(obj))


def test_function():
    def test_function_name():
        pass

    func_name = f"{test_function_name.__module__}.{test_function_name.__qualname__}"
    assert summarize_object(test_function_name) == f"<function {func_name}>"


@pytest.mark.parametrize("cls", [list, tuple])
def test_ordered_seq(cls):
    val = cls(
        [
            jnp.zeros((1, 2, 3), dtype=jnp.float32),
            jnp.zeros((4,), dtype=jnp.complex64),
            6,
            True,
        ]
    )
    separators = repr(cls())
    summary = summarize_object(val)
    assert summary[0] == separators[0]
    assert summary[-1] == separators[-1]
    assert summary[1:-1] == "f32[1,2,3], c64[4], 6, True"


def test_set():
    def test_function_name():
        pass

    val = {
        test_function_name,
        6,
        4.0,
        True,
    }
    summary = summarize_object(val)
    summary_elements = {s.strip() for s in summary[1:-1].split(",")}
    assert summary[0] == "{"
    assert summary[-1] == "}"
    assert summary_elements == {
        "6",
        "4.0",
        "True",
        summarize_object(test_function_name),
    }


def test_dict():
    val = {
        "float_arr": jnp.zeros((3,), dtype=jnp.float32),
        "complex_arr": jnp.zeros((4,), dtype=jnp.complex64),
        "bool": True,
        "int": 5,
    }
    summary = summarize_object(val)
    summary_strings = {}
    for element in summary[1:-1].split(","):
        parts = element.split(":")
        name = ast.literal_eval(parts[0])
        value = parts[1].strip()
        summary_strings[name] = value
    assert summary[0] == "{"
    assert summary[-1] == "}"
    assert summary_strings == {
        "float_arr": "f32[3]",
        "complex_arr": "c64[4]",
        "bool": "True",
        "int": "5",
    }


@pytest.mark.parametrize("obj", [3, 4.0, True, "test"])
def test_normal_repr(obj):
    assert summarize_object(obj) == repr(obj)
