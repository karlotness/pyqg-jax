import math
import warnings
import pytest
import pyqg
import numpy as np
import pyqg_jax


@pytest.mark.parametrize(
    "param",
    [
        "nz",
        "ny",
        "nx",
        "rek",
        "L",
        "W",
        "filterfac",
        "f",
        "g",
        "beta",
        "rd",
        "delta",
        "H1",
        "U1",
        "U2",
    ],
)
def test_default_parameters_match(param):
    jax_model = pyqg_jax.qg_model.QGModel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        orig_model = pyqg.QGModel(log_level=0)
    jax_param = getattr(jax_model, param)
    if param == "f":
        assert jax_param is None
        assert not hasattr(orig_model, "f")
        return
    elif param == "H1":
        orig_param = orig_model.Hi[0]
    else:
        orig_param = getattr(orig_model, param)
    if isinstance(orig_param, np.ndarray):
        np_jax_param = np.asarray(jax_param)
        assert jax_param.shape == orig_param.shape
        assert np_jax_param.dtype == orig_param.dtype
        assert np.allclose(np_jax_param, orig_param)
    elif isinstance(orig_param, float):
        assert math.isclose(orig_param, float(jax_param))
    elif isinstance(orig_param, int):
        assert jax_param == orig_param
    else:
        raise ValueError(f"unhandled type for attribute {param}")
