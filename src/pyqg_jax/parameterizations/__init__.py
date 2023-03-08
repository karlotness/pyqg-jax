# Copyright Karl Otness
# SPDX-License-Identifier: MIT


"""Utilities for adding parameterizations to core models.

This subpackage includes some of the pre-implemented parameterizations
from PyQG as well as utilities for defining custom parameterizations
and applying them to a model.
"""


__all__ = [
    "ParameterizedModelState",
    "ParameterizedModel",
    "uv_parameterization",
    "q_parameterization",
    "smagorinsky",
    "zannabolton2020",
    "noop",
]


from ._parameterized_model import ParameterizedModelState, ParameterizedModel
from ._defs import uv_parameterization, q_parameterization
from . import smagorinsky, zannabolton2020, noop
