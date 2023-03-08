# Copyright Karl Otness
# SPDX-License-Identifier: MIT


"""The functionality of `pyqg-jax` is split into several modules.

Generally, the operation of each of the base models is the same as the
equivalent in PyQG so its `documentation
<https://pyqg.readthedocs.io/en/latest/>`__ can be used as a
reference.

However as part of the port, some changes to the overall structure
have been made. These changes both improve compatibility with JAX and
separate portions of the model that research projects may wish to
modify (in particular, the time-stepping).

The expected steps to construct a model and step it forward are:

#. Select a :doc:`base model <reference.models>`
#. Apply a :doc:`parameterization <reference.parameterizations>`
   (optional)
#. Select a :doc:`time stepper <reference.steppers>`
#. Combine into a :class:`SteppedModel
   <pyqg_jax.steppers.SteppedModel>`

New states can then be randomly initialized, and the model can be
stepped forward in time.
"""


__version__ = "0.5.0"
__all__ = ["state", "qg_model", "steppers", "parameterizations"]


from . import state, qg_model, steppers, parameterizations
