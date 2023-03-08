Parameterizations
=================

.. automodule:: pyqg_jax.parameterizations

.. toctree::
   :maxdepth: 1
   :caption: Implemented Parameterizations

   reference.parameterizations.smagorinsky
   reference.parameterizations.zannabolton2020
   reference.parameterizations.noop

.. autoclass:: ParameterizedModel
   :members: get_full_state, get_updates, postprocess_state, create_initial_state, initialize_param_state

.. autoclass:: ParameterizedModelState

.. autodecorator:: uv_parameterization

.. autodecorator:: q_parameterization
