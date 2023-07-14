.. highlight:: python

Parameterizations
=================

.. automodule:: pyqg_jax.parameterizations

.. toctree::
   :maxdepth: 1
   :caption: Existing Parameterizations

   reference.parameterizations.smagorinsky
   reference.parameterizations.zannabolton2020
   reference.parameterizations.backscatterbiharmonic
   reference.parameterizations.noop


.. _sec-ref-impl-param:

Implementing Parameterizations
------------------------------

We include some utilities for applying custom parameterizations to a
core model. User-defined parameterizations are implemented as a pair
of functions one computing the parameterized updates for the model,
and the other initializing an auxiliary state.

A sample `param_func` could be::

  def param_func(model_state, param_aux, model):
      # Access attributes and methods of the inner model
      updates = model.get_updates(model_state)
      # Compute updates and new auxiliary values
      new_updates = do_calculations(model_state, updates)
      new_param_aux = compute_new_aux(param_aux)
      # Return model state updates, and new aux values
      return new_updates, new_param_aux

and a sample `init_param_aux_func`::

  def init_param_aux_func(model_state, model, *args, **kwargs):
      # Initialize the aux state based on inner model_state
      # and additional arguments
      return make_new_aux_state(model_state)

These two work together. The auxiliary state is an arbitrary object
(must be a JAX PyTree). Simple choices are tuples of JAX values
(:class:`arrays <jax.Array>`, :func:`PRNGKey <jax.random.PRNGKey>`,
etc.) or immutable python objects (:class:`str <python:str>`,
:class:`bool <python:bool>`, etc.). The auxiliary state can be
:pycode:`None` if no values are necessary.

Your `param_func` is responsible for updating the auxiliary state as
needed. :class:`ParameterizedModel` will wrap the auxiliary state in a
:class:`NoStepValue <pyqg_jax.steppers.NoStepValue>` so the
time-steppers will not manipulate it.

The additional state is provided to allow propagating extra
non-time-stepped value forward when stepping the model. Some
possibilities:

* Stochastic parameterizations will need to include and :func:`split
  <jax.random.split>` a :func:`PRNGKey <jax.random.PRNGKey>` to use
  randomness.
* Stateful parameterizations could maintain a history of previous
  model states.
* Stateless, deterministic parameterizations can use :pycode:`None` as
  their auxiliary state.

Once you have implemented your parameterization, apply it to a base
model using :class:`ParameterizedModel`, which can then be used with a
:doc:`time stepper <reference.steppers>`.

.. autoclass:: ParameterizedModel
   :members: get_full_state, get_updates, postprocess_state, create_initial_state, initialize_param_state

.. autoclass:: ParameterizedModelState

We also provide decorators which can simplify the process of
implementing common parameterizations in terms of velocity or
potential vorticity.

.. autodecorator:: uv_parameterization

.. autodecorator:: q_parameterization
