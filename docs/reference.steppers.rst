Time-Steppers
=============

.. automodule:: pyqg_jax.steppers


Combined Models
---------------

These utilities add time-stepping to base models.

.. autoclass:: SteppedModel
   :members: create_initial_state, initialize_stepper_state, step_model, get_full_state

.. autoclass:: StepperState
   :members: update

Time Stepping Schemes
---------------------

Implemented time-stepping schemes (currently only :class:`AB3Stepper`).

.. autoclass:: AB3Stepper
   :members: initialize_stepper_state, apply_updates
   :inherited-members:

State Manipulation
------------------

:class:`NoStepValue` makes it possible to shield values from
time-stepping so they can be updated manually.

.. autoclass:: NoStepValue
