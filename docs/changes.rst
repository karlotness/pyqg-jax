Changelog
=========

This document provides a brief summary of changes in each released
version of `pyqg-jax`. More information and release builds are also
available on the `GitHub releases page
<https://github.com/karlotness/pyqg-jax/releases>`__.

v0.9.0 (Unreleased)
-------------------

* Add enstrophy spectrum calculation
  :func:`~pyqg_jax.diagnostics.ens_spec_vals`
* Improve some error messages (include additional details)
* :class:`~pyqg_jax.state.Precision` enum members now have attributes
  :pycode:`dtype_real` and :pycode:`dtype_complex` storing the dtypes
  used at each precision level.
* *Breaking:* Require Python 3.10 or later

.. note::
   In this release, state class :pycode:`__init__` parameters are now
   :term:`keyword-only <python:parameter>`. If you are obtaining these
   from model classes as recommended this should require no changes.
   However if you construct these classes manually, make sure all
   arguments are passed as *keyword* arguments. This affects
   :class:`~pyqg_jax.state.PseudoSpectralState`,
   :class:`~pyqg_jax.state.FullPseudoSpectralState`,
   :class:`~pyqg_jax.parameterizations.ParameterizedModelState` and
   :class:`~pyqg_jax.steppers.StepperState`.

.. note::
  This release includes internal changes in
  :class:`~pyqg_jax.steppers.AB3Stepper` and
  :class:`~pyqg_jax.steppers.EulerStepper` which ensure consistent
  behavior regardless of the :pycode:`dt` parameter's precision and
  preventing errors where it does not have :term:`weak type <jax:weak
  type>` (i.e. when it is not a Python :class:`float`). This may
  affect calculations and exact trajectories due to minor numeric
  changes and possible differences in JIT behavior, but should not
  cause other breakage.

v0.8.1
------
* Resolve deprecation warnings from :func:`jax.numpy.linalg.solve`
  added in JAX v0.4.25

v0.8.0
------
* Add :class:`~pyqg_jax.steppers.EulerStepper`
* Add :mod:`pyqg_jax.diagnostics` module (see documentation and
  associated :doc:`example <examples.diagnostics>` for more
  information)
* New :class:`~pyqg_jax.state.Grid` class for use with diagnostics
* Fix incompatibility with JAX v0.4.24
* Fix shape errors for models with non-square states (this setting is
  still less well-tested and not recommended)

.. note::
   This release adds an internal, hidden static field to the
   :class:`~pyqg_jax.state.PseudoSpectralState` class. This field is
   an implementation detail, and if all instances are constructed from
   model classes (:meth:`model.create_initial_state
   <pyqg_jax.qg_model.QGModel.create_initial_state>`) this shouldn't
   cause issues and should require no attention. However, if you were
   constructing these objects manually using their constructors this
   will be a *breaking* change.

v0.7.0
------
* Add implementation of :class:`~pyqg_jax.sqg_model.SQGModel` from
  PyQG
* Integrate with JAX pytree `key paths
  <https://docs.jax.dev/en/latest/working-with-pytrees.html#explicit-key-paths>`__
* Improved summary formatting of built-in Python collections
* *Breaking:* Drop support for Python 3.8
* *Breaking:* Remove uq and vq attributes from
  :class:`~pyqg_jax.state.FullPseudoSpectralState`

v0.6.0
------
* Clearer error messages when using model states with the wrong shape
* Add implementation of :class:`~pyqg_jax.bt_model.BTModel` from PyQG

v0.5.1
------
* Add properties for missing full state attributes
  :attr:`~pyqg_jax.state.FullPseudoSpectralState.p` and
  :attr:`~pyqg_jax.state.FullPseudoSpectralState.dqdt`
* Summarize state objects without using computed properties

v0.5.0
------
* Fix bug that caused
  :func:`~pyqg_jax.parameterizations.q_parameterization` decorator to
  drop the auxiliary state
* Add :mod:`backscatter biharmonic
  <pyqg_jax.parameterizations.backscatterbiharmonic>` parameterization
  from PyQG

v0.4.0
------
* Add docstrings to most public API
* Rename :pycode:`ParametrizedModel` to
  :class:`~pyqg_jax.parameterizations.ParameterizedModel`
* Rename :pycode:`ParametrizedModelState` to
  :class:`~pyqg_jax.parameterizations.ParameterizedModelState`

v0.3.0
------
* Add :pycode:`__repr__` methods to most classes showing nested states
  and models
* Add a no-op :mod:`~pyqg_jax.parameterizations.noop`
  parameterization

v0.2.0
------
* Parameterizations now receive the "partial" model state, and call
  :meth:`model.get_full_state
  <pyqg_jax.qg_model.QGModel.get_full_state>` to expand it
* Fix propagation and unwrapping of parameterization states during
  time-stepping
* Move :class:`~pyqg_jax.steppers.NoStepValue` into
  steppers module
* Remove repeated names from parameterization functions

v0.1.0
------
Initial release
