Diagnostics
===========

.. automodule:: pyqg_jax.diagnostics

Scalar Diagnostics
------------------

These diagnostic functions produce a single scalar value for a single
snapshot.

.. autofunction:: total_ke

.. autofunction:: cfl

Spectral Diagnostics
--------------------

The diagnostics in this section produce array diagnostics which should
be :func:`averaged <jax.numpy.mean>` across a trajectory's time
dimension, and then further processed with :func:`calc_ispec`.

.. autofunction:: ke_spec_vals

.. autofunction:: ens_spec_vals

Computing Spectra
-----------------

These functions can be used to process the averaged values or a
spectral diagnostic into an isotropic spectrum.

.. autofunction:: calc_ispec

.. autofunction:: ispec_grid
