pyqg-jax: Quasigeostrophic Model in JAX
=======================================

This is the documentation for the `pyqg-jax` package, a port of `PyQG
<https://pyqg.readthedocs.io/en/latest/>`__ to `JAX
<https://docs.jax.dev/en/latest/>`__.

Porting the model to JAX makes it possible to run it on GPU, and apply
JAX transformations including :func:`jax.jit` and :func:`jax.vmap`.
This also makes it possible to :doc:`integrate learned
parameterizations <examples.implparam>` into the model, or :doc:`train
online <examples.onlinetrain>` through the simulation using
:func:`jax.grad` to take gradients.

That said, a note on the state of the port:

.. warning::
   This is a partial, early stage port. There may be bugs and other
   numerical issues. The API may evolve as work continues.

Even so, we hope that the port will be useful. We have successfully
made use of it in ongoing research projects, and hope that others can
do so as well.


.. toctree::
   :maxdepth: 1
   :caption: Contents

   install
   examples
   reference
   changes
   license

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
