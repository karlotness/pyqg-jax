pyqg-jax: Quasigeostrophic model in JAX
=======================================

This is the documentation for the `pyqg-jax` package, a port of `PyQG
<https://pyqg.readthedocs.io/en/latest/>`__ to `JAX
<https://jax.readthedocs.io/en/latest/>`__.

Porting the model to JAX makes it possible to run it on GPU, and apply
JAX transformations including :func:`jax.jit` and :func:`jax.vmap`.
This also makes it possible to integrate learned parameterizations
into the model, or train online through the simulation using
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
   reference
   license

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
