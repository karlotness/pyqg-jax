States
======

.. automodule:: pyqg_jax.state


.. autoclass:: PseudoSpectralState
   :members: update

.. autoclass:: FullPseudoSpectralState
   :members: update

Model instances can have their data-type precision selected as a
constructor argument. The enumeration :class:`Precision` gives
available options.

.. autoclass:: Precision

Models also expose information about the grid on which values are
computed.

.. autoclass:: Grid
   :members: get_kappa
