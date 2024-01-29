# Copyright Karl Otness
# SPDX-License-Identifier: MIT


"""Functions for computing diagnostics of simulation states.

The functions in this module can be used to compute diagnostic
quantities such as kinetic energy or various spectra. See
:doc:`examples.diagnostics` for examples of how to use them and plot
the results.
"""


import jax.numpy as jnp


__all__ = ["total_ke", "cfl"]


def _getattr_shape_check(full_state, attr, grid):
    if attr in {"q", "p", "u", "v", "dqdt"}:
        corr_shape = grid.real_state_shape
    else:
        corr_shape = grid.spectral_state_shape
    corr_dims = len(corr_shape)
    arr = getattr(full_state, attr)
    shape = jnp.shape(arr)
    dims = len(shape)
    if dims != corr_dims:
        vmap_msg = " (use jax.vmap)" if dims > corr_dims else ""
        raise ValueError(
            f"{attr} has {dims} dimensions but, should have {corr_dims}{vmap_msg}"
        )
    if shape != corr_shape:
        raise ValueError(f"{attr} has wrong shape {shape}, should be {corr_shape}")
    return arr


def total_ke(full_state, grid):
    """Compute the total kinetic energy in a single snapshot.

    The density in the KE calculation is taken such that the entire
    model grid space has a mass of one unit. To use a different
    density value, multiply the result of this calculation by the
    total mass of the full space.

    .. versionadded:: 0.8.0

    Parameters
    ----------
    full_state : FullPseudoSpectralState
        The state for which the kinetic energy is to be computed. This
        argument should be retrieved from a model, for example from
        :meth:`~pyqg_jax.qg_model.QGModel.get_full_state`.

        This function only operates on a single time step. To apply it
        to a trajectory use :func:`jax.vmap`.

    grid : Grid
        Information on the spatial grid for `full_state`. This should
        be retrieved from a model, for example from
        :meth:`~pyqg_jax.qg_model.QGModel.get_grid`.

    Returns
    -------
    float
        The total kinetic energy for the provided snapshot.
    """
    u = _getattr_shape_check(full_state, "u", grid)
    v = _getattr_shape_check(full_state, "v", grid)
    ke = (u**2 + v**2) / 2
    h_weights = jnp.expand_dims(grid.Hi / grid.H, axis=(-1, -2))
    return jnp.mean(jnp.sum(ke * h_weights, axis=-3), axis=(-1, -2))


def cfl(full_state, grid, ubg, dt):
    """Calculate the CFL condition value for a single snapshot.

    This computes the `CFL
    <https://en.wikipedia.org/wiki/Courant%E2%80%93Friedrichs%E2%80%93Lewy_condition>`__
    condition value at each grid point in a given state. To report the
    worst value across the full state, aggregate them using
    :func:`jnp.max <jax.numpy.max>`.

    .. versionadded:: 0.8.0

    Parameters
    ----------
    full_state : FullPseudoSpectralState
        The state for which the CFL condition is to be checked. This
        argument should be retrieved from a model, for example from
        :meth:`~pyqg_jax.qg_model.QGModel.get_full_state`.

        This function only operates on a single time step. To apply it
        to a trajectory use :func:`jax.vmap`.

    grid : Grid
        Information on the spatial grid for `full_state`. This should
        be retrieved from a model, for example from
        :meth:`~pyqg_jax.qg_model.QGModel.get_grid`.

    ubg : jax.Array
        The model's background velocity. Retrieve it from the same
        model as `full_state`, for example from
        :attr:`~pyqg_jax.qg_model.QGModel.Ubg`.

    dt : float
        The time step size. This should be retrieved from the relevant
        time stepper, for example form
        :attr:`~pyqg_jax.steppers.AB3Stepper.dt`.

    Returns
    -------
    jax.Array
        The CFL condition value at each spatial grid location. These
        may optionally be aggregated with :func:`jnp.max
        <jax.numpy.max>`.
    """
    u = (
        jnp.abs(
            _getattr_shape_check(full_state, "u", grid)
            + jnp.expand_dims(ubg, axis=(-1, -2))
        )
        / grid.dy
    )
    v = jnp.abs(_getattr_shape_check(full_state, "v", grid)) / grid.dx
    return dt * (u + v)
