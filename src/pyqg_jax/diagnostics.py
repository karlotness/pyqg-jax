# Copyright Karl Otness
# SPDX-License-Identifier: MIT


"""Functions for computing diagnostics of simulation states.

The functions in this module can be used to compute diagnostic
quantities such as kinetic energy or various spectra. See
:doc:`examples.diagnostics` for examples of how to use them and plot
the results.
"""


import jax.numpy as jnp
from . import _spectral


__all__ = [
    "total_ke",
    "cfl",
    "ke_spec_vals",
    "ispec_grid",
    "calc_ispec",
]


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


def ke_spec_vals(full_state, grid):
    """Calculate the kinetic energy spectrum values for a snapshot.

    The values produced by this function should be further processed
    by :func:`calc_ispec` to produce the kinetic energy spectrum.

    .. versionadded:: 0.8.0

    Parameters
    ----------
    full_state : FullPseudoSpectralState
        The state for which the KE spectrum values should be computed.
        This argument should be retrieved from a model, for example
        from :meth:`~pyqg_jax.qg_model.QGModel.get_full_state`.

        This function only operates on a single time step. To apply it
        to a trajectory use :func:`jax.vmap`.

    grid : Grid
        Information on the spatial grid for `full_state`. This should
        be retrieved from a model, for example from
        :meth:`~pyqg_jax.qg_model.QGModel.get_grid`.

    Returns
    -------
    jax.Array
        The KE spectrum values for the provided time step.

    Note
    ----
    The returned array should be treated as opaque. Values should only
    be averaged over any vmapped time dimensions, then passed to
    :func:`calc_ispec`.
    """
    ph = _getattr_shape_check(full_state, "ph", grid)
    M = grid.nx * grid.ny
    abs_ph = jnp.abs(ph)
    return grid.get_kappa(abs_ph.dtype) ** 2 * abs_ph**2 / M**2


def ispec_grid(grid):
    """Information on the spacing of values in an isotropic spectrum.

    This function produces to results: `iso_k` and `keep`. The values
    `iso_k` are the isotropic wavenumbers for each entry in the result
    of `calc_ispec`. The result `keep` is an integer which should be
    used to slice the result of `calc_ispec`. Only the first `keep`
    entries should be interpreted.

    These values from this function are useful when plotting the
    result of :func:`calc_ispec`.

    .. versionadded:: 0.8.0

    Parameters
    ----------
    grid : Grid
        The spatial grid over which the base values were defined. This
        should be retrieved from a model, for example from
        :meth:`~pyqg_jax.qg_model.QGModel.get_grid`.

    Returns
    -------
    iso_k : jax.Array
        The isotropic wavenumbers for each spectrum entry.

    keep : int
        An integer indicating how many of the first spectrum entries
        should be interpreted or plotted.
    """
    iso_k, keep = _spectral.get_plot_kr(grid, truncate=True)
    return iso_k, keep


def calc_ispec(spec_vals, grid):
    """Compute the isotropic spectrum from the given values.

    The array `spec_vals` should have been computed by one of the
    spectral diagnostics functions--for example :func:`ke_spec_vals`.

    To correctly plot or interpret the spectrum computed by this
    function, use the result of :func:`ispec_grid`.

    .. versionadded:: 0.8.0

    Parameters
    ----------
    spec_vals : jax.Array
        The input values which should be processed into an isotropic
        spectrum. These values should be a squared modulus of the
        Fourier coefficients.

    grid : Grid
        The spatial grid over which the base values were defined. This
        should be retrieved from a model, for example from
        :meth:`~pyqg_jax.qg_model.QGModel.get_grid`.

    Returns
    -------
    jax.Array
        A one-dimensional array providing the isotropic spectrum of
        `spec_vals`.
    """
    shape = spec_vals.shape
    corr_shape = grid.spectral_state_shape
    if shape != corr_shape:
        raise ValueError(
            f"mismatched shape for calc_ispec, expected {corr_shape} but got {shape}"
        )
    return _spectral.calc_ispec(spec_vals, grid, averaging=True, truncate=True)
