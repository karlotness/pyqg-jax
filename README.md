# PyQG JAX Port

[![PyQG-JAX on PyPI](https://img.shields.io/pypi/v/pyqg-jax)][pypi]
[![PyQG-JAX on conda-forge](https://img.shields.io/conda/vn/conda-forge/pyqg-jax.svg)][condaforge]
[![Documentation](https://readthedocs.org/projects/pyqg-jax/badge/?version=latest)][docs]
[![Tests](https://github.com/karlotness/pyqg-jax/actions/workflows/test.yml/badge.svg)][tests]
[![Zenodo](https://zenodo.org/badge/523137021.svg)][zenodo]

This is a partial port of [PyQG](https://github.com/pyqg/pyqg) to
[JAX](https://github.com/google/jax) which enables GPU acceleration,
batching, automatic differentiation, etc.

- **Documentation:** https://pyqg-jax.readthedocs.io/en/latest/
- **Source Code:** https://github.com/karlotness/pyqg-jax
- **Bug Reports:** https://github.com/karlotness/pyqg-jax/issues

⚠️ **Warning:** this is a partial, early stage port. There may be bugs
and other numerical issues. The API may evolve as work continues.

## Installation
Install from [PyPI][pypi] using pip:
```console
$ python -m pip install pyqg-jax
```
or from [conda-forge][condaforge]:
``` console
$ conda install -c conda-forge pyqg-jax
```
This should install required dependencies, but JAX itself may require
special attention, particularly for GPU support.
Follow the [JAX installation instructions](https://docs.jax.dev/en/latest/installation.html).

## Usage
[Documentation][docs] is a work in progress. The parameters `QGModel`
implemented here are the same as for the model in the original PyQG,
so consult the [pyqg
documentation](https://pyqg.readthedocs.io/en/latest/) for details.

However, there are a few overarching changes used to make the models
JAX-compatible:

1. The model state is now a separate, immutable object rather than
   being attributes of the `QGModel` class

2. Time-stepping is now separated from the models. Use
   `steppers.AB3Stepper` for the same time stepping as in the original
   `QGModel`.

3. Random initialization requires an explicit `key` variable as with
   all JAX random number generation.

The `QGModel` uses double precision (`float64`) values for part of its
computation regardless of the precision setting. Make sure JAX is set
to enable 64-bit. [See the
documentation](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision)
for details. One option is to set the following environment variable:
```bash
export JAX_ENABLE_X64=True
```
or use the [`%env`
magic](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-env)
in a Jupyter notebook.

### Short Example
A short example initializing a `QGModel`, adding a parameterization,
and taking a single step (for more, see the
[examples](https://pyqg-jax.readthedocs.io/en/latest/examples.html) in
the documentation).
```pycon
>>> import pyqg_jax
>>> import jax
>>> # Construct model, parameterization, and time-stepper
>>> stepped_model = pyqg_jax.steppers.SteppedModel(
...     model=pyqg_jax.parameterizations.smagorinsky.apply_parameterization(
...         pyqg_jax.qg_model.QGModel(),
...         constant=0.08,
...     ),
...     stepper=pyqg_jax.steppers.AB3Stepper(dt=3600.0),
... )
>>> # Initialize the model state (wrapped in stepper and parameterization state)
>>> stepper_state = stepped_model.create_initial_state(
...     jax.random.key(0)
... )
>>> # Compute next state
>>> next_stepper_state = stepped_model.step_model(stepper_state)
>>> # Unwrap the result from the stepper and parameterization
>>> next_param_state = next_stepper_state.state
>>> next_model_state = next_param_state.model_state
>>> final_q = next_model_state.q
```
For repeated time-stepping combine `step_model` with
[`jax.lax.scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html).

## License
This software is distributed under the MIT license. See LICENSE.txt
for the license text.

[pypi]: https://pypi.org/project/pyqg-jax
[condaforge]: https://anaconda.org/conda-forge/pyqg-jax
[docs]: https://pyqg-jax.readthedocs.io/en/latest/
[tests]: https://github.com/karlotness/pyqg-jax/actions
[zenodo]: https://zenodo.org/badge/latestdoi/523137021
