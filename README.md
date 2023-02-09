# PyQG JAX Port

This is a partial port of [PyQG](https://github.com/pyqg/pyqg) to
[JAX](https://github.com/google/jax) which enables GPU acceleration,
batching, automatic differentiation, etc.

⚠️ **Warning:** this is a partial, early stage port. There may be bugs
and other numerical issues. Only part of the `QGModel` has been
ported.

## Installation
Install from PyPI using pip:
```console
$ python -m pip install pyqg-jax
```
This should install required dependencies, but JAX itself may require
special attention. Follow the [JAX installation
instructions](https://github.com/google/jax#installation).

## Usage
Documentation is a work in progress. The parameters `QGModel`
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
documentation](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision)
for details. One option is to set the following environment variables:
```bash
export JAX_ENABLE_X64=True
export JAX_DEFAULT_DTYPE_BITS=32
```
or use the [`%env`
magic](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-env)
in a Jupyter notebook.

### Short Example
A short example initializing a `QGModel` and taking a single step.
```pycon
>>> import pyqg_jax
>>> import jax
>>> # Construct model and time-stepper
>>> model = pyqg_jax.qg_model.QGModel()
>>> stepper = steppers.AB3Stepper(dt=7200.0)
>>> # Initialize the model state, and wrap in the time-stepping state
>>> stepper_state = stepper.initialize_stepper_state(
...     model.create_initial_state(jax.random.PRNGKey(0))
... )
>>> # Compute next state
>>> state_updates = model.get_updates(stepper_state.state)
>>> next_stepper_state = stepper.apply_updates(stepper_state, state_updates)
>>> # Extract the result
>>> final_q = next_stepper_state.state.q
```
For repeated time-stepping combine `stepper.apply_updates` with
[`jax.lax.scan`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html).

### Useful Methods and attributes
A subset of methods and attributes available on common objects
- For `pyqg_jax.qg_model.QGModel`
  - `create_initial_state(jax.random.PRNGKey) -> PseudoSpectralState`: Randomly initializes the model
  - `get_full_state(PseudoSpectralState) -> FullPseudoSpectralState`: Expands the state, computing other attributes from `q`
  - `get_updates(PseudoSpectralState) -> PseudoSpectralState`: Computes time updates for `qh`. Combine with a time-stepper
- For `pyqg_jax.steppers.AB3Stepper`
  - `initialize_stepper_state(PseudoSpectralState) -> AB3State[PseudoSpectralState]`: Initialize a time-stepper state around a model state
  - `apply_updates(AB3State[PseudoSpectralState], updates=PseudoSpectralState) -> AB3State[PseudoSpectralState]`: Apply model updates to a time stepper state
- For `pyqg_jax.steppers.AB3State`
  - `state`: extract the `PseudoSpectralState` at the current time
  - `t`: the current time
  - `tc`: the current step counter
- For `pyqg_jax.state.PseudoSpectralState`
  - `q`: The potential vorticity
  - `qh`: Spectral form of potential vorticity
  - `update(q=, qh=) -> PseudoSpectralState`: Return a new `PseudoSpectralState` with the given value replacements
- For `pyqg_jax.state.FullPseudoSpectralState`
  - `dqhdt`: Spectral updates for `qh`
  - `state`: The inner `PseudoSpectralState`
  - `update(q=, qh=, dqhdt=, ...) -> PseudoSpectralState`: Return a new `FullPseudoSpectralState` with the given value replacements

## License
The code in this repository is distributed under the MIT license. See
LICENSE.txt for the license text.
