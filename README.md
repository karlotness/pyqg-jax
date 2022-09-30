# PyQG JAX Port

This is a partial port of [PyQG](https://github.com/pyqg/pyqg) to
[JAX](https://github.com/google/jax) which enables GPU acceleration,
batching, automatic differentiation, etc.

⚠️ **Warning:** this is a partial, early stage port. There may be bugs
and other numerical issues. Only part of the `QGModel` has been
ported.

## Dependencies
To run the ported code you will need a few dependencies:
- `jax`
- `numpy`
