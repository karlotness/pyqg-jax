---
file_format: mystnb
kernelspec:
  name: python3
---

# Diagnostics

This page presents several examples illustrating the use of diagnostic
routines included in this package. The available functions are
included in the {mod}`pyqg_jax.diagnostics` module.

```{code-cell} ipython3
:tags: [remove-cell]
# Note: docs builds only have CPUs
# This suppresses the JAX warning about missing GPU
# If you're running this with a GPU, delete this cell
%env JAX_PLATFORMS=cpu
```

```{code-cell} ipython3
%env JAX_ENABLE_X64=True
import functools
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import powerpax
import pyqg_jax
```

We will illustrate the use of these routines on an sample trajectory.
To begin we construct a {class}`~pyqg_jax.qg_model.QGModel` and
produce an initial state.

```{code-cell} ipython3
stepped_model = pyqg_jax.steppers.SteppedModel(
    model=pyqg_jax.qg_model.QGModel(
        nx=64,
        ny=64,
    ),
    stepper=pyqg_jax.steppers.AB3Stepper(dt=14400.0),
)
stepper_state = stepped_model.create_initial_state(
    jax.random.key(0)
)
```

Next, we produce the trajectory. To reduce the required memory we will
not keep each step. Instead we will use {func}`powerpax.sliced_scan`
to subsample them keeping states only at regular intervals. We also
collect the time for each step for use in plotting.

```{code-cell} ipython3
@functools.partial(
    jax.jit,
    static_argnames=["num_steps", "start", "stride"]
)
def roll_out_state(init_state, num_steps, start, stride):
    def loop_fn(carry, _x):
        current_state = carry
        next_state = stepped_model.step_model(current_state)
        return next_state, (next_state.state, next_state.t)

    _, (traj, t) = powerpax.sliced_scan(
        loop_fn,
        init=init_state,
        xs=None,
        length=num_steps,
        start=start,
        step=stride,
    )
    return traj, t

traj, t = roll_out_state(
    stepper_state, num_steps=10000, start=0, stride=250
)

traj
```

Note that the trajectory has a leading dimension for the steps just as
in {doc}`examples.basicstep`.

## Total Kinetic Energy

The function {func}`pyqg_jax.diagnostics.total_ke` can be used to
calculate the total kinetic energy in a particular state (see the
function's documentation for information on scaling the value to
reflect a particular density).

The provided function operates only on one state at a time so here we
use {func}`powerpax.chunked_vmap` to vectorize it across several
states at once. This function is used to limit the number of steps
computed in parallel to reduce peak memory use in cases of a very long
trajectory. The value of `chunk_size` should be configured to balance
performance on GPUs against the memory required for the intermediate
buffers. Alternatively {func}`jax.vmap` could also be used to compute
the diagnostic across steps.

```{code-cell} ipython3
def compute_ke(state, model):
    full_state = model.get_full_state(state)
    return pyqg_jax.diagnostics.total_ke(full_state, model.get_grid())

@jax.jit
def vectorized_ke(traj, model):
    return powerpax.chunked_vmap(
        functools.partial(compute_ke, model=model), chunk_size=100
    )(traj)

traj_ke = vectorized_ke(traj, stepped_model.model)
```

Finally we can plot the kinetic energy for each simulation step
against the simulation time in years.

```{code-cell} ipython3
plt.plot(t / 31536000, traj_ke)
plt.xlabel("Time (yr)")
plt.ylabel("Kinetic Energy")
plt.grid()
```

## CFL Condition

The function {func}`pyqg_jax.diagnostics.cfl` computes the
[CFL](https://en.wikipedia.org/wiki/Courant%E2%80%93Friedrichs%E2%80%93Lewy_condition)
condition value of a particular step at each location in the grid. The
sample function below vectorizes it across several steps (with
{func}`~powerpax.chunked_vmap`). The code below also demonstrates
reporting the highest CFL value for a given step using {func}`jnp.max
<jax.numpy.max>`.

```{code-cell} ipython3
def compute_cfl(state, model, dt):
    full_state = model.get_full_state(state)
    cfl = pyqg_jax.diagnostics.cfl(
        full_state=full_state,
        grid=model.get_grid(),
        ubg=model.Ubg,
        dt=dt,
    )
    return jnp.max(cfl)

@jax.jit
def vectorized_cfl(traj, stepped_model):
    return powerpax.chunked_vmap(
        functools.partial(
            compute_cfl, model=stepped_model.model, dt=stepped_model.stepper.dt
        ),
        chunk_size=100,
    )(traj)

traj_cfl = vectorized_cfl(traj, stepped_model)
```

Finally, we plot the CFL values for each step.

```{code-cell} ipython3
plt.plot(t / 31536000, traj_cfl)
plt.xlabel("Time (yr)")
plt.ylabel("CFL Condition")
plt.grid()
```
