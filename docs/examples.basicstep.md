---
file_format: mystnb
kernelspec:
  name: python3
---

# Basic Time Stepping

In this example we demonstrate basic use of this package and how to
initialize one of the models and step a simulation through time.

For this example we will run some calculations using `float64`. We
need to enable JAX's double precision support by setting a few
environment variables before importing anything.

```{code-cell} ipython3
:tags: [remove-cell]
# Note: docs builds only have CPUs
# This suppresses the JAX warning about missing GPU
# If you're running this with a GPU, delete this cell
%env JAX_PLATFORM_NAME=cpu
```

```{code-cell} ipython3
%env JAX_ENABLE_X64=True
%env JAX_DEFAULT_DTYPE_BITS=32
```

Any time you are using double precision support, you should set these
environment variables before you import JAX.

With that done we can begin by importing JAX, and the `pyqg_jax`
package.

```{code-cell} ipython3
import operator
import functools
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import pyqg_jax
```

## Constructing a Model

In an effort to make this package as flexible as possible for use in
research projects, there are several components that must be combined
to produce a model which is ready for time stepping:

1. A base model determining the behavior of the simulation
2. An (optional) parameterization
3. A time stepper

We will show how to combine these to produce a final
{class}`SteppedModel <pyqg_jax.steppers.SteppedModel>`.

First we construct the base model. This includes setting the precision
and shape of the state variables, as well as the physical parameters.
In this example we will be using the {class}`QGModel
<pyqg_jax.qg_model.QGModel>` which is a two-layer quasi-geostrophic
model.

```{code-cell} ipython3
base_model = pyqg_jax.qg_model.QGModel(
    nx=64,
    ny=64,
    precision=pyqg_jax.state.Precision.DOUBLE,
)

base_model
```

Notice how the printed description of the model shows the current
value of all parameters. These values are accessible as attributes on
the `base_model` object.

In this initial example we will apply a {mod}`Smagorinsky
<pyqg_jax.parameterizations.smagorinsky>` parameterization, but in
your own use you can skip this step.

```{code-cell} ipython3
param_model = pyqg_jax.parameterizations.smagorinsky.apply_parameterization(
    base_model, constant=0.08,
)
```

Finally, we can combine this model with a time stepper.

```{code-cell} ipython3
# Note that this time step was made larger for demonstration purposes
stepper = pyqg_jax.steppers.AB3Stepper(dt=14400.0)

stepped_model = pyqg_jax.steppers.SteppedModel(
    param_model, stepper
)
```

We can examine the final combined `stepped_model` object:

```{code-cell} ipython3
stepped_model
```

This description is quite long, but it shows all available attributes
and how the several components have been nested. If you are unsure of
how objects have been combined and need information on how they are
nested, printing the objects can provide some guidance.

## Initializing States

We can initialize a state directly from the stepped model. Its printed
representation also includes abbreviated information on the shape and
data type of its contents. Consult the documentation on
{class}`PseudoSpectralState <pyqg_jax.state.PseudoSpectralState>` for
information on additional attributes and properties.

```{code-cell} ipython3
init_state = stepped_model.create_initial_state(
    jax.random.PRNGKey(0)
)

init_state
```

We can plot the initial conditions. Note that these initial states do
not resemble the states after several warmup time steps. See below for
a sample of a more typical state produced after several time steps.

```{code-cell} ipython3
inner_state = init_state.state.model_state
for layer in range(2):
    data = inner_state.q[layer]
    vmax = jnp.abs(data).max()
    plt.subplot(1, 2, layer + 1)
    plt.title(f"Layer {layer}")
    plt.imshow(data, cmap="bwr", vmin=-vmax, vmax=vmax)
```

This state can now be stepped forward in time to produce a trajectory.
Generating the initial condition uses a {func}`jax.random.PRNGKey` for
random number generation.

### Wrapping an External Array

In more advanced use cases you might need to initialize a
`PseudoSpectralState` from a raw array. The best way to do this is to
obtain a state from the `base_model` and *replace* its contents:

```{code-cell} ipython3
# Stand in for an externally-computed value (perhaps from a file)
new_q = jnp.linspace(0, 1, 64 * 64 * 2, dtype=jnp.float64).reshape((2, 64, 64))
# Create a state and perform the replacement
dummy_state = base_model.create_initial_state(jax.random.PRNGKey(0))
base_state = dummy_state.update(q=new_q)

base_state
```

This produces a new state with the value we provided. However it is
not wrapped in for use with the parameterization or the time stepper.
We need to pass it up through both of these:

```{code-cell} ipython3
# Skip this next line if you didn't use a parameterization
wrapped_in_param = param_model.initialize_param_state(base_state)
wrapped_in_stepper = stepped_model.initialize_stepper_state(wrapped_in_param)

wrapped_in_stepper
```

Notice how the state is now wrapped just like `init_state` above.
Consult the documentation for {meth}`initialize_param_state
<pyqg_jax.parameterizations.ParameterizedModel.initialize_param_state>`
{meth}`initialize_stepper_state
<pyqg_jax.steppers.SteppedModel.initialize_stepper_state>` for more
information.

## Generate Full Trajectories

Now that we have our `stepped_model` and `init_state` we can generate
a trajectory by stepping forward in time. The most natural way to do
this is to perform the stepping using {func}`jax.lax.scan`.

```{tip}
For long trajectories with many steps you may wish to keep only a
subset or skip a warmup phase. One solution is to use
{func}`powerpax.sliced_scan` and set *start* and *step* to subsample
the trajectory.
```

```{code-cell} ipython3
@functools.partial(jax.jit, static_argnames=["num_steps"])
def roll_out_state(state, num_steps):

    def loop_fn(carry, _x):
        current_state = carry
        next_state = stepped_model.step_model(current_state)
        return next_state, next_state

    _final_carry, traj_steps = jax.lax.scan(
        loop_fn, state, None, length=num_steps
    )
    return traj_steps
```

Notice how we had to make `num_steps` a compile time constant since
this affects the shape of the result. We use {func}`jax.jit` here for
the best performance.

With this we can roll out our trajectory for several steps:

```{code-cell} ipython3
traj = roll_out_state(init_state, num_steps=7500)

traj
```

Notice how all the attributes have a leading dimension of `7500`. This
is the time dimension for each array. These are stored in
[struct-of-arrays](https://en.wikipedia.org/wiki/AoS_and_SoA) format.

To slice into these, the simplest approach is to use
{func}`jax.tree_util.tree_map` to apply a slice to each element.

```{code-cell} ipython3
jax.tree_util.tree_map(lambda leaf: leaf[-5:], traj)
```

or equivalently we can use {func}`operator.itemgetter` and
{class}`slice`.

```{code-cell} ipython3
jax.tree_util.tree_map(operator.itemgetter(slice(-5, None)), traj)
```

We can use this approach to visualize the final state:
```{code-cell} ipython3
final_state = jax.tree_util.tree_map(operator.itemgetter(-1), traj)
final_q = final_state.state.model_state.q

for layer in range(2):
    # final_q is now a plain JAX array, we can slice it directly
    data = final_q[layer]
    vmax = jnp.abs(data).max()
    plt.subplot(1, 2, layer + 1)
    plt.title(f"Layer {layer}")
    plt.imshow(data, cmap="bwr", vmin=-vmax, vmax=vmax)
```
