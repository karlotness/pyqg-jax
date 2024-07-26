---
file_format: mystnb
kernelspec:
  name: python3
---

# Surface Quasi-Geostrophic (SQG) Vortex

This example is reworked from the original {doc}`PyQG SQG
example<pyqg:examples/sqg>`. We reproduce a plot from the paper
"[Surface quasi-geostrophic
dynamics](https://doi.org/10.1017/S0022112095000012)".

```{code-cell} ipython3
:tags: [remove-cell]
# Note: docs builds only have CPUs
# This suppresses the JAX warning about missing GPU
# If you're running this with a GPU, delete this cell
%env JAX_PLATFORMS=cpu
```

```{code-cell} ipython3
import operator
import functools
import math
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import powerpax
import pyqg_jax
```

## Construct the Model

We construct the {class}`~pyqg_jax.sqg_model.SQGModel` object with
adjusted parameters, and wrap it in a {class}`stepper
<pyqg_jax.steppers.SteppedModel>` for later simulation.

```{code-cell} ipython3
DT = 0.005
T_MAX = 8
SNAP_INTERVAL = 2

stepped_model = pyqg_jax.steppers.SteppedModel(
    pyqg_jax.sqg_model.SQGModel(
        L=2 * jnp.pi,
        nx=512,
        beta=0,
        Nb=1,
        H=1,
        f_0=1,
    ),
    pyqg_jax.steppers.AB3Stepper(dt=DT),
)

stepped_model
```

## Configure Initial Condition

The initial condition in this example is an elliptical vortex

$$ -\alpha\exp\bigg(\!-\frac{x^2 + (4y)^2}{(L / 6)^2}\bigg) $$

The amplitude is $\alpha = 1$ which sets the strength and speed of the
vortex. The aspect ratio in this example is 4 and gives rise to an
instability.

We calculate the values for this initial condition

```{code-cell} ipython3
x = stepped_model.model.x - jnp.pi
y = stepped_model.model.y - jnp.pi
vortex = -jnp.exp(-(x**2 + (4 * y) ** 2)/( stepped_model.model.L / 6) ** 2)
```

We can examine the initial state

```{code-cell} ipython3
plt.imshow(
    vortex,
    cmap="RdBu",
    vmin=-1,
    vmax=0,
    extent=(0, stepped_model.model.W, 0, stepped_model.model.L),
)
plt.colorbar()
```

and finally package it into a stepped state object

```{code-cell} ipython3
init_state = stepped_model.create_initial_state(jax.random.key(0)).update(
    state=stepped_model.model.create_initial_state(jax.random.key(0)).update(
        q=jnp.expand_dims(vortex, 0)
    ),
)

init_state
```

## Run the Model

We roll out the initial state up to `T_MAX` taking snapshots according
to `SNAP_INTERVAL`.

```{code-cell} ipython3
@functools.partial(jax.jit, static_argnames=["num_steps", "subsample"])
def roll_out_state(state, num_steps, subsample):
    def loop_fn(carry, _x):
        current_state = carry
        next_state = stepped_model.step_model(current_state)
        return next_state, current_state

    _final_carry, traj_steps = powerpax.sliced_scan(
        loop_fn, state, None, length=num_steps, step=subsample,
    )
    return traj_steps
```

Note the use of {func}`powerpax.sliced_scan` above to skip steps
between each snapshot. This produces a trajectory `traj` that we can
examine.

```{code-cell} ipython3
num_steps = math.ceil(T_MAX / DT) + 1
snap_subsample = math.ceil(SNAP_INTERVAL / DT)

traj = roll_out_state(init_state, num_steps, snap_subsample)
```

## Plot States

We plot each of the snapshots taken from the simulation. With access
to more time (or ideally a GPU) this model can be simulated for a
longer time period by adjusting `T_MAX` above.

```{code-cell} ipython3
cols = 3
rows = math.ceil(traj.tc.shape[0] / 3)
fig, axs = plt.subplots(
    rows,
    cols,
    layout="constrained",
    figsize=(6, 2.25 * rows),
    sharex=True,
    sharey=True,
)

for step_i, ax in enumerate(axs.ravel()):
    if step_i >= traj.tc.shape[0]:
        fig.delaxes(ax)
        continue
    step = jax.tree.map(operator.itemgetter(step_i), traj)
    data = step.state.q[0]
    ax.imshow(
        data,
        vmin=-1,
        vmax=0,
        cmap="RdBu",
        extent=(0, stepped_model.model.W, 0, stepped_model.model.L),
    )
    ax.set_title(f"Time = {step.t.item():.0f}")
```
