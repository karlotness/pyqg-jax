---
file_format: mystnb
kernelspec:
  name: python3
---

# Implementing a Parameterization

In addition to the built-in parameterizations provided in this
package, it is possible to use your own code to implement a custom
parameterization. For example, a parameterization implemented by a
neural network. Additional details on custom parameterizations are
available in the documentation section "{ref}`sec-ref-impl-param`".

We illustrate the creation of a simple neural network
parameterization. For the neural network itself we use
[Equinox](https://github.com/patrick-kidger/equinox), but other
libraries such as [Flax](https://github.com/google/flax) could be used
as well.

```{code-cell} ipython3
import functools
import jax
import jax.numpy as jnp
import equinox as eqx
import pyqg_jax
```

In our example we will use a small convolutional network. The network
here is randomly initialized, but in real use it would likely use
trained weights loaded from a file. Also, the architecture here has a
padding size configured to keep the state sizes constant. This pads
with zeros, but because system states are periodic it may be desirable
to use periodic padding for real applications.

```{code-cell} ipython3
class NNParam(eqx.Module):
    ops: eqx.nn.Sequential

    def __init__(self, key):
        key1, key2 = jax.random.split(key, 2)
        self.ops = eqx.nn.Sequential(
            [
                eqx.nn.Conv2d(2, 5, kernel_size=3, padding=1, key=key1),
                eqx.nn.Lambda(jax.nn.relu),
                eqx.nn.Conv2d(5, 2, kernel_size=3, padding=1, key=key2),
            ]
        )

    def __call__(self, x, *, key=None):
        return self.ops(x, key=key)

net = NNParam(key=jax.random.PRNGKey(0))

net
```

Next, we write a function wrapping the network so that it is suitable
for use with {class}`ParameterizedModel
<pyqg_jax.parameterizations.ParameterizedModel>`. We illustrate a
parameterization updating `dqdt` and so use {func}`q_parameterization
<pyqg_jax.parameterizations.q_parameterization>` to decorate our
function.

This parameterization largely just evaluates the network. However,
because the network weights are `float32` while the simulation state
is `float64` we add casting around the network as needed.

Finally, note that this parameterization is *stateless* and depends
only on the current model state. The `param_aux` value is always
`None` and we always return `None` as the updated state value.

```{code-cell} ipython3
@pyqg_jax.parameterizations.q_parameterization
def net_parameterization(state, param_aux, model):
    assert param_aux is None
    q = state.q
    q_param = net(q.astype(jnp.float32))
    return q_param.astype(q.dtype), None
```

Next we construct our base {class}`QGModel
<pyqg_jax.qg_model.QGModel>` wrapped in a {class}`ParameterizedModel
<pyqg_jax.parameterizations.ParameterizedModel>`. Because our
parameterization is stateless we can use the default value for
`init_param_aux_func` which initializes the state to `None`.

```{code-cell} ipython3
model = pyqg_jax.steppers.SteppedModel(
    model=pyqg_jax.parameterizations.ParameterizedModel(
        model=pyqg_jax.qg_model.QGModel(
            nx=32,
            ny=32,
            precision=pyqg_jax.state.Precision.DOUBLE,
        ),
        param_func=net_parameterization,
    ),
    stepper=pyqg_jax.steppers.AB3Stepper(dt=3600.0),
)

model
```

As in {doc}`examples.basicstep` we now write a JIT compiled function
to roll out a trajectory from an initial state. Because the states and
parameterized model are all JAX PyTrees they can be passed as
arguments through the function.

```{code-cell} ipython3
@functools.partial(jax.jit, static_argnames=["num_steps"])
def roll_out_state(state, stepped_model, num_steps):

    def loop_fn(carry, _x):
        current_state = carry
        next_state = stepped_model.step_model(current_state)
        return next_state, next_state

    _final_state, traj = jax.lax.scan(
        loop_fn, state, None, length=num_steps
    )
    return traj
```

We initialize our model state. Note the added {class}`NoStepValue
<pyqg_jax.steppers.NoStepValue>` wrapping the `None` state. This
interacts with the {class}`time stepper
<pyqg_jax.steppers.AB3Stepper>` so that the auxiliary state values are
not time-stepped. It is up to your parameterization to provide new
values for these, as needed.

```{code-cell} ipython3
init_state = model.create_initial_state(jax.random.PRNGKey(0))

init_state
```

Finally we roll out the resulting trajectory.

```{code-cell} ipython3
traj = roll_out_state(init_state, model, num_steps=10)

traj
```

Notice how---as in past examples---we have a new leading time
dimension of size 10.

## Stateful Parameterizations

The parameterization above was *stateless* which is to say that its
value depended only on the model state and no on any history of
previous steps. No all have this structure, particularly in JAX and
this package has facilities to integrate *stateful* trajectories into
the time-stepped models.

Here we illustrate an extension of the neural network parameterization
above, managing an additional auxiliary state value. These can be
arbitrary JAX PyTrees which providing flexibility for a variety of use
cases. In particular here we use a nested tuple of several values.

Our auxiliary state will have two components:

1. a {func}`PRNGKey <jax.random.PRNGKey>` to provide randomness for
   use with the parameterization
2. a two-step shift register of past parameterization outputs

These illustrate two different values that one might need. The first
illustrates how to manage random states for use in stochastic
parameterizations, and the second illustrate one possible approach to
implementing parameterizations that depend on a history of previous
states.

The first step required is to provide code to initialize our
`param_aux` values. Here, our function takes on additional argument
`seed` which we use to construct a `PRNGKey`. We also produce two
placeholders for the past model states, in this case arrays with the
proper shapes filled with zeros.

```{code-cell} ipython3
def net_init_aux(model_state, model, seed):
    rng = jax.random.PRNGKey(seed)
    init_state = jnp.zeros_like(model_state.q, dtype=jnp.float32)
    init_states = (init_state, init_state)
    return rng, init_states
```

Next we extend `net_parameterization`, defined above. This new version
makes use of the `param_aux` argument, unpacking it to retrieve
previous states. As before we cast the current state to `float32`
before handing it to the network and we {func}`split
<jax.random.split>` the RNG to provide a separate state to pass to our
network. The second step, we shift the past states, dropping the
oldest and producing `new_states`.

This function returns the parameterization output as its first return
value, and the new `param_aux` value as its second.

```{code-cell} ipython3
@pyqg_jax.parameterizations.q_parameterization
def net_key_parameterization(state, param_aux, model):
    old_rng, (pp_param, p_param) = param_aux
    rng, new_rng = jax.random.split(old_rng)
    orig_dtype = state.q.dtype
    q_param = net(state.q.astype(jnp.float32), key=rng)
    out_param = jnp.mean(jnp.stack([q_param, pp_param, p_param]), axis=0)
    new_states = (p_param, q_param)
    return out_param.astype(orig_dtype), (new_rng, new_states)
```

Now we can use both of these functions as arguments to
`ParameterizedModel`.

```{code-cell} ipython3
state_model = pyqg_jax.steppers.SteppedModel(
    model=pyqg_jax.parameterizations.ParameterizedModel(
        model=pyqg_jax.qg_model.QGModel(
            nx=32,
            ny=32,
            precision=pyqg_jax.state.Precision.DOUBLE,
        ),
        param_func=net_key_parameterization,
        init_param_aux_func=net_init_aux,
    ),
    stepper=pyqg_jax.steppers.AB3Stepper(dt=14400.0),
)
```

We can create a new model state with the parameterization values. Note
that we provide the additional `seed` argument which is passed through
to `net_init_aux`.

```{code-cell} ipython3
init_key_state = state_model.create_initial_state(jax.random.PRNGKey(0), seed=10)

init_key_state
```

Notice that the `param_aux` value wrapped inside the `NoStepValue`
object is no longer `None`. The `PRNGKey`'s `uint32` array is visible,
along with two `float32` arrays for past parameterization outputs.

Finally, as before we can roll out a trajectory:

```{code-cell} ipython3
state_traj = roll_out_state(init_key_state, state_model, num_steps=10)

state_traj
```

Note that the parameterization states also have time dimensions.
However, their values are not time-stepped by the `AB3Stepper`, but
instead the updated values produced by `net_key_parameterization` are
used directly and are left untouched by the time-stepper.
