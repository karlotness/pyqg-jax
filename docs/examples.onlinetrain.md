---
file_format: mystnb
kernelspec:
  name: python3
---

# Online Training

Other examples have illustrated some of the capabilities that are
available with the implemented in JAX. One additional JAX
transformation that has not yet been used is {func}`jax.grad`. Because
the base models are implemented in JAX, we can take gradients through
multiple simulation time steps, training a parameterization *online*
through a live simulation, as opposed to training on static snapshots.

Some [existing work](https://doi.org/10.1029/2022MS003124) has
explored the impact of this training approach with QG models
implemented in PyTorch. Here we provide a sketch of an online training
setup using [Equinox](https://github.com/patrick-kidger/equinox) for
neural networks and [Optax](https://github.com/deepmind/optax) for
optimizers.

```{code-cell} ipython3
:tags: [remove-cell]
# Note: docs builds only have CPUs
# This suppresses the JAX warning about missing GPU
# If you're running this with a GPU, delete this cell
%env JAX_PLATFORM_NAME=cpu
```

```{code-cell} ipython3
%env JAX_ENABLE_X64=True
import functools
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
import matplotlib.pyplot as plt
import pyqg_jax
```

To carry out our training we will make use of several elements
demonstrated in other examples. In particular:

- `Operator1` from "{doc}`examples.coarsen`"
- `NNParam` and `module_to_single` from "{doc}`examples.implparam`"

```{code-cell} ipython3
:tags: [remove-cell]

import inspect
import abc
import operator

def model_to_args(model):
    return {
        arg: getattr(model, arg) for arg in inspect.signature(type(model)).parameters
    }


def coarsen_model(big_model, small_nx):
    if big_model.nx != big_model.ny:
        raise ValueError("coarsening tested only for square shapes")
    if small_nx >= big_model.nx:
        raise ValueError(
            f"coarsening output is not strictly smaller (got {big_model.nx} to {small_nx})"
        )
    if small_nx % 2 != 0:
        raise ValueError(f"coarsening output should be even-valued, requested {small_nx}")
    model_args = model_to_args(big_model)
    model_args["nx"] = small_nx
    model_args["ny"] = small_nx
    return type(big_model)(**model_args)


class SpectralCoarsener(abc.ABC):
    def __init__(self, big_model, small_nx):
        self.big_model = big_model
        self.small_nx = small_nx

    @property
    def small_model(self):
        return coarsen_model(self.big_model, self.small_nx)

    @property
    def ratio(self):
        return self.big_model.nx / self.small_nx

    def coarsen_state(self, state):
        if (
            jax.eval_shape(operator.attrgetter("q"), state).shape
            != (self.big_model.nz, self.big_model.ny, self.big_model.nx)
        ):
            raise ValueError(f"incorrect input size {state.qh.shape}")
        out_state = self.small_model.create_initial_state(
            jax.random.key(0)
        )
        nk = out_state.qh.shape[-2] // 2
        trunc = jnp.concatenate(
            [
                state.qh[:, :nk, :nk + 1],
                state.qh[:, -nk:, :nk + 1],
            ],
            axis=-2,
        )
        filtered = trunc * self.spectral_filter / self.ratio**2
        return out_state.update(qh=filtered)

    def compute_q_total_forcing(self, state):
        coarsened_deriv = self.coarsen_state(self.big_model.get_updates(state))
        small_deriv = self.small_model.get_updates(self.coarsen_state(state))
        return coarsened_deriv.q - small_deriv.q

    @property
    @abc.abstractmethod
    def spectral_filter(self):
        pass

    def tree_flatten(self):
        return [self.big_model], self.small_nx

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(big_model=children[0], small_nx=aux_data)


@jax.tree_util.register_pytree_node_class
class Operator1(SpectralCoarsener):
    @property
    def spectral_filter(self):
        return self.small_model.filtr


def param_to_single(param):
    if eqx.is_inexact_array(param):
        if param.dtype == jnp.dtype(jnp.float64):
            return param.astype(jnp.float32)
        elif param.dtype == jnp.dtype(jnp.complex128):
            return param.astype(jnp.complex64)
    return param


def module_to_single(module):
    return jax.tree_util.tree_map(param_to_single, module)


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
```

In addition to the high-resolution stepped model (size 128), the
network, and the coarsening operator we also configure an Adam
optimizer to train our network. For more information on combining
these optimizers with Equinox, consult the [Equinox
documentation](https://docs.kidger.site/equinox/) and the [Optax
documentation](https://optax.readthedocs.io/en/latest/). Note that
Optax provides a separate object, here `optim_state`, representing the
state of the optimizer that must be updated as a part of training.

```{code-cell} ipython3
DT = 3600.0
LEARNING_RATE = 1e-4

big_model = pyqg_jax.steppers.SteppedModel(
    model=pyqg_jax.qg_model.QGModel(
        nx=128,
        ny=128,
        precision=pyqg_jax.state.Precision.SINGLE,
    ),
    stepper=pyqg_jax.steppers.AB3Stepper(dt=DT),
)

coarse_op = Operator1(big_model.model, 32)

# Ensure that all module weights are float32
net = module_to_single(NNParam(key=jax.random.key(123)))

optim = optax.adam(LEARNING_RATE)
optim_state = optim.init(eqx.filter(net, eqx.is_array))
```

With our network and optimizer initialized we generate several sample
states to represent training data. These states are generated at the
high resolution of size 128, and coarsened to the low resolution of
size 32. These small states form our training targets, `target_q`. In
a real application, these reference trajectories would likely be
pre-computed and loaded from disk.

Note that we do not generate any explicit forcing targets here since
we will be supervising on the states directly.

```{code-cell} ipython3
@functools.partial(jax.jit, static_argnames=["num_steps"])
def generate_train_data(seed, num_steps):

    def step(carry, _x):
        next_state = big_model.step_model(carry)
        small_state = coarse_op.coarsen_state(carry.state)
        return next_state, small_state.q

    _final_big_state, target_q = jax.lax.scan(
        step, big_model.create_initial_state(jax.random.key(seed)), None, length=num_steps
    )
    return target_q

target_q = generate_train_data(123, num_steps=100)
```

Next we provide a function to roll out a trajectory starting from some
initial state. In this case we provide the state as a bare {class}`JAX
array <jax.Array>` and have to package it into a model state. Another
example of this process is included in "{doc}`examples.basicstep`."
See "{doc}`examples.implparam`" for another example of using a neural
network parameterization.

```{code-cell} ipython3
def roll_out_with_net(init_q, net, num_steps):

    @pyqg_jax.parameterizations.q_parameterization
    def net_parameterization(state, param_aux, model):
        assert param_aux is None
        q = state.q
        # Scale states to improve stability
        # This 1e-6 is for illustration only
        q_in = (q / 1e-6).astype(jnp.float32)
        q_param = net(q.astype(jnp.float32))
        return 1e-6 * q_param.astype(q.dtype), None

    # Extrace the small model from the coarsener
    # Then wrap it in the network parameterization and stepper
    # Make sure to match time steps
    small_model = pyqg_jax.steppers.SteppedModel(
        model=pyqg_jax.parameterizations.ParameterizedModel(
            model=coarse_op.small_model,
            param_func=net_parameterization,
        ),
        stepper=pyqg_jax.steppers.AB3Stepper(dt=DT),
    )
    # Package our state
    # First, package it for the base model
    base_state = small_model.model.model.create_initial_state(
        jax.random.key(0)
    ).update(q=init_q)
    # Next, wrap it for the parameterization and stepper
    init_state = small_model.initialize_stepper_state(
        small_model.model.initialize_param_state(base_state)
    )

    def step(carry, _x):
        next_state = small_model.step_model(carry)
        # NOTE: Be careful! We output the *old* state for the trajectory
        # Otherwise the initial step would be skipped
        return next_state, carry.state.model_state.q

    # Roll out the state
    _final_step, traj = jax.lax.scan(
        step, init_state, None, length=num_steps
    )
    return traj
```

We provide a function using the above to roll out a trajectory at the
low resolution and compute errors against the reference trajectory
`target_q`. In this case we use a simple MSE loss for training. We
also use Equinox's "filtered" transforms (`filter_jit`,
`filter_value_and_grad`) since these interact more naturally with the
Equinox neural network modules.

```{note}
Online training with long rollouts may lead to out-of-memory errors.
One solution is to use {func}`jax.checkpoint` inside the {func}`scan
<jax.lax.scan>` to save memory through recomputation.

An implementation of this is available in
{func}`powerpax.checkpoint_chunked_scan`, or see this
[sample code](https://github.com/google/jax/issues/2139#issuecomment-1189382794)
for a starting point.
```

```{code-cell} ipython3
def compute_traj_errors(target_q, net):
    rolled_out = roll_out_with_net(
        init_q=target_q[0],
        net=net,
        num_steps=target_q.shape[0],
    )
    err = rolled_out - target_q
    return err

@eqx.filter_jit
def train_batch(batch, net, optim_state):

    def loss_fn(net, batch):
        err = jax.vmap(functools.partial(compute_traj_errors, net=net))(batch)
        mse = jnp.mean(err**2)
        return mse

    # Compute loss value and gradients
    loss, grads = eqx.filter_value_and_grad(loss_fn)(net, batch)
    # Update the network weights
    updates, new_optim_state = optim.update(grads, optim_state, net)
    new_net = eqx.apply_updates(net, updates)
    # Return the loss, updated net, updated optimizer state
    return loss, new_net, new_optim_state
```

We use the components we have to run a short training loop and report
the loss after each step. The training steps are all JIT compiled.

```{code-cell} ipython3
BATCH_SIZE = 8
BATCH_STEPS = 10

np_rng = np.random.default_rng(seed=456)
losses = []
for batch_i in range(25):
    # Rudimentary shuffling in lieu of real data loader
    batch = np.stack(
        [
            target_q[start:start+BATCH_STEPS]
            for start in np_rng.integers(
                0, target_q.shape[0] - BATCH_STEPS, size=BATCH_SIZE
            )
        ]
    )
    loss, net, optim_state = train_batch(batch, net, optim_state)
    losses.append(loss)
    print(f"Step {batch_i + 1:02}: loss={loss.item():.6f}")
```

```{code-cell} ipython3
plt.plot(np.arange(len(losses)) + 1, losses)
plt.xlabel("Step")
plt.ylabel("Step Loss")
plt.grid(True)
```
