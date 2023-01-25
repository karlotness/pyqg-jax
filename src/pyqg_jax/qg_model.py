# Copyright Karl Otness
# SPDX-License-Identifier: MIT


import dataclasses
import math
import json
import jax
import jax.numpy as jnp
import jax.random
from . import model
from .kernel import DTYPE_COMPLEX, DTYPE_REAL


@jax.tree_util.register_pytree_node_class
class QGModel(model.Model):
    def __init__(
            self,
            beta=1.5e-11,
            rd=15000.0,
            delta=0.25,
            H1=500,
            U1=0.025,
            U2=0.0,
            **kwargs,
    ):
        super().__init__(nz=2, **kwargs)
        self.beta = beta
        self.rd = rd
        self.delta = delta
        self.U1 = U1
        self.U2 = U2
        self.H1 = H1

    @property
    def U(self):
        return self.U1 - self.U2

    @property
    def Hi(self):
        return jnp.array([self.H1, self.H1/self.delta], dtype=DTYPE_REAL)

    @property
    def H(self):
        return self.Hi.sum()

    @property
    def Ubg(self):
        return jnp.array([self.U1, self.U2], dtype=DTYPE_REAL)

    @property
    def F1(self):
        return self.rd**-2 / (1 + self.delta)

    @property
    def F2(self):
        return self.delta * self.F1

    @property
    def Qy1(self):
        return self.beta + self.F1 * (self.U1 - self.U2)

    @property
    def Qy2(self):
        return self.beta - self.F2 * (self.U1 - self.U2)

    @property
    def Qy(self):
        return jnp.array([self.Qy1, self.Qy2], dtype=DTYPE_REAL)

    @property
    def _ikQy(self):
        return 1j * (jnp.expand_dims(self.kk, 0) * jnp.expand_dims(self.Qy, -1))

    @property
    def ikQy1(self):
        return self.Qy1 * 1j * self.k

    @property
    def ikQy2(self):
        return self.Qy2 * 1j * self.k

    @property
    def ikQy(self):
        return jnp.vstack([jnp.expand_dims(self.ikQy1, axis=0), jnp.expand_dims(self.ikQy2, axis=0)])

    @property
    def ilQx(self):
        return 0

    @property
    def del1(self):
        return self.delta / (self.delta + 1)

    @property
    def del2(self):
        return (self.delta + 1) ** -1

        # INITIALIZE FORCING (nothing to do)

    def _set_q1q2(self, state, q1, q2):
        return self.set_q(state, jnp.vstack([jnp.expand_dims(q1, axis=0), jnp.expand_dims(q2, axis=0)]))

    def create_initial_state(self, rng):
        state = super().create_initial_state()
        # initial conditions (pv anomalies)
        rng_a, rng_b = jax.random.split(rng, num=2)
        q1 = 1e-7 * jax.random.uniform(rng_a, shape=(self.ny, self.nx), dtype=DTYPE_REAL) + 1e-6 * (jnp.ones((self.ny, 1), dtype=DTYPE_REAL) * jax.random.uniform(rng_b, shape=(1, self.nx), dtype=DTYPE_REAL))
        q2 = jnp.zeros_like(self.x, dtype=DTYPE_REAL)
        state = self._set_q1q2(state, q1, q2)
        return state

    def _apply_a_ph(self, state):
        qh = jnp.moveaxis(state.qh, 0, -1)
        qh_orig_shape = qh.shape
        qh = qh.reshape((-1, 2))
        # Compute inversion matrix
        dk = 2 * math.pi / self.L
        dl = 2 * math.pi / self.W
        ll = dl * jnp.concatenate(
            [
                jnp.arange(0, self.nx / 2, dtype=jnp.float64),
                jnp.arange(-self.nx / 2, 0, dtype=jnp.float64),
            ]
        )
        kk = dk * jnp.arange(0, self.nk, dtype=jnp.float64)
        k, l = jnp.meshgrid(kk, ll)
        wv2 = k**2 + l**2
        inv_mat2 = jnp.moveaxis(
            jnp.array(
                [
                    [
                        # 0, 0
                        -(wv2 + self.F1),
                        # 0, 1
                        jnp.full_like(wv2, self.F1),
                    ],
                    [
                        # 1, 0
                        jnp.full_like(wv2, self.F2),
                        # 1, 1
                        -(wv2 + self.F2),
                    ],
                ]
            ),
            (0, 1),
            (-2, -1)
        ).reshape((-1, 2, 2))[1:]
        # Solve the system for the tail
        ph_tail = jnp.linalg.solve(inv_mat2, qh[1:].astype(jnp.complex128)).astype(DTYPE_COMPLEX)
        # Fill zeros for the head
        ph_head = jnp.expand_dims(jnp.zeros_like(qh[0]), 0)
        # Combine and return
        ph = jnp.concatenate([ph_head, ph_tail], axis=0).reshape(qh_orig_shape)
        return jnp.moveaxis(ph, -1, 0)

    def tree_flatten(self):
        attributes = ["beta", "rd", "delta", "H1", "U1", "U2"]
        children = [getattr(self, attr) for attr in attributes]
        super_children, super_aux = super().tree_flatten()
        for key, val in zip(super_aux, super_children, strict=True):
            if key == "nz":
                # Need to remove parameter nz since QGModel sets it internally
                continue
            attributes.append(key)
            children.append(val)
        return children, tuple(attributes)
