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
        self.Hi = jnp.array([H1, H1/delta], dtype=DTYPE_REAL)
        self.U1 = U1
        self.U2 = U2
        self.H1 = H1

        # initialize background, inversion matrix, forcing

        # INITIALIZE BACKGROUND
        self.H = self.Hi.sum()
        self.Ubg = jnp.array([self.U1, self.U2], dtype=DTYPE_REAL)
        self.U = self.U1 - self.U2
        # The F parameters
        self.F1 = self.rd**-2 / (1 + self.delta)
        self.F2 = self.delta * self.F1
        # The meridional PV gradients in each layer
        self.Qy1 = self.beta + self.F1 * (self.U1 - self.U2)
        self.Qy2 = self.beta - self.F2 * (self.U1 - self.U2)
        self.Qy = jnp.array([self.Qy1, self.Qy2], dtype=DTYPE_REAL)
        self._ikQy = 1j * (jnp.expand_dims(self.kk, 0) * jnp.expand_dims(self.Qy, -1))
        # complex versions, multiplied by k, speeds up computations to precompute
        self.ikQy1 = self.Qy1 * 1j * self.k
        self.ikQy2 = self.Qy2 * 1j * self.k
        # vector version
        self.ikQy = jnp.vstack([jnp.expand_dims(self.ikQy1, axis=0), jnp.expand_dims(self.ikQy2, axis=0)])
        self.ilQx = 0
        #layer spacing
        self.del1 = self.delta / (self.delta + 1)
        self.del2 = (self.delta + 1) ** -1

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

    def param_json(self):
        super_params = json.loads(super().param_json())
        del super_params["nz"]
        super_params.update(
            {
                "beta": self.beta,
                "rd": self.rd,
                "delta": self.delta,
                "H1": self.H1,
                "U1": self.U1,
                "U2": self.U2,
            }
        )
        return json.dumps(super_params)

    @classmethod
    def from_param_json(cls, param_str):
        return cls(**json.loads(param_str))
