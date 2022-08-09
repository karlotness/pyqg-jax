import dataclasses
import json
import jax
import jax.numpy as jnp
import jax.random
from . import model

class QGModel(model.Model):
    def __init__(
            self,
            beta=1.5e-11,
            rd=15000.0,
            delta=0.25,
            H1 = 500,
            U1=0.025,
            U2=0.0,
            **kwargs,
    ):
        super().__init__(nz=2, **kwargs)
        self.beta = beta
        self.rd = rd
        self.delta = delta
        self.Hi = jnp.array([H1, H1/delta])
        self.U1 = U1
        self.U2 = U2
        self.H1 = H1

        # initialize background, inversion matrix, forcing

        # INITIALIZE BACKGROUND
        self.H = self.Hi.sum()
        self.Ubg = jnp.array([self.U1, self.U2])
        self.U = self.U1 - self.U2
        # The F parameters
        self.F1 = self.rd**-2 / (1 + self.delta)
        self.F2 = self.delta * self.F1
        # The meridional PV gradients in each layer
        self.Qy1 = self.beta + self.F1 * (self.U1 - self.U2)
        self.Qy2 = self.beta - self.F2 * (self.U1 - self.U2)
        self.Qy = jnp.array([self.Qy1, self.Qy2])
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

        # INITIALIZE INVERSION MATRIX
        self.inv_mat2 = jnp.moveaxis(
            jnp.array(
                [
                    [
                        # 0, 0
                    -(self.wv2 + self.F1),
                        # 0, 1
                    self.F1 * jnp.ones_like(self.wv2),
                    ],
                    [
                        # 1, 0
                    self.F2 * jnp.ones_like(self.wv2),
                        # 1, 1
                    -(self.wv2 + self.F2),
                    ],
                ]
            ),
            (0, 1),
            (-2, -1)
        )

        # INITIALIZE FORCING (nothing to do)

    # calc cfl
    def _calc_cfl(self, state):
        return jnp.abs(
            jnp.hstack([state.u + jnp.expand_dims(self.Ubg, axis=(1, 2)), state.v])
        ).max() * self.dt / self.dx

    # calc ke
    def _calc_ke(self, state):
        ke1 = 0.5 * self.Hi[0] * self.spec_var(self.wv * state.ph[0])
        ke2 = 0.5 * self.Hi[1] * self.spec_var(self.wv * state.ph[1])
        return (ke1.sum() + ke2.sum()) / self.H

    def _calc_eddy_time(self, state):
        ens = 0.5 * self.Hi[0] * self.spec_var(self.wv2 * self.ph1) + 0.5 * self.Hi[1] * self.spec_var(self.wv2 * self.ph2)
        return 2 * jnp.pi * jnp.sqrt(self.H / ens) / 86400

    def _set_q1q2(self, state, q1, q2):
        return self.set_q(state, jnp.vstack([jnp.expand_dims(q1, axis=0), jnp.expand_dims(q2, axis=0)]))

    def create_initial_state(self, rng):
        state = super().create_initial_state()
        # initial conditions (pv anomalies)
        rng_a, rng_b = jax.random.split(rng, num=2)
        q1 = 1e-7 * jax.random.uniform(rng_a, shape=(self.ny, self.nx)) + 1e-6 * (jnp.ones((self.ny, 1)) * jax.random.uniform(rng_b, shape=(1, self.nx)))
        q2 = jnp.zeros_like(self.x)
        state = self._set_q1q2(state, q1, q2)
        return state

    def _apply_a_ph(self, state):
        qh = jnp.moveaxis(state.qh, 0, -1)
        ph = jnp.linalg.solve(self.inv_mat2, qh)
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
