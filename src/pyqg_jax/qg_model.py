# Copyright Karl Otness
# SPDX-License-Identifier: MIT


__all__ = ["QGModel"]


import math
import jax
import jax.numpy as jnp
from . import _model, _utils


@_utils.register_pytree_node_class_private
class QGModel(_model.Model):
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

    def create_initial_state(self, key):
        state = super().create_initial_state()
        # initial conditions (pv anomalies)
        rng_a, rng_b = jax.random.split(key, num=2)
        q1 = 1e-7 * jax.random.uniform(
            rng_a, shape=(self.ny, self.nx), dtype=self._dtype_real
        ) + 1e-6 * (
            jnp.ones((self.ny, 1), dtype=self._dtype_real)
            * jax.random.uniform(rng_b, shape=(1, self.nx), dtype=self._dtype_real)
        )
        q2 = jnp.zeros_like(self.x, dtype=self._dtype_real)
        state = state.update(
            q=jnp.vstack([jnp.expand_dims(q1, axis=0), jnp.expand_dims(q2, axis=0)])
        )
        return state

    @property
    def U(self):
        return self.U1 - self.U2

    @property
    def Hi(self):
        return jnp.array([self.H1, self.H1 / self.delta], dtype=self._dtype_real)

    @property
    def H(self):
        return self.Hi.sum()

    @property
    def Ubg(self):
        return jnp.array([self.U1, self.U2], dtype=self._dtype_real)

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
        return jnp.array([self.Qy1, self.Qy2], dtype=self._dtype_real)

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
        return jnp.vstack(
            [jnp.expand_dims(self.ikQy1, axis=0), jnp.expand_dims(self.ikQy2, axis=0)]
        )

    @property
    def ilQx(self):
        return 0

    @property
    def del1(self):
        return self.delta / (self.delta + 1)

    @property
    def del2(self):
        return (self.delta + 1) ** -1

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
            (-2, -1),
        ).reshape((-1, 2, 2))[1:]
        # Solve the system for the tail
        ph_tail = jnp.linalg.solve(inv_mat2, qh[1:].astype(jnp.complex128)).astype(
            self._dtype_complex
        )
        # Fill zeros for the head
        ph_head = jnp.expand_dims(jnp.zeros_like(qh[0]), 0)
        # Combine and return
        ph = jnp.concatenate([ph_head, ph_tail], axis=0).reshape(qh_orig_shape)
        return jnp.moveaxis(ph, -1, 0)

    def _tree_flatten(self):
        super_children, (
            super_attrs,
            super_static_vals,
            super_static_attrs,
        ) = super()._tree_flatten()
        new_attrs = ("beta", "rd", "delta", "U1", "U2", "H1")
        new_children = [getattr(self, name) for name in new_attrs]
        children = [*super_children, *new_children]
        new_attrs = (*super_attrs, *new_attrs)
        return children, (new_attrs, super_static_vals, super_static_attrs)

    def __repr__(self):
        nx_summary = _utils.indent_repr(_utils.summarize_object(self.nx), 2)
        ny_summary = _utils.indent_repr(_utils.summarize_object(self.ny), 2)
        L_summary = _utils.indent_repr(_utils.summarize_object(self.L), 2)
        W_summary = _utils.indent_repr(_utils.summarize_object(self.W), 2)
        rek_summary = _utils.indent_repr(_utils.summarize_object(self.rek), 2)
        filterfac_summary = _utils.indent_repr(
            _utils.summarize_object(self.filterfac), 2
        )
        f_summary = _utils.indent_repr(_utils.summarize_object(self.f), 2)
        g_summary = _utils.indent_repr(_utils.summarize_object(self.g), 2)
        beta_summary = _utils.indent_repr(_utils.summarize_object(self.beta), 2)
        rd_summary = _utils.indent_repr(_utils.summarize_object(self.rd), 2)
        delta_summary = _utils.indent_repr(_utils.summarize_object(self.delta), 2)
        U1_summary = _utils.indent_repr(_utils.summarize_object(self.U1), 2)
        U2_summary = _utils.indent_repr(_utils.summarize_object(self.U2), 2)
        H1_summary = _utils.indent_repr(_utils.summarize_object(self.H1), 2)
        precision_summary = self._precision.name
        return f"""\
QGModel(
  nx={nx_summary},
  ny={ny_summary},
  L={L_summary},
  W={W_summary},
  rek={rek_summary},
  filterfac={filterfac_summary},
  f={f_summary},
  g={g_summary},
  beta={beta_summary},
  rd={rd_summary},
  delta={delta_summary},
  U1={U1_summary},
  U2={U2_summary},
  H1={H1_summary},
  precision={precision_summary},
)"""
