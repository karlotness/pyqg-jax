# Copyright 2024 Karl Otness
# SPDX-License-Identifier: MIT


import dataclasses
import jax
import jax.numpy as jnp
from . import state, _utils


@dataclasses.dataclass
class KrElems:
    dkr: jax.Array
    keep_num: int
    edge_kr: jax.Array


def get_kr_elems(grid: state.Grid, *, truncate: bool = True) -> KrElems:
    for name in ["L", "W", "dk", "dl"]:
        # Each of these attributes should be a 0-dim scalar
        if (dims := jax.eval_shape(_utils.AttrGetter(name), grid)).ndim != 0:
            raise ValueError(
                f"grid.{name} has {dims} dimensions, but should have 0 (use jax.vmap)"
            )
    max_buckets = max(grid.nx // 2, grid.ny // 2)
    ll_max = (2 * jnp.pi / grid.L) * (grid.nx // 2)
    kk_max = (2 * jnp.pi / grid.W) * (grid.ny // 2)
    kmax = jax.lax.cond(
        truncate,
        lambda lm, km: jnp.minimum(lm, km),
        lambda lm, km: jnp.sqrt(lm**2 + km**2),
        ll_max,
        kk_max,
    )
    dkr = jnp.sqrt(grid.dk**2 + grid.dl**2)
    kr = jnp.arange(max_buckets) * dkr
    keep_num = jnp.clip(jnp.ceil(kmax / dkr), 0, max_buckets).astype(jnp.uint32)
    return KrElems(
        dkr=dkr,
        keep_num=keep_num,
        edge_kr=kr,
    )


def get_plot_kr(grid: state.Grid, *, truncate: bool = True) -> tuple[jax.Array, int]:
    kr_elems = get_kr_elems(grid, truncate=truncate)
    kr = kr_elems.edge_kr + kr_elems.dkr / 2
    return kr, kr_elems.keep_num


def calc_ispec(
    var: jax.Array, grid: state.Grid, *, averaging: bool = True, truncate: bool = True
) -> jax.Array:
    kr_elems = get_kr_elems(grid, truncate=truncate)
    var = jnp.concatenate(
        [
            jnp.expand_dims(var[..., 0] / 2, -1),
            var[..., 1:-1],
            jnp.expand_dims(var[..., -1] / 2, -1),
        ],
        axis=-1,
    )
    buckets = jnp.floor(grid.get_kappa() / kr_elems.dkr).astype(jnp.uint32)
    sums = jax.vmap(
        lambda v: jax.ops.segment_sum(
            v.ravel(),
            buckets.ravel(),
            num_segments=kr_elems.edge_kr.size,
            indices_are_sorted=False,
            unique_indices=False,
        )
    )(var)

    def _avg(val):
        bucket_sizes = jax.ops.segment_sum(
            jnp.ones_like(buckets.ravel()),
            buckets.ravel(),
            num_segments=kr_elems.edge_kr.size,
            indices_are_sorted=False,
            unique_indices=False,
        )
        val = val / jnp.maximum(bucket_sizes, 1)
        return (
            val * (kr_elems.edge_kr + kr_elems.dkr / 2) * jnp.pi / (grid.dk * grid.dl)
        )

    val = (
        jax.lax.cond(
            averaging,
            jax.vmap(_avg),
            jax.vmap(lambda v: v / kr_elems.dkr),
            sums,
        )
        * 2
    )
    return val
