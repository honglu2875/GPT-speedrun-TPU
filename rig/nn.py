"""Model building blocks that do not themselves describe a model.

A normalization, a rotary embedding, an affine projection, a normal
initializer, and two ways of walking a parameter tree. Every recipe needs
them and none of them encode an architecture, so a fork should inherit the
exact arithmetic rather than copy it and drift.

What deliberately stays in each recipe: the block wiring, the loss, and the
initialization *scales*, because those are the parameterization under study.
"""

from __future__ import annotations

from typing import Any, Mapping
import jax
import jax.numpy as jnp
import numpy as np


def normal(
    rng: np.random.Generator, shape: tuple[int, ...], scale: float
) -> np.ndarray:
    return rng.standard_normal(shape, dtype=np.float32) * np.float32(scale)


def rms_norm(x: jax.Array, scale: jax.Array, dtype: Any) -> jax.Array:
    """Apply pre-normalization without mean centering in FP32."""

    x32 = x.astype(jnp.float32)
    normalized = x32 * jax.lax.rsqrt(
        jnp.mean(jnp.square(x32), axis=-1, keepdims=True) + 1.0e-5
    )
    return normalized.astype(dtype) * scale.astype(dtype)


def apply_rotary(x: jax.Array) -> jax.Array:
    """Apply interleaved base-10,000 rotary positions to one BTHD tensor."""

    length = x.shape[1]
    head_dim = x.shape[-1]
    if head_dim % 2:
        raise ValueError("rotary attention requires an even head dimension")
    fraction = jnp.arange(0, head_dim, 2, dtype=jnp.float32) / float(head_dim)
    inverse_frequency = jnp.power(10000.0, -fraction)
    angle = jnp.arange(length, dtype=jnp.float32)[:, None] * inverse_frequency[None, :]
    cosine = jnp.cos(angle)[None, :, None, :].astype(x.dtype)
    sine = jnp.sin(angle)[None, :, None, :].astype(x.dtype)
    even = x[..., 0::2]
    odd = x[..., 1::2]
    return jnp.stack(
        (even * cosine - odd * sine, even * sine + odd * cosine), axis=-1
    ).reshape(x.shape)


def linear(x: jax.Array, weight: jax.Array, bias: jax.Array, dtype: Any) -> jax.Array:
    return jnp.einsum("...d,df->...f", x, weight.astype(dtype)) + bias.astype(dtype)


def parameter_count(params: Any) -> int:
    return sum(int(value.size) for value in jax.tree_util.tree_leaves(params))


def flatten_arrays(tree: Any, prefix: str = "params") -> dict[str, np.ndarray]:
    flat: dict[str, np.ndarray] = {}

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key in sorted(value):
                visit(value[key], f"{path}/{key}")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f"{path}/{index}")
        else:
            flat[path] = np.asarray(value)

    visit(tree, prefix)
    return flat
