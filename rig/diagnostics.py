"""Shared sparse statistics for GPT-shaped parameter trees.

Recipes own when diagnostics run and which update they observe. This module
only defines the stable scope partition and the numerical reductions used by
the run-log protocol.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp

from rig.metrics import DIAGNOSTIC_FAMILIES


def diagnostic_scopes(
    tree: Mapping[str, Any],
) -> tuple[tuple[str, int | None, tuple[Any, ...]], ...]:
    """Group a GPT parameter-shaped tree into stable logical report scopes."""

    embeddings = tuple(jax.tree_util.tree_leaves(tree["token_embedding"]))
    blocks = tuple(
        (
            "block",
            layer,
            tuple(jax.tree_util.tree_leaves(block)),
        )
        for layer, block in enumerate(tree["blocks"])
    )
    final_norm = tuple(jax.tree_util.tree_leaves(tree["final_ln_scale"]))
    output = (
        ("overall", None, tuple(jax.tree_util.tree_leaves(tree))),
        ("embeddings", None, embeddings),
        *blocks,
        ("final_norm", None, final_norm),
    )
    if "output_embedding" in tree:
        output = (
            output[0],
            output[1],
            (
                "unembedding",
                None,
                tuple(jax.tree_util.tree_leaves(tree["output_embedding"])),
            ),
            *output[2:],
        )
    return output


def diagnostic_scope_metadata(
    params: Mapping[str, Any],
) -> tuple[tuple[str, int | None, int], ...]:
    """Return scope labels and exact element counts without device work."""

    return tuple(
        (scope, layer, sum(int(value.size) for value in leaves))
        for scope, layer, leaves in diagnostic_scopes(params)
    )


def _diagnostic_stat_vector(values: Sequence[jax.Array]) -> jax.Array:
    """Return norms and stable two-pass centered moments for several arrays."""

    values32 = tuple(value.astype(jnp.float32) for value in values)
    count = sum(int(value.size) for value in values32)
    if count <= 0:  # pragma: no cover - model scopes are statically nonempty
        raise ValueError("diagnostic scope cannot be empty")
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    total = sum((jnp.sum(value) for value in values32), zero)
    mean = total / float(count)

    # Complete the mean before the centered reduction instead of deriving
    # higher moments from cancellation-prone raw power sums.
    l1_sum = sum((jnp.sum(jnp.abs(value)) for value in values32), zero)
    square_sum = sum((jnp.sum(jnp.square(value)) for value in values32), zero)
    variance_sum = sum((jnp.sum(jnp.square(value - mean)) for value in values32), zero)
    third_sum = sum((jnp.sum(jnp.power(value - mean, 3)) for value in values32), zero)
    fourth_sum = sum((jnp.sum(jnp.power(value - mean, 4)) for value in values32), zero)
    return jnp.stack(
        (
            l1_sum,
            jnp.sqrt(jnp.maximum(square_sum, zero)),
            mean,
            jnp.sqrt(jnp.maximum(variance_sum / float(count), zero)),
            third_sum / float(count),
            fourth_sum / float(count),
        )
    ).astype(jnp.float32)


def diagnostic_values(
    params_before: Mapping[str, Any],
    raw_gradients: Mapping[str, Any],
    params_after: Mapping[str, Any],
) -> jax.Array:
    """Return the protocol's ``[scope, family, statistic]`` diagnostic grid.

    ``param`` observes the parameter after this step, so the final point exactly
    matches the checkpoint. ``grad`` is the raw gradient before global clipping.
    ``update`` is the signed actual delta ``params_after - params_before``,
    including clipping, optimizer behavior, and decay.
    """

    updates = jax.tree_util.tree_map(
        lambda after, before: after - before, params_after, params_before
    )
    family_scopes = tuple(
        diagnostic_scopes(tree) for tree in (params_after, raw_gradients, updates)
    )
    scope_count = len(family_scopes[0])
    return jnp.stack(
        tuple(
            jnp.stack(
                tuple(
                    _diagnostic_stat_vector(family_scopes[family][scope][2])
                    for family in range(len(DIAGNOSTIC_FAMILIES))
                )
            )
            for scope in range(scope_count)
        )
    )


__all__ = (
    "diagnostic_scope_metadata",
    "diagnostic_scopes",
    "diagnostic_values",
)
