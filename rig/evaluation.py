"""Evaluation passes and the schedules that decide when they run.

Every recipe scores the same way: a deterministic prefix of the validation
split for the headline number, packed fixed-shape batches for a downstream
domain, and sparse probes on a cadence during training. None of that varies
with the model, so none of it belongs in an entry program.

Both passes take plain shapes rather than a recipe's config, and both are
deliberately outside the timed training region -- ``train_seconds`` measures
training, and an evaluation that crept inside it would flatter the run.
"""

from __future__ import annotations

from typing import Any
import math
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding

from rig.mesh import (
    finite_metric,
    local_batch_size,
    local_device_get,
    put_host_local_array,
    rank_local_slice,
)
from rig.tokens import DownstreamDomain, TokenDataset, downstream_batches


def should_run_validation_probe(step: int, *, every: int, final_step: int) -> bool:
    """Return whether this step gets a non-canonical fixed-prefix probe."""

    return (
        every > 0
        and step < final_step
        and step % every == 0
    )


def should_run_diagnostics(step: int, *, every: int, final_step: int) -> bool:
    """Capture the first/final updates plus the configured sparse cadence."""

    return every > 0 and (
        step == 1 or step == final_step or step % every == 0
    )


def evaluate_validation_prefix(
    params: Any,
    dataset: TokenDataset,
    compiled_eval: Any,
    data_sharding: NamedSharding,
    *,
    batch_size: int,
    seq_len: int,
    semantic_vocab_size: int,
    batches: int,
    mesh: Mesh | None = None,
    process_index: int = 0,
    process_count: int = 1,
) -> tuple[float, float]:
    """Synchronously evaluate batches ``0..batches-1`` of the fixed prefix."""

    if batches <= 0:
        raise ValueError("validation batch count must be positive")
    started = time.perf_counter()
    loss_sum = 0.0
    scored_tokens = 0
    if process_count > 1 and mesh is None:
        raise ValueError("a global mesh is required for multi-process evaluation")
    local_batch = local_batch_size(batch_size, process_count)
    mask_host = np.ones((local_batch, seq_len), dtype=np.float32)
    if mesh is None:
        mask = jax.device_put(mask_host, data_sharding)
    else:
        mask = put_host_local_array(
            mask_host, mesh, P("data", None), data_sharding, process_count
        )
    for eval_index in range(batches):
        eval_x_host, eval_y_host = dataset.validation_batch(
            eval_index,
            batch_size,
            seq_len,
            semantic_vocab_size,
        )
        eval_x_host = rank_local_slice(eval_x_host, process_index, process_count)
        eval_y_host = rank_local_slice(eval_y_host, process_index, process_count)
        if mesh is None:
            eval_x = jax.device_put(eval_x_host, data_sharding)
            eval_y = jax.device_put(eval_y_host, data_sharding)
        else:
            eval_x = put_host_local_array(
                eval_x_host, mesh, P("data", None), data_sharding, process_count
            )
            eval_y = put_host_local_array(
                eval_y_host, mesh, P("data", None), data_sharding, process_count
            )
        batch_loss_sum, batch_scored = local_device_get(
            compiled_eval(params, eval_x, eval_y, mask)
        )
        loss_sum += float(batch_loss_sum)
        scored_tokens += int(batch_scored)
    elapsed = max(time.perf_counter() - started, 1.0e-12)
    expected_tokens = batches * batch_size * seq_len
    if scored_tokens != expected_tokens:
        raise RuntimeError(
            f"validation executable scored {scored_tokens:,} tokens; expected "
            f"{expected_tokens:,}"
        )
    return (
        finite_metric("validation_loss", loss_sum / scored_tokens),
        finite_metric("validation_seconds", elapsed, positive=True),
    )


def evaluate_downstream_domain(
    params: Any,
    domain: DownstreamDomain,
    compiled_eval: Any,
    data_sharding: NamedSharding,
    *,
    batch_size: int,
    seq_len: int,
    mesh: Mesh | None = None,
    process_index: int = 0,
    process_count: int = 1,
) -> dict[str, float | int]:
    """Evaluate one domain with exact masking and the shared eval executable."""

    started = time.perf_counter()
    loss_sum = 0.0
    scored_tokens = 0
    if process_count > 1 and mesh is None:
        raise ValueError("a global mesh is required for multi-process evaluation")
    for x_host, y_host, mask_host in downstream_batches(
        domain, seq_len=seq_len, batch_size=batch_size
    ):
        x_host = rank_local_slice(x_host, process_index, process_count)
        y_host = rank_local_slice(y_host, process_index, process_count)
        mask_host = rank_local_slice(mask_host, process_index, process_count)
        if mesh is None:
            x = jax.device_put(x_host, data_sharding)
            y = jax.device_put(y_host, data_sharding)
            mask = jax.device_put(mask_host, data_sharding)
        else:
            x = put_host_local_array(
                x_host, mesh, P("data", None), data_sharding, process_count
            )
            y = put_host_local_array(
                y_host, mesh, P("data", None), data_sharding, process_count
            )
            mask = put_host_local_array(
                mask_host, mesh, P("data", None), data_sharding, process_count
            )
        batch_loss_sum, batch_scored = local_device_get(
            compiled_eval(params, x, y, mask)
        )
        loss_sum += float(batch_loss_sum)
        scored_tokens += int(batch_scored)
    elapsed = finite_metric(
        f"downstream {domain.name} seconds",
        max(time.perf_counter() - started, 1.0e-12),
        positive=True,
    )
    if scored_tokens != domain.scored_tokens:
        raise RuntimeError(
            f"downstream {domain.name} scored {scored_tokens:,} tokens; expected "
            f"{domain.scored_tokens:,}"
        )
    loss = finite_metric(f"downstream {domain.name} loss", loss_sum / scored_tokens)
    return {
        "loss": loss,
        "perplexity": perplexity_from_loss(loss),
        "scored_tokens": scored_tokens,
        "seconds": elapsed,
    }


def perplexity_from_loss(loss: float) -> float:
    try:
        perplexity = math.exp(loss)
    except OverflowError as exc:
        raise FloatingPointError(f"loss {loss!r} overflows perplexity") from exc
    return finite_metric("perplexity", perplexity, positive=True)
