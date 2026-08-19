# Copyright 2023 The JAX Authors.
# Copyright 2026 TPU Flash contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dense causal attention backends for TPU Flash.

The public interface deliberately resembles JAX SplashAttention: callers
construct a static configuration once and receive a ``(q, k, v) -> output``
callable.  Inputs use logical ``[batch, heads, sequence, head_dim]`` (BHSD)
layout.  Tile sizes are compilation parameters; they never become dynamic JAX
values and are never benchmarked while a training executable is compiling.

There are three deliberately distinct implementations:

``jax_flash``
    JAX 0.11's supported, trainable TPU FlashAttention kernel.  We supply a
    shape-selected block configuration, including backward blocks.  This is a
    baseline wrapper and is not described as the custom TPU Flash kernel.
``tpu_flash``
    Experimental Pallas forward and separate dQ/dKV kernels implementing
    tiled online softmax.  They never materialize ``[sequence, sequence]``
    scores in HBM.
``reference``
    A small pure-JAX correctness oracle.  It is differentiable but constructs
    the complete attention matrix and is therefore inappropriate for a real
    rig.

The Pallas scheduling and online-softmax recurrence are a minimal adaptation
of ``jax.experimental.pallas.ops.tpu.flash_attention`` from JAX 0.11.0
(Apache-2.0).  The fork intentionally supports only dense causal self-attention
while the numerical contract and tiling strategy are being established:

https://github.com/jax-ml/jax/blob/jax-v0.11.0/jax/experimental/pallas/ops/tpu/flash_attention.py
"""

from __future__ import annotations

from dataclasses import dataclass
import functools
from typing import Any, Callable, Literal

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas.ops.tpu import flash_attention as jax_flash_attention
import jax.numpy as jnp

from .autotune import (
    AttentionTilePlan as AttentionTiles,
    generate_attention_tile_candidates,
    is_legal_attention_tile_plan,
    padded_sequence_length,
)


Backend = Literal["auto", "tpu_flash", "jax_flash", "reference"]
TPU_VECTOR_LANES = 128
TPU_SUBLANES = 8
_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.float32).max)
_QK_DIM_NUMBERS = (((1,), (1,)), ((), ()))


@dataclass(frozen=True, slots=True)
class AttentionConfig:
    """Static attention configuration captured outside the training step.

    ``interpret`` runs the Pallas kernels through their CPU-compatible
    interpreter and is intended only for unit tests.  ``auto`` selects the
    supported JAX kernel on TPU and the correctness reference elsewhere;
    select ``tpu_flash`` explicitly to use the custom implementation.
    """

    backend: Backend = "auto"
    tiles: AttentionTiles | None = None
    softmax_scale: float | None = None
    interpret: bool = False
    debug: bool = False

    def __post_init__(self) -> None:
        if self.backend not in ("auto", "tpu_flash", "jax_flash", "reference"):
            raise ValueError(f"unknown attention backend: {self.backend!r}")
        if self.softmax_scale is not None and self.softmax_scale <= 0.0:
            raise ValueError("softmax_scale must be positive")
        if self.interpret and self.backend not in ("tpu_flash", "auto"):
            raise ValueError("interpret mode applies only to the TPU Flash backend")


def attention_tile_candidates(
    *, sequence: int, head_dim: int, training: bool = True
) -> tuple[AttentionTiles, ...]:
    """Return legal, bounded 128-aligned candidates for offline tuning."""

    return generate_attention_tile_candidates(
        sequence=sequence,
        head_dim=head_dim,
        mode="forward_backward" if training else "forward",
    )


def select_attention_tiles(
    *, sequence: int, head_dim: int, training: bool = True
) -> AttentionTiles:
    """Choose deterministic tiles without performing runtime benchmarking.

    The preferred ``512/512/256`` neighborhood is the best measured setting
    for the repository's per-chip ``B=8,H=12,T=1024,D=64`` TPU-v4 workload.
    For other shapes the closest legal candidate is selected deterministically.
    Exact runtime-fingerprinted cache/LUT resolution and explicit synthetic
    bootstrapping live in :mod:`rig.kernels.autotune`.
    """

    candidates = attention_tile_candidates(
        sequence=sequence, head_dim=head_dim, training=training
    )
    target = (512, 512, 256)

    def rank(tiles: AttentionTiles) -> tuple[int, int, int, int]:
        distance = (
            abs(tiles.block_q - target[0])
            + abs(tiles.block_kv - target[1])
            + abs(tiles.block_kv_compute - target[2])
        )
        return (
            distance,
            -tiles.block_kv_compute,
            -tiles.block_q,
            -tiles.block_kv,
        )

    return min(candidates, key=rank)


def _validate_qkv(q: jax.Array, k: jax.Array, v: jax.Array) -> None:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(
            "q, k, and v must use BHSD rank-4 layout; "
            f"got {q.shape}, {k.shape}, and {v.shape}"
        )
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(
            "dense causal self-attention requires identical q/k/v shapes; "
            f"got {q.shape}, {k.shape}, and {v.shape}"
        )
    if not all(q.shape):
        raise ValueError(f"q/k/v dimensions must be nonzero, got {q.shape}")
    if q.shape[-1] > TPU_VECTOR_LANES or q.shape[-1] % TPU_SUBLANES:
        raise ValueError(
            "the TPU kernel requires head_dim <= 128 and divisible by 8; "
            f"got {q.shape[-1]}"
        )
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise TypeError(
            f"q, k, and v dtypes must match; got {q.dtype}, {k.dtype}, {v.dtype}"
        )
    if not jnp.issubdtype(q.dtype, jnp.floating):
        raise TypeError(f"q, k, and v must be floating point, got {q.dtype}")


def reference_causal_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    softmax_scale: float | None = None,
    segment_ids: jax.Array | None = None,
) -> jax.Array:
    """Pure-JAX causal attention oracle using an FP32 softmax.

    ``segment_ids`` is an optional ``[batch, sequence]`` int32 array giving each
    position's document index inside its window. Positions may attend only
    within their own segment, which makes attention block-diagonal over
    documents on top of the causal mask. This is the oracle the Pallas kernels
    are checked against, so it defines the semantics.
    """

    _validate_qkv(q, k, v)
    sequence = q.shape[2]
    scale = float(q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale)
    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k, preferred_element_type=jnp.float32)
    scores = scores.astype(jnp.float32) * scale
    row = lax.broadcasted_iota(jnp.int32, (sequence, sequence), 0)
    column = lax.broadcasted_iota(jnp.int32, (sequence, sequence), 1)
    visible = column <= row
    if segment_ids is not None:
        # [batch, 1, q, k]: same document only. Broadcasts across heads.
        same = segment_ids[:, None, :, None] == segment_ids[:, None, None, :]
        visible = jnp.logical_and(visible[None, None], same)
    scores = jnp.where(visible, scores, _MASK_VALUE)
    probabilities = jax.nn.softmax(scores, axis=-1).astype(v.dtype)
    return jnp.einsum(
        "bhqk,bhkd->bhqd",
        probabilities,
        v,
        preferred_element_type=jnp.float32,
    ).astype(q.dtype)


def _below_or_on_diagonal(
    q_block_index: jax.Array,
    block_q: int,
    kv_block_index: jax.Array,
    block_kv: int,
) -> jax.Array:
    """Whether the bottom-left corner of a tile is causal/visible."""

    return ((q_block_index + 1) * block_q - 1) >= kv_block_index * block_kv


def _broadcast_row(value: jax.Array, width: int) -> jax.Array:
    """Broadcast TPU's 128-lane replicated row statistic to ``width``."""

    # Reductions over a matrix formed in the kernel produce logical ``[rows,
    # 1]`` values, whereas saved/VMEM row statistics use TPU's physically
    # replicated ``[rows, 128]`` layout.  Normalize both representations before
    # tiling; merely tiling a logical scalar column would yield ``[rows, 2]``
    # for width=256 instead of the required ``[rows, 256]``.
    if value.shape[1] == 1:
        value = jnp.broadcast_to(value, (value.shape[0], TPU_VECTOR_LANES))
    if width < TPU_VECTOR_LANES:
        return value[:, :width]
    return jnp.tile(value, (1, width // TPU_VECTOR_LANES))


def _tpu_flash_forward_kernel(
    *refs: Any,
    block_q: int,
    block_kv: int,
    block_kv_compute: int,
    sequence: int,
    valid_sequence: int,
    softmax_scale: float,
    has_segments: bool = False,
) -> None:
    """One online-softmax pass over a query tile and successive KV tiles.

    The segment refs are present only when document masking is requested, so
    the operand list is unpacked rather than named: passing a dummy array
    instead would move a block of it through VMEM on every tile for nothing.
    """

    if has_segments:
        (
            q_ref,
            k_ref,
            v_ref,
            q_segment_ref,
            kv_segment_ref,
            output_ref,
            logsumexp_ref,
            max_scratch_ref,
            sum_scratch_ref,
            accumulator_scratch_ref,
        ) = refs
    else:
        (
            q_ref,
            k_ref,
            v_ref,
            output_ref,
            logsumexp_ref,
            max_scratch_ref,
            sum_scratch_ref,
            accumulator_scratch_ref,
        ) = refs
        q_segment_ref = kv_segment_ref = None

    kv_block_index = pl.program_id(3)

    @pl.when(kv_block_index == 0)
    def initialize() -> None:
        max_scratch_ref[...] = jnp.full(max_scratch_ref.shape, -jnp.inf, jnp.float32)
        sum_scratch_ref[...] = jnp.zeros(sum_scratch_ref.shape, jnp.float32)
        accumulator_scratch_ref[...] = jnp.zeros(
            accumulator_scratch_ref.shape, jnp.float32
        )

    q_block_index = pl.program_id(2)
    should_run = _below_or_on_diagonal(q_block_index, block_q, kv_block_index, block_kv)

    @pl.when(should_run)
    def visit_kv_tile() -> None:
        q_tile = q_ref[0, 0, :, :]
        for start_kv in range(0, block_kv, block_kv_compute):
            max_previous = max_scratch_ref[0, 0, :, :]
            sum_previous = sum_scratch_ref[0, 0, :, :]
            key_tile = k_ref[0, 0, pl.dslice(start_kv, block_kv_compute), :]
            scores = lax.dot_general(
                q_tile,
                key_tile,
                _QK_DIM_NUMBERS,
                preferred_element_type=jnp.float32,
            )
            scores *= softmax_scale

            row_ids = lax.broadcasted_iota(jnp.int32, (block_q, block_kv_compute), 0)
            row_ids += q_block_index * block_q
            column_ids = lax.broadcasted_iota(jnp.int32, (block_q, block_kv_compute), 1)
            column_ids += kv_block_index * block_kv + start_kv
            mask = jnp.logical_and(column_ids <= row_ids, column_ids < valid_sequence)
            if q_segment_ref is not None:
                # Block-diagonal over documents: a position may attend only
                # inside its own segment. Purely additive to the causal mask,
                # so the tile-skipping above stays valid.
                rows = q_segment_ref[:][:, None]
                columns = kv_segment_ref[start_kv : start_kv + block_kv_compute]
                mask = jnp.logical_and(mask, rows == columns[None, :])
            scores += jnp.where(mask, 0.0, _MASK_VALUE)

            max_current = jnp.max(scores, axis=1)[:, None]
            max_next = jnp.maximum(max_previous, max_current)
            probabilities = jnp.exp(scores - _broadcast_row(max_next, block_kv_compute))
            correction = jnp.exp(max_previous - max_next)
            corrected_sum = correction * sum_previous
            sum_next = jnp.sum(probabilities, axis=1)[:, None] + corrected_sum
            inverse_sum = jnp.where(sum_next == 0.0, 1.0, 1.0 / sum_next)

            accumulator_scratch_ref[0, 0, :, :] *= _broadcast_row(
                corrected_sum * inverse_sum, q_tile.shape[-1]
            )
            value_tile = v_ref[0, 0, pl.dslice(start_kv, block_kv_compute), :]
            contribution = lax.dot(
                probabilities.astype(value_tile.dtype),
                value_tile,
                preferred_element_type=jnp.float32,
            )
            accumulator_scratch_ref[0, 0, :, :] += contribution * _broadcast_row(
                inverse_sum, q_tile.shape[-1]
            )
            max_scratch_ref[0, 0, :, :] = max_next
            sum_scratch_ref[0, 0, :, :] = sum_next

    @pl.when(kv_block_index == sequence // block_kv - 1)
    def store() -> None:
        output_ref[...] = accumulator_scratch_ref[...].astype(output_ref.dtype)
        logsumexp = jnp.log(sum_scratch_ref[...]) + max_scratch_ref[...]
        logsumexp_ref[...] = logsumexp.astype(logsumexp_ref.dtype)


def _tpu_flash_forward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    tiles: AttentionTiles,
    softmax_scale: float,
    valid_sequence: int,
    interpret: bool,
    debug: bool,
    segment_ids: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    batch, heads, sequence, head_dim = q.shape
    if segment_ids is not None and segment_ids.shape != (batch, sequence):
        raise ValueError(
            "segment_ids must be [batch, sequence]; got "
            f"{segment_ids.shape} for q {q.shape}"
        )
    if sequence % tiles.block_q or sequence % tiles.block_kv:
        raise ValueError(
            "TPU Flash tiles must divide the sequence length; "
            f"got T={sequence}, block_q={tiles.block_q}, block_kv={tiles.block_kv}"
        )
    if tiles.block_q % TPU_VECTOR_LANES:
        raise ValueError("block_q must be a multiple of 128")
    if tiles.block_kv % tiles.block_kv_compute:
        raise ValueError("block_kv_compute must divide block_kv")
    if tiles.block_kv_compute % TPU_VECTOR_LANES:
        raise ValueError("block_kv_compute must be a multiple of 128")

    grid = (
        batch,
        heads,
        sequence // tiles.block_q,
        sequence // tiles.block_kv,
    )

    def q_index(batch_index: Any, head_index: Any, q_index: Any, _: Any):
        return batch_index, head_index, q_index, 0

    def kv_index(batch_index: Any, head_index: Any, q_index: Any, kv_index: Any):
        # Skipped causal tiles still need a legal prefetch address.
        visible_index = lax.select(
            _below_or_on_diagonal(q_index, tiles.block_q, kv_index, tiles.block_kv),
            kv_index,
            0,
        )
        return batch_index, head_index, visible_index, 0

    def q_segment_index(batch_index: Any, _head: Any, q_index: Any, _: Any):
        return batch_index, 0, q_index

    def kv_segment_index(batch_index: Any, _head: Any, q_index: Any, kv_index: Any):
        visible_index = lax.select(
            _below_or_on_diagonal(q_index, tiles.block_q, kv_index, tiles.block_kv),
            kv_index,
            0,
        )
        return batch_index, 0, visible_index

    q_spec = pl.BlockSpec((1, 1, tiles.block_q, head_dim), q_index)
    kv_spec = pl.BlockSpec((1, 1, tiles.block_kv, head_dim), kv_index)
    # Segments are per (batch, position); heads share them.
    q_segment_spec = pl.BlockSpec((None, None, tiles.block_q), q_segment_index)
    kv_segment_spec = pl.BlockSpec((None, None, tiles.block_kv), kv_segment_index)
    output_spec = pl.BlockSpec((1, 1, tiles.block_q, head_dim), q_index)
    logsumexp_spec = pl.BlockSpec((1, 1, tiles.block_q, TPU_VECTOR_LANES), q_index)
    output_shapes = (
        jax.ShapeDtypeStruct(q.shape, q.dtype),
        jax.ShapeDtypeStruct((batch, heads, sequence, TPU_VECTOR_LANES), jnp.float32),
    )
    kernel = functools.partial(
        _tpu_flash_forward_kernel,
        block_q=tiles.block_q,
        block_kv=tiles.block_kv,
        block_kv_compute=tiles.block_kv_compute,
        sequence=sequence,
        valid_sequence=valid_sequence,
        softmax_scale=softmax_scale,
        has_segments=segment_ids is not None,
    )

    with jax.named_scope(
        "tpu_flash_attention_"
        f"q{tiles.block_q}_kv{tiles.block_kv}_inner{tiles.block_kv_compute}"
    ):
        output, logsumexp_replicated = pl.pallas_call(
            kernel,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=(
                    q_spec,
                    kv_spec,
                    kv_spec,
                    *(
                        (q_segment_spec, kv_segment_spec)
                        if segment_ids is not None
                        else ()
                    ),
                ),
                out_specs=(output_spec, logsumexp_spec),
                scratch_shapes=(
                    pltpu.VMEM((1, 1, tiles.block_q, TPU_VECTOR_LANES), jnp.float32),
                    pltpu.VMEM((1, 1, tiles.block_q, TPU_VECTOR_LANES), jnp.float32),
                    pltpu.VMEM((1, 1, tiles.block_q, head_dim), jnp.float32),
                ),
            ),
            out_shape=output_shapes,
            interpret=interpret,
            debug=debug,
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel", "parallel", "parallel", "arbitrary")
            ),
            name="tpu_flash_causal_attention_fwd",
        )(
            q,
            k,
            v,
            *(
                (segment_ids[:, None, :], segment_ids[:, None, :])
                if segment_ids is not None
                else ()
            ),
        )
    return output, logsumexp_replicated[..., 0]


def _tpu_flash_dq_kernel(
    *refs: Any,
    block_q: int,
    block_kv: int,
    block_kv_compute: int,
    sequence: int,
    valid_sequence: int,
    softmax_scale: float,
    has_segments: bool = False,
) -> None:
    """Query-owned backward pass; recomputes P and accumulates dQ."""

    if has_segments:
        (
            q_ref,
            k_ref,
            v_ref,
            output_ref,
            output_cotangent_ref,
            logsumexp_ref,
            q_segment_ref,
            kv_segment_ref,
            dq_ref,
            dq_scratch_ref,
        ) = refs
    else:
        (
            q_ref,
            k_ref,
            v_ref,
            output_ref,
            output_cotangent_ref,
            logsumexp_ref,
            dq_ref,
            dq_scratch_ref,
        ) = refs
        q_segment_ref = kv_segment_ref = None

    kv_block_index = pl.program_id(3)

    @pl.when(kv_block_index == 0)
    def initialize() -> None:
        dq_scratch_ref[...] = jnp.zeros(dq_scratch_ref.shape, jnp.float32)

    q_block_index = pl.program_id(2)
    should_run = _below_or_on_diagonal(q_block_index, block_q, kv_block_index, block_kv)

    @pl.when(should_run)
    def visit_kv_tile() -> None:
        q_tile = q_ref[0, 0, :, :]
        output_tile = output_ref[0, 0, :, :]
        cotangent_tile = output_cotangent_ref[0, 0, :, :]
        logsumexp = logsumexp_ref[0, 0, :, :]
        delta = jnp.sum(
            output_tile.astype(jnp.float32) * cotangent_tile.astype(jnp.float32),
            axis=1,
        )[:, None]
        for start_kv in range(0, block_kv, block_kv_compute):
            key_tile = k_ref[0, 0, pl.dslice(start_kv, block_kv_compute), :]
            value_tile = v_ref[0, 0, pl.dslice(start_kv, block_kv_compute), :]
            scores = lax.dot_general(
                q_tile,
                key_tile,
                _QK_DIM_NUMBERS,
                preferred_element_type=jnp.float32,
            )
            scores *= softmax_scale
            row_ids = lax.broadcasted_iota(jnp.int32, (block_q, block_kv_compute), 0)
            row_ids += q_block_index * block_q
            column_ids = lax.broadcasted_iota(jnp.int32, (block_q, block_kv_compute), 1)
            column_ids += kv_block_index * block_kv + start_kv
            mask = jnp.logical_and(column_ids <= row_ids, column_ids < valid_sequence)
            if q_segment_ref is not None:
                rows = q_segment_ref[:][:, None]
                columns = kv_segment_ref[start_kv : start_kv + block_kv_compute]
                mask = jnp.logical_and(mask, rows == columns[None, :])
            probabilities = jnp.exp(
                scores - _broadcast_row(logsumexp, block_kv_compute)
            )
            probabilities = jnp.where(mask, probabilities, 0.0)
            dp = lax.dot_general(
                cotangent_tile,
                value_tile,
                _QK_DIM_NUMBERS,
                preferred_element_type=jnp.float32,
            )
            ds = (dp - _broadcast_row(delta, block_kv_compute)) * probabilities
            ds *= softmax_scale
            dq_scratch_ref[0, 0, :, :] += lax.dot(
                ds.astype(key_tile.dtype),
                key_tile,
                preferred_element_type=jnp.float32,
            )

    @pl.when(kv_block_index == sequence // block_kv - 1)
    def store() -> None:
        dq_ref[...] = dq_scratch_ref[...].astype(dq_ref.dtype)


def _tpu_flash_dkv_kernel(
    *refs: Any,
    block_q: int,
    block_kv: int,
    block_q_compute: int,
    block_kv_compute: int,
    sequence: int,
    valid_sequence: int,
    softmax_scale: float,
    has_segments: bool = False,
) -> None:
    """KV-owned backward pass; recomputes P and accumulates dK and dV."""

    if has_segments:
        (
            q_ref,
            k_ref,
            v_ref,
            output_ref,
            output_cotangent_ref,
            logsumexp_ref,
            q_segment_ref,
            kv_segment_ref,
            dk_ref,
            dv_ref,
            dk_scratch_ref,
            dv_scratch_ref,
        ) = refs
    else:
        (
            q_ref,
            k_ref,
            v_ref,
            output_ref,
            output_cotangent_ref,
            logsumexp_ref,
            dk_ref,
            dv_ref,
            dk_scratch_ref,
            dv_scratch_ref,
        ) = refs
        q_segment_ref = kv_segment_ref = None

    q_block_index = pl.program_id(3)

    @pl.when(q_block_index == 0)
    def initialize() -> None:
        dk_scratch_ref[...] = jnp.zeros(dk_scratch_ref.shape, jnp.float32)
        dv_scratch_ref[...] = jnp.zeros(dv_scratch_ref.shape, jnp.float32)

    kv_block_index = pl.program_id(2)
    should_run = _below_or_on_diagonal(q_block_index, block_q, kv_block_index, block_kv)

    @pl.when(should_run)
    def visit_q_tile() -> None:
        for start_q in range(0, block_q, block_q_compute):
            q_tile = q_ref[0, 0, pl.dslice(start_q, block_q_compute), :]
            output_tile = output_ref[0, 0, pl.dslice(start_q, block_q_compute), :]
            cotangent_tile = output_cotangent_ref[
                0, 0, pl.dslice(start_q, block_q_compute), :
            ]
            logsumexp = logsumexp_ref[0, 0, pl.dslice(start_q, block_q_compute), :]
            delta = jnp.sum(
                output_tile.astype(jnp.float32) * cotangent_tile.astype(jnp.float32),
                axis=1,
            )[:, None]
            for start_kv in range(0, block_kv, block_kv_compute):
                key_tile = k_ref[0, 0, pl.dslice(start_kv, block_kv_compute), :]
                value_tile = v_ref[0, 0, pl.dslice(start_kv, block_kv_compute), :]
                scores = lax.dot_general(
                    q_tile,
                    key_tile,
                    _QK_DIM_NUMBERS,
                    preferred_element_type=jnp.float32,
                )
                scores *= softmax_scale
                row_ids = lax.broadcasted_iota(
                    jnp.int32, (block_q_compute, block_kv_compute), 0
                )
                row_ids += q_block_index * block_q + start_q
                column_ids = lax.broadcasted_iota(
                    jnp.int32, (block_q_compute, block_kv_compute), 1
                )
                column_ids += kv_block_index * block_kv + start_kv
                mask = jnp.logical_and(
                    column_ids <= row_ids, column_ids < valid_sequence
                )
                if q_segment_ref is not None:
                    rows = q_segment_ref[start_q : start_q + block_q_compute][
                        :, None
                    ]
                    columns = kv_segment_ref[start_kv : start_kv + block_kv_compute]
                    mask = jnp.logical_and(mask, rows == columns[None, :])
                probabilities = jnp.exp(
                    scores - _broadcast_row(logsumexp, block_kv_compute)
                )
                probabilities = jnp.where(mask, probabilities, 0.0)
                dv_scratch_ref[0, 0, pl.dslice(start_kv, block_kv_compute), :] += (
                    lax.dot(
                        probabilities.T.astype(cotangent_tile.dtype),
                        cotangent_tile,
                        preferred_element_type=jnp.float32,
                    )
                )
                dp = lax.dot_general(
                    cotangent_tile,
                    value_tile,
                    _QK_DIM_NUMBERS,
                    preferred_element_type=jnp.float32,
                )
                ds = (dp - _broadcast_row(delta, block_kv_compute)) * probabilities
                ds *= softmax_scale
                dk_scratch_ref[0, 0, pl.dslice(start_kv, block_kv_compute), :] += (
                    lax.dot(
                        ds.T.astype(q_tile.dtype),
                        q_tile,
                        preferred_element_type=jnp.float32,
                    )
                )

    @pl.when(q_block_index == sequence // block_q - 1)
    def store() -> None:
        dk_ref[...] = dk_scratch_ref[...].astype(dk_ref.dtype)
        dv_ref[...] = dv_scratch_ref[...].astype(dv_ref.dtype)


def _tpu_flash_backward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    output: jax.Array,
    logsumexp: jax.Array,
    output_cotangent: jax.Array,
    *,
    tiles: AttentionTiles,
    softmax_scale: float,
    valid_sequence: int,
    interpret: bool,
    debug: bool,
    segment_ids: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Launch separate query-owned dQ and key-owned dK/dV kernels."""

    if not tiles.has_backward_tiles:
        raise ValueError("TPU Flash backward requires a complete tile plan")
    assert tiles.block_q_dkv is not None
    assert tiles.block_q_dkv_compute is not None
    assert tiles.block_kv_dkv is not None
    assert tiles.block_kv_dkv_compute is not None
    assert tiles.block_q_dq is not None
    assert tiles.block_kv_dq is not None
    assert tiles.block_kv_dq_compute is not None
    batch, heads, sequence, head_dim = q.shape
    # Store one scalar statistic in TPU's 128-lane native row layout.
    logsumexp_replicated = jnp.broadcast_to(
        logsumexp[..., None], (*logsumexp.shape, TPU_VECTOR_LANES)
    )

    def q_index(batch_index: Any, head_index: Any, q_index: Any, _: Any):
        return batch_index, head_index, q_index, 0

    def dq_kv_index(batch_index: Any, head_index: Any, q_index: Any, kv_index: Any):
        visible = lax.select(
            _below_or_on_diagonal(
                q_index, tiles.block_q_dq, kv_index, tiles.block_kv_dq
            ),
            kv_index,
            0,
        )
        return batch_index, head_index, visible, 0

    def dq_q_segment_index(batch_index: Any, _head: Any, q_index: Any, _: Any):
        return batch_index, 0, q_index

    def dq_kv_segment_index(batch_index: Any, _head: Any, q_index: Any, kv_index: Any):
        visible = lax.select(
            _below_or_on_diagonal(
                q_index, tiles.block_q_dq, kv_index, tiles.block_kv_dq
            ),
            kv_index,
            0,
        )
        return batch_index, 0, visible

    dq_q_spec = pl.BlockSpec((1, 1, tiles.block_q_dq, head_dim), q_index)
    dq_kv_spec = pl.BlockSpec((1, 1, tiles.block_kv_dq, head_dim), dq_kv_index)
    dq_lse_spec = pl.BlockSpec((1, 1, tiles.block_q_dq, TPU_VECTOR_LANES), q_index)
    dq_q_segment_spec = pl.BlockSpec((None, None, tiles.block_q_dq), dq_q_segment_index)
    dq_kv_segment_spec = pl.BlockSpec((None, None, tiles.block_kv_dq), dq_kv_segment_index)
    segment_operands = (
        (segment_ids[:, None, :], segment_ids[:, None, :])
        if segment_ids is not None
        else ()
    )
    dq_kernel = functools.partial(
        _tpu_flash_dq_kernel,
        block_q=tiles.block_q_dq,
        block_kv=tiles.block_kv_dq,
        block_kv_compute=tiles.block_kv_dq_compute,
        sequence=sequence,
        valid_sequence=valid_sequence,
        softmax_scale=softmax_scale,
        has_segments=segment_ids is not None,
    )
    dq = pl.pallas_call(
        dq_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(
                batch,
                heads,
                sequence // tiles.block_q_dq,
                sequence // tiles.block_kv_dq,
            ),
            in_specs=(
                dq_q_spec,
                dq_kv_spec,
                dq_kv_spec,
                dq_q_spec,
                dq_q_spec,
                dq_lse_spec,
                *(
                    (dq_q_segment_spec, dq_kv_segment_spec)
                    if segment_ids is not None
                    else ()
                ),
            ),
            out_specs=dq_q_spec,
            scratch_shapes=(
                pltpu.VMEM((1, 1, tiles.block_q_dq, head_dim), jnp.float32),
            ),
        ),
        out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
        interpret=interpret,
        debug=debug,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel", "arbitrary")
        ),
        name="tpu_flash_causal_attention_bwd_dq",
    )(q, k, v, output, output_cotangent, logsumexp_replicated, *segment_operands)

    def dkv_kv_index(batch_index: Any, head_index: Any, kv_index: Any, _: Any):
        return batch_index, head_index, kv_index, 0

    def dkv_q_index(batch_index: Any, head_index: Any, kv_index: Any, q_index: Any):
        visible = lax.select(
            _below_or_on_diagonal(
                q_index, tiles.block_q_dkv, kv_index, tiles.block_kv_dkv
            ),
            q_index,
            0,
        )
        return batch_index, head_index, visible, 0

    dkv_q_spec = pl.BlockSpec((1, 1, tiles.block_q_dkv, head_dim), dkv_q_index)
    dkv_lse_spec = pl.BlockSpec(
        (1, 1, tiles.block_q_dkv, TPU_VECTOR_LANES), dkv_q_index
    )
    dkv_kv_spec = pl.BlockSpec((1, 1, tiles.block_kv_dkv, head_dim), dkv_kv_index)

    def dkv_kv_segment_index(batch_index: Any, _head: Any, kv_index: Any, _: Any):
        return batch_index, 0, kv_index

    def dkv_q_segment_index(batch_index: Any, _head: Any, kv_index: Any, q_index: Any):
        visible = lax.select(
            _below_or_on_diagonal(
                q_index, tiles.block_q_dkv, kv_index, tiles.block_kv_dkv
            ),
            q_index,
            0,
        )
        return batch_index, 0, visible

    dkv_q_segment_spec = pl.BlockSpec((None, None, tiles.block_q_dkv), dkv_q_segment_index)
    dkv_kv_segment_spec = pl.BlockSpec((None, None, tiles.block_kv_dkv), dkv_kv_segment_index)
    dkv_kernel = functools.partial(
        _tpu_flash_dkv_kernel,
        block_q=tiles.block_q_dkv,
        block_kv=tiles.block_kv_dkv,
        block_q_compute=tiles.block_q_dkv_compute,
        block_kv_compute=tiles.block_kv_dkv_compute,
        sequence=sequence,
        valid_sequence=valid_sequence,
        softmax_scale=softmax_scale,
        has_segments=segment_ids is not None,
    )
    dk, dv = pl.pallas_call(
        dkv_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(
                batch,
                heads,
                sequence // tiles.block_kv_dkv,
                sequence // tiles.block_q_dkv,
            ),
            in_specs=(
                dkv_q_spec,
                dkv_kv_spec,
                dkv_kv_spec,
                dkv_q_spec,
                dkv_q_spec,
                dkv_lse_spec,
                *(
                    (dkv_q_segment_spec, dkv_kv_segment_spec)
                    if segment_ids is not None
                    else ()
                ),
            ),
            out_specs=(dkv_kv_spec, dkv_kv_spec),
            scratch_shapes=(
                pltpu.VMEM((1, 1, tiles.block_kv_dkv, head_dim), jnp.float32),
                pltpu.VMEM((1, 1, tiles.block_kv_dkv, head_dim), jnp.float32),
            ),
        ),
        out_shape=(
            jax.ShapeDtypeStruct(k.shape, k.dtype),
            jax.ShapeDtypeStruct(v.shape, v.dtype),
        ),
        interpret=interpret,
        debug=debug,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel", "arbitrary")
        ),
        name="tpu_flash_causal_attention_bwd_dkv",
    )(q, k, v, output, output_cotangent, logsumexp_replicated, *segment_operands)
    return dq, dk, dv


def _official_flash_blocks(tiles: AttentionTiles) -> jax_flash_attention.BlockSizes:
    """Translate Splash-style tile names to JAX FlashAttention 0.11 names."""

    if not tiles.has_backward_tiles:
        raise ValueError("jax_flash training requires a complete backward tile plan")
    assert tiles.block_q_dkv is not None
    assert tiles.block_q_dkv_compute is not None
    assert tiles.block_kv_dkv is not None
    assert tiles.block_kv_dkv_compute is not None
    assert tiles.block_q_dq is not None
    assert tiles.block_kv_dq is not None
    assert tiles.block_kv_dq_compute is not None
    return jax_flash_attention.BlockSizes(
        block_q=tiles.block_q,
        block_k_major=tiles.block_kv,
        block_k=tiles.block_kv_compute,
        block_b=1,
        block_q_major_dkv=tiles.block_q_dkv,
        block_k_major_dkv=tiles.block_kv_dkv,
        block_k_dkv=tiles.block_kv_dkv_compute,
        block_q_dkv=tiles.block_q_dkv_compute,
        block_k_major_dq=tiles.block_kv_dq,
        block_k_dq=tiles.block_kv_dq_compute,
        block_q_dq=tiles.block_q_dq,
    )


def _jax_flash(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    tiles: AttentionTiles,
    softmax_scale: float,
    debug: bool,
    segment_ids: jax_flash_attention.SegmentIds | None = None,
) -> jax.Array:
    return jax_flash_attention.flash_attention(
        q,
        k,
        v,
        segment_ids=segment_ids,
        causal=True,
        sm_scale=softmax_scale,
        block_sizes=_official_flash_blocks(tiles),
        debug=debug,
    )


def _resolve_backend(config: AttentionConfig) -> Backend:
    if config.backend != "auto":
        return config.backend
    return "jax_flash" if jax.default_backend() == "tpu" else "reference"


def make_causal_attention(config: AttentionConfig = AttentionConfig()) -> Callable:
    """Create a shape-polymorphic-looking callable with static shape tuning.

    The returned Python callable selects tiles from the first concrete input
    shape during tracing.  Each distinct shape still produces a distinct XLA
    executable, as expected for Pallas block sizes.
    """

    backend = _resolve_backend(config)

    def attention(
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        segment_ids: jax.Array | None = None,
    ) -> jax.Array:
        """Causal attention, optionally block-diagonal over documents.

        ``segment_ids`` is ``[batch, sequence]`` int32: positions sharing an id
        may attend to each other, subject to causality. Padding introduced to
        reach a legal tile length is given a distinct id so it matches nothing.
        """

        _validate_qkv(q, k, v)
        if backend == "tpu_flash" and not config.interpret and q.dtype != jnp.bfloat16:
            raise TypeError(
                "the compiled TPU Flash kernel currently supports bfloat16 q/k/v; "
                f"got {q.dtype}"
            )
        original_sequence = q.shape[2]
        scale = float(
            q.shape[-1] ** -0.5
            if config.softmax_scale is None
            else config.softmax_scale
        )
        if backend == "reference":
            return reference_causal_attention(
                q, k, v, softmax_scale=scale, segment_ids=segment_ids
            )
        tiles = config.tiles or select_attention_tiles(
            sequence=q.shape[2], head_dim=q.shape[3], training=True
        )
        # TPU attention tiles are 128-wide.  Right-padding K/V is safe only if
        # padded keys are explicitly excluded.  Causality alone would let a
        # padded query see padded keys, so the official backend receives two
        # O(T) segment-ID arrays.  The custom kernel uses ``valid_sequence``
        # directly in its mask.  Outputs are sliced back to the logical
        # sequence; reverse-mode AD therefore slices/pads cotangents
        # symmetrically.
        padded_sequence = padded_sequence_length(original_sequence)
        if not is_legal_attention_tile_plan(
            tiles,
            sequence=original_sequence,
            head_dim=q.shape[3],
            mode="forward_backward",
        ):
            raise ValueError(
                "attention tile plan is not legal for "
                f"sequence={original_sequence}, head_dim={q.shape[3]}"
            )
        if padded_sequence != original_sequence:
            padding = (
                (0, 0),
                (0, 0),
                (0, padded_sequence - original_sequence),
                (0, 0),
            )
            q_padded = jnp.pad(q, padding)
            k_padded = jnp.pad(k, padding)
            v_padded = jnp.pad(v, padding)
            if segment_ids is not None:
                # -1 matches no real document, so padding stays invisible even
                # before the valid_sequence bound is applied.
                segments_padded = jnp.pad(
                    segment_ids,
                    ((0, 0), (0, padded_sequence - original_sequence)),
                    constant_values=-1,
                )
            else:
                segments_padded = None
        else:
            q_padded, k_padded, v_padded = q, k, v
            segments_padded = segment_ids
        if backend == "jax_flash":
            if padded_sequence != original_sequence:
                # Segment zero contains real tokens and segment one contains
                # right-padding.  Real queries therefore cannot see padding,
                # without constructing an O(T^2) bias tensor.
                segment = (jnp.arange(padded_sequence) >= original_sequence).astype(
                    jnp.int32
                )
                segment = jnp.broadcast_to(
                    segment[None, :], (q.shape[0], padded_sequence)
                )
                if segments_padded is not None:
                    # Distinct ids per (document, padding) pair.
                    segment = segments_padded * 2 + segment
                flash_segments = jax_flash_attention.SegmentIds(q=segment, kv=segment)
            elif segments_padded is not None:
                flash_segments = jax_flash_attention.SegmentIds(
                    q=segments_padded, kv=segments_padded
                )
            else:
                flash_segments = None
            output = _jax_flash(
                q_padded,
                k_padded,
                v_padded,
                tiles=tiles,
                softmax_scale=scale,
                debug=config.debug,
                segment_ids=flash_segments,
            )
            return output[:, :, :original_sequence, :]

        # ``segments`` is an explicit argument, never a closure capture. A
        # traced array closed over by a custom_vjp is hoisted into the
        # computation as an implicit operand -- one per call site, so a
        # 12-layer model compiled for 12 more inputs than the caller passes
        # ("compiled for 385 inputs but called with 373"). Passing it through
        # and returning a ``None`` cotangent keeps the arity honest.
        @jax.custom_vjp
        def tpu_flash_attention(
            q_value: jax.Array,
            k_value: jax.Array,
            v_value: jax.Array,
            segments: jax.Array | None,
        ) -> jax.Array:
            output, _ = _tpu_flash_forward(
                q_value,
                k_value,
                v_value,
                tiles=tiles,
                softmax_scale=scale,
                valid_sequence=original_sequence,
                interpret=config.interpret,
                debug=config.debug,
                segment_ids=segments,
            )
            return output[:, :, :original_sequence, :]

        def forward_rule(
            q_value: jax.Array,
            k_value: jax.Array,
            v_value: jax.Array,
            segments: jax.Array | None,
        ):
            output, logsumexp = _tpu_flash_forward(
                q_value,
                k_value,
                v_value,
                tiles=tiles,
                softmax_scale=scale,
                valid_sequence=original_sequence,
                interpret=config.interpret,
                debug=config.debug,
                segment_ids=segments,
            )
            logical_output = output[:, :, :original_sequence, :]
            return logical_output, (
                q_value,
                k_value,
                v_value,
                output,
                logsumexp,
                segments,
            )

        def backward_rule(
            residuals: tuple[jax.Array, ...], output_cotangent: jax.Array
        ):
            q_value, k_value, v_value, output, logsumexp, segments = residuals
            if padded_sequence != original_sequence:
                output_cotangent = jnp.pad(
                    output_cotangent,
                    (
                        (0, 0),
                        (0, 0),
                        (0, padded_sequence - original_sequence),
                        (0, 0),
                    ),
                )
            dq, dk, dv = _tpu_flash_backward(
                q_value,
                k_value,
                v_value,
                output,
                logsumexp,
                output_cotangent,
                tiles=tiles,
                softmax_scale=scale,
                valid_sequence=original_sequence,
                interpret=config.interpret,
                debug=config.debug,
                segment_ids=segments,
            )
            # No cotangent for the segment index: it is data, not a parameter.
            return dq, dk, dv, None

        tpu_flash_attention.defvjp(forward_rule, backward_rule)
        return tpu_flash_attention(q_padded, k_padded, v_padded, segments_padded)

    return attention


def causal_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    config: AttentionConfig = AttentionConfig(),
) -> jax.Array:
    """Convenience wrapper around :func:`make_causal_attention`."""

    return make_causal_attention(config)(q, k, v)


__all__ = (
    "AttentionConfig",
    "AttentionTiles",
    "attention_tile_candidates",
    "causal_attention",
    "make_causal_attention",
    "reference_causal_attention",
    "select_attention_tiles",
)
