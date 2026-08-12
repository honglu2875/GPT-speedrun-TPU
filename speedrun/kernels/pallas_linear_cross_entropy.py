"""Experimental TPU Pallas fused tied projection + cross entropy.

The production fallback in :mod:`speedrun.kernels.linear_cross_entropy` asks
XLA to tile a vocabulary loop.  This module gives the forward pass explicit
TPU-local ownership: one program owns a tile of token rows and streams
vocabulary blocks through VMEM while retaining only FP32 online-softmax state.

The implementation is intentionally kept out of the public kernel exports
until it has been shown to improve a complete training step. It uses two
different backward ownership patterns (token-owned ``d_hidden`` and
vocabulary-owned ``d_embedding``), because TPU Pallas has no global-memory
atomic add.

The exact TPU v4 per-chip GPT-2-small microbenchmark (8,192 tokens, width 768,
50,304 storage rows, 50,257 semantic rows, BF16 MXU operands) measured 14.63 ms
for value + both gradients, versus 12.16 ms for the production pure-JAX tiled
custom VJP and 22.59 ms for the dense oracle. The kernel is therefore a useful
correctness-checked research baseline, not a production selection.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


TPU_LANES = 128


def _forward_kernel(
    hidden_ref,
    embedding_ref,
    log_normalizer_ref,
    maximum_scratch_ref,
    exponential_sum_scratch_ref,
    *,
    semantic_vocab_size: int,
    padded_vocab_size: int,
    vocab_major_size: int,
    vocab_compute_size: int,
) -> None:
    """Stream one vocabulary-major block for a token-owned program."""

    vocab_major_index = pl.program_id(1)

    @pl.when(vocab_major_index == 0)
    def initialize() -> None:
        maximum_scratch_ref[...] = jnp.full(
            maximum_scratch_ref.shape, -jnp.inf, jnp.float32
        )
        exponential_sum_scratch_ref[...] = jnp.zeros(
            exponential_sum_scratch_ref.shape, jnp.float32
        )

    hidden = hidden_ref[...]
    token_width = hidden_ref.shape[0]
    repeats = vocab_compute_size // TPU_LANES

    @pl.loop(0, vocab_major_size, step=vocab_compute_size, unroll=True)
    def compute(vocab_offset: int) -> None:
        weights = embedding_ref[
            pl.dslice(vocab_offset, vocab_compute_size), :
        ]
        logits = lax.dot_general(
            hidden,
            weights,
            (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )

        vocab_ids = lax.broadcasted_iota(
            jnp.int32, (token_width, vocab_compute_size), 1
        )
        vocab_ids += vocab_major_index * vocab_major_size + vocab_offset
        valid_vocab = vocab_ids < semantic_vocab_size
        logits = jnp.where(valid_vocab, logits, -jnp.inf)

        previous_maximum = maximum_scratch_ref[...]
        previous_sum = exponential_sum_scratch_ref[...]
        block_maximum = jnp.max(logits, axis=1)[:, None]
        block_maximum = jnp.tile(block_maximum, (1, TPU_LANES))
        next_maximum = jnp.maximum(previous_maximum, block_maximum)
        maximum_repeated = jnp.tile(next_maximum, (1, repeats))
        exponentials = jnp.exp(logits - maximum_repeated)
        block_sum = jnp.sum(exponentials, axis=1)[:, None]
        block_sum = jnp.tile(block_sum, (1, TPU_LANES))
        next_sum = previous_sum * jnp.exp(previous_maximum - next_maximum)
        next_sum += block_sum
        maximum_scratch_ref[...] = next_maximum
        exponential_sum_scratch_ref[...] = next_sum

    last_major_index = padded_vocab_size // vocab_major_size - 1

    @pl.when(vocab_major_index == last_major_index)
    def store() -> None:
        log_normalizer = maximum_scratch_ref[...] + jnp.log(
            exponential_sum_scratch_ref[...]
        )
        log_normalizer_ref[...] = log_normalizer


@partial(
    jax.jit,
    static_argnames=(
        "semantic_vocab_size",
        "token_tile_size",
        "vocab_major_size",
        "vocab_compute_size",
        "compute_dtype",
        "interpret",
        "debug",
    ),
)
def pallas_tied_cross_entropy_forward(
    hidden: jax.Array,
    embedding: jax.Array,
    targets: jax.Array,
    *,
    semantic_vocab_size: int,
    token_tile_size: int = 128,
    vocab_major_size: int = 2_048,
    vocab_compute_size: int = 256,
    compute_dtype: Any = jnp.bfloat16,
    interpret: bool = False,
    debug: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Return per-token losses and log normalizers without dense logits.

    All dimensions and tile sizes are static at lowering time. Storage padding
    and conversion to the MXU operand dtype occur outside the Pallas call; all
    logits and softmax reductions inside the kernel are FP32.
    """

    if hidden.ndim < 2:
        raise ValueError("hidden must have shape [..., width]")
    if embedding.ndim != 2 or hidden.shape[-1] != embedding.shape[-1]:
        raise ValueError("embedding must have shape [storage_vocab, hidden_width]")
    if targets.shape != hidden.shape[:-1]:
        raise ValueError("targets must match hidden's leading dimensions")
    if not jnp.issubdtype(targets.dtype, jnp.integer):
        raise TypeError("targets must have integer dtype")
    if not 0 < semantic_vocab_size <= embedding.shape[0]:
        raise ValueError("semantic_vocab_size must fit inside embedding storage")
    for name, value in (
        ("token_tile_size", token_tile_size),
        ("vocab_major_size", vocab_major_size),
        ("vocab_compute_size", vocab_compute_size),
    ):
        if value <= 0 or value % TPU_LANES:
            raise ValueError(f"{name} must be a positive multiple of 128")
    if vocab_major_size % vocab_compute_size:
        raise ValueError("vocab_compute_size must divide vocab_major_size")

    compute_dtype = jnp.dtype(compute_dtype)
    if compute_dtype not in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float32)):
        raise TypeError("compute_dtype must be bfloat16 or float32")

    original_leading_shape = targets.shape
    flat_hidden = hidden.reshape((-1, hidden.shape[-1]))
    flat_targets = targets.reshape((-1,))
    token_count = flat_hidden.shape[0]
    if token_count % token_tile_size:
        raise ValueError("flattened token count must be divisible by token_tile_size")
    if hidden.shape[-1] % TPU_LANES:
        raise ValueError("hidden width must be divisible by 128")

    storage_vocab = embedding.shape[0]
    padded_vocab = (
        (storage_vocab + vocab_major_size - 1) // vocab_major_size
    ) * vocab_major_size
    if padded_vocab != storage_vocab:
        embedding = jnp.pad(
            embedding,
            ((0, padded_vocab - storage_vocab), (0, 0)),
        )
    embedding = embedding.astype(compute_dtype)
    flat_hidden = flat_hidden.astype(compute_dtype)

    # Mosaic TPU vector values have 128 lanes. Broadcasting scalar outputs
    # avoids unsupported one-dimensional VMEM tiles; callers see only lane zero.
    lane_shape = jax.ShapeDtypeStruct(
        (token_count, TPU_LANES), dtype=jnp.float32
    )

    def hidden_index(token_index, _):
        return (token_index, 0)

    def embedding_index(_, vocab_index):
        return (vocab_index, 0)

    def token_lanes_index(token_index, _):
        return (token_index, 0)

    kernel = partial(
        _forward_kernel,
        semantic_vocab_size=semantic_vocab_size,
        padded_vocab_size=padded_vocab,
        vocab_major_size=vocab_major_size,
        vocab_compute_size=vocab_compute_size,
    )
    log_normalizer_lanes = pl.pallas_call(
        kernel,
        out_shape=lane_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(token_count // token_tile_size, padded_vocab // vocab_major_size),
            in_specs=(
                pl.BlockSpec(
                    (token_tile_size, hidden.shape[-1]), hidden_index
                ),
                pl.BlockSpec(
                    (vocab_major_size, hidden.shape[-1]), embedding_index
                ),
            ),
            out_specs=pl.BlockSpec(
                (token_tile_size, TPU_LANES), token_lanes_index
            ),
            scratch_shapes=(
                pltpu.VMEM((token_tile_size, TPU_LANES), jnp.float32),
                pltpu.VMEM((token_tile_size, TPU_LANES), jnp.float32),
            ),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary")
        ),
        interpret=interpret,
        debug=debug,
        name="tpu_linear_cross_entropy_fwd",
    )(flat_hidden, embedding)
    log_normalizer = log_normalizer_lanes[:, 0].reshape(original_leading_shape)
    # Target selection is O(ND), rather than O(NV), and lets the Pallas kernel
    # avoid carrying a lane-broadcast target array through every vocabulary
    # tile.  It normally lowers to a gather plus one vector reduction.
    safe_targets = jnp.clip(flat_targets, 0, semantic_vocab_size - 1)
    target_weights = embedding[safe_targets].astype(compute_dtype)
    target_logits = jnp.einsum(
        "nd,nd->n",
        flat_hidden,
        target_weights,
        preferred_element_type=jnp.float32,
    ).reshape(original_leading_shape)
    valid_target = (targets >= 0) & (targets < semantic_vocab_size)
    losses = jnp.where(valid_target, log_normalizer - target_logits, jnp.inf)
    return losses, log_normalizer


def _dhidden_kernel(
    hidden_ref,
    embedding_ref,
    log_normalizer_ref,
    loss_cotangent_ref,
    grad_hidden_ref,
    grad_hidden_scratch_ref,
    *,
    semantic_vocab_size: int,
    padded_vocab_size: int,
    vocab_major_size: int,
    vocab_compute_size: int,
) -> None:
    """Vocabulary-streaming, token-owned hidden-gradient kernel."""

    vocab_major_index = pl.program_id(1)

    @pl.when(vocab_major_index == 0)
    def initialize() -> None:
        grad_hidden_scratch_ref[...] = jnp.zeros(
            grad_hidden_scratch_ref.shape, jnp.float32
        )

    hidden = hidden_ref[...]
    log_normalizer = log_normalizer_ref[...]
    cotangent = loss_cotangent_ref[...]
    token_width = hidden.shape[0]
    repeats = vocab_compute_size // TPU_LANES

    @pl.loop(0, vocab_major_size, step=vocab_compute_size, unroll=True)
    def compute(vocab_offset: int) -> None:
        weights = embedding_ref[
            pl.dslice(vocab_offset, vocab_compute_size), :
        ]
        logits = lax.dot_general(
            hidden,
            weights,
            (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        vocab_ids = lax.broadcasted_iota(
            jnp.int32, (token_width, vocab_compute_size), 1
        )
        vocab_ids += vocab_major_index * vocab_major_size + vocab_offset
        probabilities = jnp.where(
            vocab_ids < semantic_vocab_size,
            jnp.exp(logits - jnp.tile(log_normalizer, (1, repeats))),
            0.0,
        )
        dlogits = probabilities * jnp.tile(cotangent, (1, repeats))
        update = lax.dot(
            dlogits.astype(hidden.dtype),
            weights,
            preferred_element_type=jnp.float32,
        )
        grad_hidden_scratch_ref[...] += update

    last_major_index = padded_vocab_size // vocab_major_size - 1

    @pl.when(vocab_major_index == last_major_index)
    def store() -> None:
        grad_hidden_ref[...] = grad_hidden_scratch_ref[...].astype(
            grad_hidden_ref.dtype
        )


def _dembedding_kernel(
    hidden_ref,
    embedding_ref,
    log_normalizer_ref,
    loss_cotangent_ref,
    grad_embedding_ref,
    grad_embedding_scratch_ref,
    *,
    semantic_vocab_size: int,
    token_count: int,
    token_major_size: int,
    token_compute_size: int,
) -> None:
    """Token-streaming, vocabulary-owned embedding-gradient kernel."""

    token_major_index = pl.program_id(1)

    @pl.when(token_major_index == 0)
    def initialize() -> None:
        grad_embedding_scratch_ref[...] = jnp.zeros(
            grad_embedding_scratch_ref.shape, jnp.float32
        )

    weights = embedding_ref[...]
    vocabulary_width = weights.shape[0]
    vocab_base = pl.program_id(0) * vocabulary_width
    vocab_ids = lax.broadcasted_iota(
        jnp.int32, (token_compute_size, vocabulary_width), 1
    )
    vocab_ids += vocab_base
    @pl.loop(0, token_major_size, step=token_compute_size, unroll=True)
    def compute(token_offset: int) -> None:
        hidden = hidden_ref[
            pl.dslice(token_offset, token_compute_size), :
        ]
        logits = lax.dot_general(
            hidden,
            weights,
            (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        log_normalizer = log_normalizer_ref[
            pl.dslice(token_offset, token_compute_size), :
        ]
        normalizer_repeats = vocabulary_width // TPU_LANES
        probabilities = jnp.where(
            vocab_ids < semantic_vocab_size,
            jnp.exp(logits - jnp.tile(log_normalizer, (1, normalizer_repeats))),
            0.0,
        )
        cotangent = loss_cotangent_ref[
            pl.dslice(token_offset, token_compute_size), :
        ]
        dlogits = probabilities * jnp.tile(
            cotangent, (1, vocabulary_width // TPU_LANES)
        )
        update = lax.dot(
            dlogits.T.astype(hidden.dtype),
            hidden,
            preferred_element_type=jnp.float32,
        )
        grad_embedding_scratch_ref[...] += update

    last_major_index = token_count // token_major_size - 1

    @pl.when(token_major_index == last_major_index)
    def store() -> None:
        grad_embedding_ref[...] = grad_embedding_scratch_ref[...].astype(
            grad_embedding_ref.dtype
        )


@partial(
    jax.jit,
    static_argnames=(
        "semantic_vocab_size",
        "token_tile_size",
        "vocab_major_size",
        "vocab_compute_size",
        "weight_tile_size",
        "token_major_size",
        "token_compute_size",
        "compute_dtype",
        "interpret",
        "debug",
    ),
)
def pallas_tied_cross_entropy_backward(
    hidden: jax.Array,
    embedding: jax.Array,
    targets: jax.Array,
    log_normalizer: jax.Array,
    loss_cotangent: jax.Array,
    *,
    semantic_vocab_size: int,
    token_tile_size: int = 128,
    vocab_major_size: int = 2_048,
    vocab_compute_size: int = 256,
    weight_tile_size: int = 256,
    token_major_size: int = 2_048,
    token_compute_size: int = 256,
    compute_dtype: Any = jnp.bfloat16,
    interpret: bool = False,
    debug: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Return explicit gradients using token- and vocabulary-owned kernels."""

    if (
        log_normalizer.shape != targets.shape
        or loss_cotangent.shape != targets.shape
    ):
        raise ValueError("normalizer and cotangent must match targets")
    flat_hidden = hidden.reshape((-1, hidden.shape[-1])).astype(compute_dtype)
    flat_targets = targets.reshape((-1,))
    flat_normalizer = log_normalizer.reshape((-1,)).astype(jnp.float32)
    flat_cotangent = loss_cotangent.reshape((-1,)).astype(jnp.float32)
    valid_target = (flat_targets >= 0) & (flat_targets < semantic_vocab_size)
    # Invalid targets intentionally produce infinite loss. They have no class
    # contribution, so suppress both the dense denominator and sparse target
    # gradient paths before broadcasting cotangents into the Pallas kernels.
    flat_cotangent = jnp.where(valid_target, flat_cotangent, 0.0)
    token_count = flat_hidden.shape[0]
    hidden_width = flat_hidden.shape[1]
    storage_vocab = embedding.shape[0]

    if token_count % token_major_size or token_major_size % token_compute_size:
        raise ValueError(
            "token_compute_size must divide token_major_size and token count"
        )
    if token_count % token_tile_size:
        raise ValueError("token_tile_size must divide token count")
    if hidden_width % TPU_LANES:
        raise ValueError("hidden width must be divisible by 128")
    for name, value in (
        ("token_tile_size", token_tile_size),
        ("vocab_major_size", vocab_major_size),
        ("vocab_compute_size", vocab_compute_size),
        ("weight_tile_size", weight_tile_size),
        ("token_major_size", token_major_size),
        ("token_compute_size", token_compute_size),
    ):
        if value <= 0 or value % TPU_LANES:
            raise ValueError(f"{name} must be a positive multiple of 128")
    if vocab_major_size % vocab_compute_size:
        raise ValueError("vocab_compute_size must divide vocab_major_size")

    padded_vocab = (
        (storage_vocab + vocab_major_size - 1) // vocab_major_size
    ) * vocab_major_size
    padded_vocab = (
        (padded_vocab + weight_tile_size - 1) // weight_tile_size
    ) * weight_tile_size
    padded_embedding = jnp.pad(
        embedding,
        ((0, padded_vocab - storage_vocab), (0, 0)),
    ).astype(compute_dtype)
    normalizer_lanes = jnp.broadcast_to(
        flat_normalizer[:, None], (token_count, TPU_LANES)
    )
    cotangent_lanes = jnp.broadcast_to(
        flat_cotangent[:, None], (token_count, TPU_LANES)
    )

    def token_hidden_index(token_index, _):
        return (token_index, 0)

    def vocab_major_index(_, vocab_index):
        return (vocab_index, 0)

    def token_lanes_index(token_index, _):
        return (token_index, 0)

    dh_kernel = partial(
        _dhidden_kernel,
        semantic_vocab_size=semantic_vocab_size,
        padded_vocab_size=padded_vocab,
        vocab_major_size=vocab_major_size,
        vocab_compute_size=vocab_compute_size,
    )
    grad_hidden = pl.pallas_call(
        dh_kernel,
        out_shape=jax.ShapeDtypeStruct(flat_hidden.shape, flat_hidden.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(token_count // token_tile_size, padded_vocab // vocab_major_size),
            in_specs=(
                pl.BlockSpec((token_tile_size, hidden_width), token_hidden_index),
                pl.BlockSpec((vocab_major_size, hidden_width), vocab_major_index),
                pl.BlockSpec((token_tile_size, TPU_LANES), token_lanes_index),
                pl.BlockSpec((token_tile_size, TPU_LANES), token_lanes_index),
            ),
            out_specs=pl.BlockSpec(
                (token_tile_size, hidden_width), token_hidden_index
            ),
            scratch_shapes=(
                pltpu.VMEM((token_tile_size, hidden_width), jnp.float32),
            ),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary")
        ),
        interpret=interpret,
        debug=debug,
        name="tpu_linear_cross_entropy_dhidden",
    )(
        flat_hidden,
        padded_embedding,
        normalizer_lanes,
        cotangent_lanes,
    )

    def all_tokens_index(_, token_index):
        return (token_index, 0)

    def weight_index(vocab_index, _):
        return (vocab_index, 0)

    de_kernel = partial(
        _dembedding_kernel,
        semantic_vocab_size=semantic_vocab_size,
        token_count=token_count,
        token_major_size=token_major_size,
        token_compute_size=token_compute_size,
    )
    grad_embedding = pl.pallas_call(
        de_kernel,
        # Parameter gradients remain FP32 even though the MXU operands are
        # BF16.  This matches the master-parameter/optimizer representation.
        out_shape=jax.ShapeDtypeStruct(padded_embedding.shape, embedding.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(padded_vocab // weight_tile_size, token_count // token_major_size),
            in_specs=(
                pl.BlockSpec((token_major_size, hidden_width), all_tokens_index),
                pl.BlockSpec((weight_tile_size, hidden_width), weight_index),
                pl.BlockSpec((token_major_size, TPU_LANES), all_tokens_index),
                pl.BlockSpec((token_major_size, TPU_LANES), all_tokens_index),
            ),
            out_specs=pl.BlockSpec(
                (weight_tile_size, hidden_width), weight_index
            ),
            scratch_shapes=(
                pltpu.VMEM((weight_tile_size, hidden_width), jnp.float32),
            ),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary")
        ),
        interpret=interpret,
        debug=debug,
        name="tpu_linear_cross_entropy_dembedding",
    )(
        flat_hidden,
        padded_embedding,
        normalizer_lanes,
        cotangent_lanes,
    )

    # Apply the sparse ``-one_hot(target)`` term outside Pallas.  This avoids
    # broadcasting target ids to 128 lanes in both large streaming kernels.
    safe_targets = jnp.clip(flat_targets, 0, semantic_vocab_size - 1)
    valid_cotangent = flat_cotangent
    target_hidden_update = (
        valid_cotangent[:, None]
        * embedding[safe_targets].astype(jnp.float32)
    )
    grad_hidden = grad_hidden.astype(jnp.float32) - target_hidden_update
    target_embedding_update = (
        -valid_cotangent[:, None] * flat_hidden.astype(jnp.float32)
    )
    grad_embedding = grad_embedding.at[safe_targets].add(target_embedding_update)

    return (
        grad_hidden.reshape(hidden.shape).astype(hidden.dtype),
        grad_embedding[:storage_vocab].astype(embedding.dtype),
    )
