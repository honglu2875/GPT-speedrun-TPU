"""Reusable JAX kernels shared by rig recipes.

The competition still keeps the model/training algorithm in one entry script;
these kernels are deliberately limited to fundamental, shape-generic building
blocks whose behavior can be validated independently.
"""

from .linear_cross_entropy import (
    DEFAULT_VOCAB_TILE_SIZE,
    tiled_tied_cross_entropy,
    tiled_tied_cross_entropy_losses,
)
from .tpu_flash_attention import (
    AttentionConfig,
    AttentionTiles,
    attention_tile_candidates,
    causal_attention,
    make_causal_attention,
    reference_causal_attention,
    select_attention_tiles,
)

__all__ = (
    "DEFAULT_VOCAB_TILE_SIZE",
    "AttentionConfig",
    "AttentionTiles",
    "attention_tile_candidates",
    "causal_attention",
    "make_causal_attention",
    "reference_causal_attention",
    "select_attention_tiles",
    "tiled_tied_cross_entropy",
    "tiled_tied_cross_entropy_losses",
)
