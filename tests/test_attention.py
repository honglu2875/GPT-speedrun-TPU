"""Contracts for recipe-independent attention runtime plumbing."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np

from rig import attention
from rig.kernels import AttentionTiles


def _tiles() -> AttentionTiles:
    return AttentionTiles(512, 512, 256, 512, 256, 512, 256, 256, 512, 256)


@dataclass(frozen=True)
class FakeDevice:
    platform: str = "tpu"
    device_kind: str = "TPU v4"
    process_index: int = 0


class AttentionRuntimeTests(unittest.TestCase):
    def test_softmax_scale_keeps_the_recipe_policy_explicit(self) -> None:
        self.assertEqual(
            attention.attention_softmax_scale("inverse_head_dim", 64), 1.0 / 64.0
        )
        self.assertEqual(
            attention.attention_softmax_scale("inverse_sqrt_head_dim", 64), 1.0 / 8.0
        )
        with self.assertRaisesRegex(ValueError, "unsupported attention scale"):
            attention.attention_softmax_scale("hidden_policy", 64)

    def test_dense_runtime_has_no_tile_provenance(self) -> None:
        with patch.object(attention, "make_runtime_key") as make_key:
            runtime = attention.resolve_attention_runtime(
                backend="dense",
                dtype=jnp.float32,
                global_batch_size=1,
                heads=1,
                sequence=8,
                head_dim=8,
                devices=(),
            )
        make_key.assert_not_called()
        self.assertEqual(
            attention.attention_runtime_metadata(runtime),
            {
                "key_digest": None,
                "resolution_source": "dense",
                "tune_seconds": 0.0,
                "tiles": None,
            },
        )
        self.assertEqual(
            attention.attention_console_rows(runtime),
            (
                ("attention tuning", "not applicable (dense)"),
                ("attention plan", "not applicable"),
            ),
        )

    def test_flash_runtime_uses_the_exact_local_shape_without_benchmarking(
        self,
    ) -> None:
        devices = tuple(FakeDevice() for _ in range(4))
        tiles = _tiles()
        key = SimpleNamespace(digest="a" * 64)
        resolved = SimpleNamespace(source="shipped", tiles=tiles)
        with (
            patch.object(attention, "make_runtime_key", return_value=key) as make_key,
            patch.object(
                attention, "resolve_attention_tile_plan", return_value=resolved
            ) as resolve,
        ):
            runtime = attention.resolve_attention_runtime(
                backend="tpu_flash",
                dtype=jnp.bfloat16,
                global_batch_size=128,
                heads=10,
                sequence=1_024,
                head_dim=64,
                devices=devices,
            )

        self.assertEqual(
            make_key.call_args.kwargs,
            {
                "backend": "tpu_flash",
                "dtype": jnp.bfloat16,
                "batch": 32,
                "heads": 10,
                "sequence": 1_024,
                "head_dim": 64,
                "mode": "forward_backward",
                "device": devices[0],
            },
        )
        resolve.assert_called_once_with(key)
        self.assertEqual(
            runtime, attention.AttentionRuntime("a" * 64, "shipped", tiles, 0.0)
        )

    def test_flash_runtime_rejects_an_invalid_device_partition(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one device"):
            attention.resolve_attention_runtime(
                backend="tpu_flash",
                dtype=jnp.bfloat16,
                global_batch_size=128,
                heads=10,
                sequence=1_024,
                head_dim=64,
                devices=(),
            )
        with self.assertRaisesRegex(ValueError, "global batch must divide"):
            attention.resolve_attention_runtime(
                backend="tpu_flash",
                dtype=jnp.bfloat16,
                global_batch_size=127,
                heads=10,
                sequence=1_024,
                head_dim=64,
                devices=tuple(FakeDevice() for _ in range(4)),
            )

    def test_runtime_validation_rejects_incoherent_provenance(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not carry"):
            attention.AttentionRuntime("a" * 64, "dense", _tiles(), 0.0)
        with self.assertRaisesRegex(ValueError, "requires a tuning key"):
            attention.AttentionRuntime(None, "shipped", _tiles(), 0.0)
        with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
            attention.AttentionRuntime("a" * 64, "shipped", _tiles(), np.nan)

    def test_tile_metadata_and_console_rows_are_stable(self) -> None:
        runtime = attention.AttentionRuntime("c" * 64, "cache", _tiles(), 0.0)
        metadata = attention.attention_runtime_metadata(runtime)
        self.assertEqual(metadata["key_digest"], "c" * 64)
        self.assertEqual(metadata["resolution_source"], "cache")
        self.assertEqual(len(metadata["tiles"]), 10)
        self.assertEqual(
            attention.attention_console_rows(runtime),
            (
                ("attention tuning", "cache · key cccccccccccc"),
                ("attention fwd", "q512 · kv512/256"),
                ("attention dK/dV", "q512/256 · kv512/256"),
                ("attention dQ", "q256 · kv512/256"),
            ),
        )


class MeshAttentionTests(unittest.TestCase):
    def test_flash_boundary_uses_the_pre_resolved_plan(self) -> None:
        tiles = _tiles()
        local_attention = object()
        with (
            patch.object(
                attention, "make_causal_attention", return_value=local_attention
            ) as make_attention,
            patch.object(attention.jax, "shard_map", return_value="mapped") as shard,
        ):
            actual = attention.make_mesh_attention(
                backend="tpu_flash",
                mesh=object(),
                tiles=tiles,
                softmax_scale=1.0 / 64.0,
                document_masking=True,
            )

        self.assertEqual(actual, "mapped")
        config = make_attention.call_args.args[0]
        self.assertEqual(config.backend, "tpu_flash")
        self.assertEqual(config.tiles, tiles)
        self.assertEqual(config.softmax_scale, 1.0 / 64.0)
        self.assertIs(shard.call_args.args[0], local_attention)
        self.assertEqual(len(shard.call_args.kwargs["in_specs"]), 4)

    def test_dense_boundary_is_unnecessary_and_flash_requires_tiles(self) -> None:
        self.assertIsNone(
            attention.make_mesh_attention(
                backend="dense",
                mesh=object(),
                tiles=None,
                softmax_scale=1.0,
                document_masking=False,
            )
        )
        with self.assertRaisesRegex(ValueError, "resolved tile plan"):
            attention.make_mesh_attention(
                backend="tpu_flash",
                mesh=object(),
                tiles=None,
                softmax_scale=1.0,
                document_masking=False,
            )

    def test_document_segments_count_boundaries_per_window(self) -> None:
        tokens = jnp.asarray([[9, 1, 2, 9, 3], [4, 9, 5, 9, 9]], jnp.int32)
        np.testing.assert_array_equal(
            attention.document_segments(tokens, 9),
            np.asarray([[1, 1, 1, 2, 2], [0, 1, 1, 2, 3]], np.int32),
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
