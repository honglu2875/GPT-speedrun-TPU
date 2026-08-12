from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import warnings

from speedrun.kernels.autotune import (
    AttentionTilePlan,
    AutotuneCacheError,
    AutotuneKey,
    AutotuneRecord,
    CandidateMeasurement,
    JAX_FLASH_REVISION,
    JAX_FLASH_SOURCE_SHA256,
    KERNEL_REVISION,
    NoSuccessfulCandidateError,
    TPU_FLASH_SOURCE_SHA256,
    benchmark_tile_candidates,
    estimate_attention_vmem_bytes,
    find_cached_tuning,
    generate_attention_tile_candidates,
    heuristic_attention_tile_plan,
    is_legal_attention_tile_plan,
    kernel_implementation_hash,
    load_autotune_cache,
    lookup_shipped_tuning,
    resolve_attention_tile_plan,
    save_autotune_record,
)


def make_key(**overrides: object) -> AutotuneKey:
    values: dict[str, object] = {
        "kernel_revision": JAX_FLASH_REVISION,
        "implementation_hash": JAX_FLASH_SOURCE_SHA256,
        "backend": "jax_flash",
        "platform": "tpu",
        "device_kind": "TPU v4",
        "device_count": 4,
        "local_device_count": 4,
        "jax_version": "0.11.0",
        "jaxlib_version": "0.11.0",
        "libtpu_version": "0.0.44.1",
        "dtype": "bfloat16",
        "batch": 8,
        "heads": 12,
        "sequence": 1024,
        "head_dim": 64,
        "mode": "forward_backward",
        "causal": True,
        "backward_strategy": "separate",
        "q_layout": "head_dim_minor",
        "k_layout": "head_dim_minor",
        "v_layout": "head_dim_minor",
        "buffer_count": 2,
        "lookahead": 1,
        "exponential": "native",
        "conditional_rescale": False,
    }
    values.update(overrides)
    return AutotuneKey(**values)  # type: ignore[arg-type]


def measurement(tiles: AttentionTilePlan, seconds: float) -> CandidateMeasurement:
    return CandidateMeasurement(
        tiles=tiles,
        status="ok",
        compile_seconds=1.0,
        samples_seconds=(seconds,) * 3,
        median_seconds=seconds,
        mad_seconds=0.0,
    )


class AttentionTilePolicyTests(unittest.TestCase):
    def test_canonical_candidates_are_bounded_legal_and_deterministic(self) -> None:
        first = generate_attention_tile_candidates(
            sequence=1024, head_dim=64, mode="forward_backward"
        )
        second = generate_attention_tile_candidates(
            sequence=1024, head_dim=64, mode="forward_backward"
        )
        self.assertEqual(first, second)
        self.assertGreaterEqual(len(first), 5)
        self.assertLessEqual(len(first), 9)
        self.assertIn(
            AttentionTilePlan(512, 512, 256, 512, 256, 512, 256, 256, 512, 256),
            first,
        )
        self.assertTrue(
            all(
                is_legal_attention_tile_plan(
                    tiles,
                    sequence=1024,
                    head_dim=64,
                    mode="forward_backward",
                )
                for tiles in first
            )
        )

    def test_odd_sequence_uses_padded_divisibility(self) -> None:
        candidates = generate_attention_tile_candidates(
            sequence=769, head_dim=64, mode="forward"
        )
        self.assertTrue(candidates)
        self.assertTrue(all(896 % candidate.block_q == 0 for candidate in candidates))

    def test_legality_rejects_bad_alignment_head_and_vmem(self) -> None:
        good = AttentionTilePlan(128, 128, 128)
        bad_alignment = AttentionTilePlan(192, 128, 128)
        self.assertTrue(
            is_legal_attention_tile_plan(
                good, sequence=128, head_dim=64, mode="forward"
            )
        )
        self.assertFalse(
            is_legal_attention_tile_plan(
                bad_alignment, sequence=384, head_dim=64, mode="forward"
            )
        )
        self.assertFalse(
            is_legal_attention_tile_plan(
                good, sequence=128, head_dim=65, mode="forward"
            )
        )
        self.assertFalse(
            is_legal_attention_tile_plan(
                good,
                sequence=128,
                head_dim=64,
                mode="forward",
                vmem_bytes=1024,
            )
        )
        self.assertGreater(
            estimate_attention_vmem_bytes(
                good, head_dim=64, mode="forward", buffer_count=2
            ),
            1024,
        )

    def test_exact_shipped_lut_and_runtime_invalidation(self) -> None:
        self.assertEqual(
            kernel_implementation_hash(backend="tpu_flash"),
            TPU_FLASH_SOURCE_SHA256,
        )
        self.assertEqual(
            kernel_implementation_hash(backend="jax_flash"),
            JAX_FLASH_SOURCE_SHA256,
        )
        expected = AttentionTilePlan(
            512, 512, 256, 512, 256, 512, 256, 256, 512, 256
        )
        self.assertEqual(lookup_shipped_tuning(make_key()), expected)
        self.assertIsNone(lookup_shipped_tuning(make_key(jax_version="0.11.1")))
        self.assertIsNone(
            lookup_shipped_tuning(make_key(implementation_hash="new-kernel"))
        )
        tpu_flash_key = make_key(
            kernel_revision=KERNEL_REVISION,
            implementation_hash=TPU_FLASH_SOURCE_SHA256,
            backend="tpu_flash",
        )
        self.assertEqual(lookup_shipped_tuning(tpu_flash_key), expected)
        self.assertIsNone(
            lookup_shipped_tuning(
                make_key(
                    kernel_revision=KERNEL_REVISION,
                    implementation_hash="stale-custom-source",
                    backend="tpu_flash",
                )
            )
        )
        self.assertEqual(heuristic_attention_tile_plan(make_key()), expected)


class AutotuneCacheTests(unittest.TestCase):
    def test_key_digest_is_order_independent_and_sensitive(self) -> None:
        key = make_key()
        reordered = AutotuneKey.from_dict(
            dict(reversed(list(key.to_dict().items())))
        )
        self.assertEqual(key.digest, reordered.digest)
        self.assertNotEqual(key.digest, make_key(sequence=512).digest)
        self.assertNotEqual(
            key.digest, make_key(conditional_rescale=True).digest
        )

    def test_round_trip_merges_records(self) -> None:
        tiles = AttentionTilePlan(
            128, 128, 128, 128, 128, 128, 128, 128, 128, 128
        )
        key_one = make_key(sequence=128)
        key_two = make_key(sequence=256)
        record_one = AutotuneRecord(
            key_one, tiles, (measurement(tiles, 0.01),), "2026-08-12T00:00:00+00:00"
        )
        record_two = AutotuneRecord(
            key_two, tiles, (measurement(tiles, 0.02),), "2026-08-12T00:00:01+00:00"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attention.json"
            self.assertEqual(load_autotune_cache(path), {})
            save_autotune_record(path, record_one)
            save_autotune_record(path, record_two)
            self.assertEqual(find_cached_tuning(path, key_one), record_one)
            self.assertEqual(find_cached_tuning(path, key_two), record_two)
            document = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(document["schema_version"], 1)
            self.assertEqual(len(document["entries"]), 2)

    def test_digest_tampering_is_rejected(self) -> None:
        tiles = AttentionTilePlan(
            128, 128, 128, 128, 128, 128, 128, 128, 128, 128
        )
        record = AutotuneRecord(
            make_key(sequence=128),
            tiles,
            (measurement(tiles, 0.01),),
            "2026-08-12T00:00:00+00:00",
        )
        document = {
            "schema_version": 1,
            "entries": {"not-the-digest": record.to_dict()},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attention.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(AutotuneCacheError, "digest mismatch"):
                load_autotune_cache(path)

    def test_cache_and_lock_symlinks_are_refused_without_touching_target(self) -> None:
        tiles = AttentionTilePlan(
            128, 128, 128, 128, 128, 128, 128, 128, 128, 128
        )
        record = AutotuneRecord(
            make_key(sequence=128),
            tiles,
            (measurement(tiles, 0.01),),
            "2026-08-12T00:00:00+00:00",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            victim = root / "victim"
            victim.write_text("do-not-touch", encoding="utf-8")
            cache_link = root / "cache.json"
            cache_link.symlink_to(victim)
            with self.assertRaisesRegex(AutotuneCacheError, "symlink"):
                load_autotune_cache(cache_link)
            with self.assertRaisesRegex(AutotuneCacheError, "symlink"):
                save_autotune_record(cache_link, record)
            self.assertEqual(victim.read_text(encoding="utf-8"), "do-not-touch")

            cache_link.unlink()
            lock_link = root / "cache.json.lock"
            lock_link.unlink(missing_ok=True)
            lock_link.symlink_to(victim)
            with self.assertRaisesRegex(AutotuneCacheError, "safely open"):
                save_autotune_record(cache_link, record)
            self.assertEqual(victim.read_text(encoding="utf-8"), "do-not-touch")

    def test_resolve_priority_cache_then_shipped_then_heuristic(self) -> None:
        key = make_key()
        cached_tiles = AttentionTilePlan(
            256, 512, 128, 256, 128, 512, 128, 128, 512, 128
        )
        record = AutotuneRecord(
            key,
            cached_tiles,
            (measurement(cached_tiles, 0.01),),
            "2026-08-12T00:00:00+00:00",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attention.json"
            save_autotune_record(path, record)
            resolved = resolve_attention_tile_plan(key, cache_path=path)
            self.assertEqual(resolved.source, "cache")
            self.assertEqual(resolved.tiles, cached_tiles)
            path.write_text("broken", encoding="utf-8")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                resolved = resolve_attention_tile_plan(key, cache_path=path)
            self.assertEqual(resolved.source, "shipped")
            self.assertTrue(caught)
            with self.assertRaises(AutotuneCacheError):
                resolve_attention_tile_plan(key, cache_path=path, strict_cache=True)


class BenchmarkHarnessTests(unittest.TestCase):
    def test_compiles_once_warms_synchronizes_and_uses_median(self) -> None:
        slow = AttentionTilePlan(128, 128, 128)
        fast = AttentionTilePlan(256, 256, 128)
        compile_calls: list[AttentionTilePlan] = []
        run_counts = {slow: 0, fast: 0}
        duration_samples = {
            slow: iter((5.0, 1.0, 5.0)),
            fast: iter((2.0, 2.0, 2.0)),
        }
        active: list[AttentionTilePlan | None] = [None]
        now = [0.0]

        def compile_candidate(tiles: AttentionTilePlan):
            compile_calls.append(tiles)

            def run() -> AttentionTilePlan:
                run_counts[tiles] += 1
                active[0] = tiles
                return tiles

            return run

        sync_calls: list[AttentionTilePlan] = []

        def synchronize(value: AttentionTilePlan) -> None:
            sync_calls.append(value)
            # Warmup calls happen before the three measured duration samples.
            if run_counts[value] > 1:
                now[0] += next(duration_samples[value])

        def clock() -> float:
            return now[0]

        winner, results = benchmark_tile_candidates(
            (slow, fast, slow),
            compile_candidate,
            warmup_runs=1,
            measured_runs=3,
            synchronize=synchronize,
            clock=clock,
        )
        self.assertEqual(winner, fast)
        self.assertEqual(compile_calls, [slow, fast])
        self.assertEqual(run_counts, {slow: 4, fast: 4})
        self.assertEqual(len(sync_calls), 8)
        self.assertEqual(results[0].median_seconds, 5.0)
        self.assertEqual(results[1].median_seconds, 2.0)

    def test_near_tie_prefers_smaller_plan_independent_of_input_order(self) -> None:
        smaller = AttentionTilePlan(128, 128, 128)
        larger = AttentionTilePlan(256, 256, 128)

        def run_once(order: tuple[AttentionTilePlan, ...]) -> AttentionTilePlan:
            counts = {smaller: 0, larger: 0}
            now = [0.0]

            def compile_candidate(tiles: AttentionTilePlan):
                def run() -> AttentionTilePlan:
                    counts[tiles] += 1
                    return tiles

                return run

            def synchronize(value: AttentionTilePlan) -> None:
                now[0] += 1.005 if value == smaller else 1.0

            winner, _ = benchmark_tile_candidates(
                order,
                compile_candidate,
                warmup_runs=0,
                measured_runs=3,
                synchronize=synchronize,
                clock=lambda: now[0],
                head_dim=64,
            )
            return winner

        self.assertEqual(run_once((larger, smaller)), smaller)
        self.assertEqual(run_once((smaller, larger)), smaller)

    def test_candidate_failure_is_recorded_and_all_failed_raises(self) -> None:
        bad = AttentionTilePlan(128, 128, 128)

        def fail(_: AttentionTilePlan):
            raise RuntimeError("compiler detail\nwithout secrets")

        with self.assertRaisesRegex(
            NoSuccessfulCandidateError, "RuntimeError"
        ) as caught:
            benchmark_tile_candidates(
                (bad,), fail, warmup_runs=0, measured_runs=1
            )
        self.assertEqual(len(caught.exception.measurements), 1)
        self.assertEqual(caught.exception.measurements[0].status, "error")


if __name__ == "__main__":
    unittest.main()
