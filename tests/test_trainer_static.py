from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
import hashlib
import importlib.util
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


TRAINER_PATH = Path(__file__).parents[1] / "submissions" / "reference" / "train.py"
SPEC = importlib.util.spec_from_file_location("reference_train", TRAINER_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - importlib invariant
    raise RuntimeError(f"could not import {TRAINER_PATH}")
trainer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = trainer
SPEC.loader.exec_module(trainer)


@dataclass(frozen=True)
class FakeDevice:
    platform: str
    device_kind: str


class TrainerStaticTests(unittest.TestCase):
    def test_train_tokens_derives_exact_steps_and_is_exclusive(self) -> None:
        parser = trainer.build_parser()
        config = trainer.resolve_config(
            parser.parse_args(
                [
                    "--profile", "official",
                    "--train-tokens", "655360",
                    "--batch-size", "32",
                    "--seq-len", "1024",
                ]
            ),
            "tpu",
            50_304,
        )
        self.assertEqual(config.steps, 20)
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            trainer.resolve_config(
                parser.parse_args(
                    [
                        "--profile", "official",
                        "--train-tokens", "655361",
                        "--batch-size", "32",
                        "--seq-len", "1024",
                    ]
                ),
                "tpu",
                50_304,
            )
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--steps", "20", "--train-tokens", "655360"])

    def test_explicit_warmup_may_extend_past_short_profile_run(self) -> None:
        parser = trainer.build_parser()
        explicit = trainer.resolve_config(
            parser.parse_args(
                ["--profile", "official", "--steps", "100", "--warmup-steps", "715"]
            ),
            "tpu",
            50_304,
        )
        defaulted = trainer.resolve_config(
            parser.parse_args(["--profile", "official", "--steps", "100"]),
            "tpu",
            50_304,
        )
        self.assertEqual(explicit.warmup_steps, 715)
        self.assertEqual(defaulted.warmup_steps, 100)

    def test_xprof_diagnostic_contract_and_capture_window(self) -> None:
        parser = trainer.build_parser()
        valid = parser.parse_args(
            [
                "--xprof-dir", "trace",
                "--xprof-start-step", "11",
                "--xprof-steps", "20",
                "--no-final-validation",
                "--no-checkpoint",
            ]
        )
        trainer.validate_args(valid)
        self.assertEqual(trainer.xprof_step_window(valid, 100), (11, 30))
        self.assertFalse(
            trainer.should_compile_evaluation(
                valid, SimpleNamespace(val_every=0), ()
            )
        )

        normal = parser.parse_args([])
        trainer.validate_args(normal)
        self.assertIsNone(trainer.xprof_step_window(normal, 100))
        self.assertTrue(
            trainer.should_compile_evaluation(
                normal, SimpleNamespace(val_every=0), ()
            )
        )

        invalid_commands = (
            ["--xprof-start-step", "1"],
            ["--xprof-dir", "trace", "--xprof-start-step", "1"],
            ["--xprof-dir", "trace", "--xprof-start-step", "1", "--xprof-steps", "1", "--no-checkpoint"],
            ["--no-final-validation", "--no-checkpoint"],
            [
                "--profile", "official",
                "--xprof-dir", "trace",
                "--xprof-start-step", "1",
                "--xprof-steps", "1",
                "--no-final-validation",
                "--no-checkpoint",
            ],
        )
        for command in invalid_commands:
            with self.subTest(command=command), self.assertRaises(ValueError):
                trainer.validate_args(parser.parse_args(command))
        with self.assertRaisesRegex(ValueError, "must fit inside"):
            trainer.xprof_step_window(valid, 25)

        options = trainer.profiler_options("tpu", 4)
        self.assertEqual(options.python_tracer_level, 0)
        self.assertEqual(options.host_tracer_level, 2)
        self.assertEqual(
            options.advanced_configuration,
            {
                "tpu_trace_mode": "TRACE_COMPUTE_AND_SYNC",
                "tpu_num_chips_to_profile_per_task": 4,
            },
        )

    def test_diagnostic_main_omits_competition_result(self) -> None:
        stdout = StringIO()
        with (
            patch.object(trainer, "run", return_value=None),
            redirect_stdout(stdout),
        ):
            self.assertEqual(trainer.main([]), 0)
        self.assertEqual(stdout.getvalue(), "")

    def test_periodic_validation_defaults_and_cli_overrides(self) -> None:
        parser = trainer.build_parser()

        official = trainer.resolve_config(
            parser.parse_args(["--profile", "official"]), "tpu", 50_304
        )
        self.assertEqual(official.steps, 19_073)
        self.assertEqual(
            official.steps * official.batch_size * official.seq_len,
            624_984_064,
        )
        self.assertEqual(official.val_every, 250)
        self.assertEqual(official.val_probe_batches, 8)
        self.assertEqual(official.eval_batches, 320)
        self.assertEqual(official.diagnostics_every, 250)

        for profile in ("smoke", "dev"):
            with self.subTest(profile=profile):
                config = trainer.resolve_config(
                    parser.parse_args(["--profile", profile]), "cpu", 256
                )
                self.assertEqual(config.val_every, 0)
                self.assertEqual(config.diagnostics_every, 0)

        overridden = trainer.resolve_config(
            parser.parse_args(
                [
                    "--profile",
                    "dev",
                    "--val-every",
                    "5",
                    "--val-probe-batches",
                    "3",
                ]
            ),
            "cpu",
            256,
        )
        self.assertEqual(overridden.val_every, 5)
        self.assertEqual(overridden.val_probe_batches, 3)
        disabled = trainer.resolve_config(
            parser.parse_args(
                [
                    "--profile",
                    "official",
                    "--steps",
                    "1",
                    "--batch-size",
                    "128",
                    "--seq-len",
                    "16384",
                    "--val-every",
                    "0",
                    "--val-probe-batches",
                    "99",
                ]
            ),
            "tpu",
            50_304,
        )
        self.assertEqual(disabled.val_every, 0)
        self.assertEqual(disabled.eval_batches, 5)
        self.assertEqual(disabled.val_probe_batches, 99)
        custom_official = trainer.resolve_config(
            parser.parse_args(
                [
                    "--profile",
                    "official",
                    "--steps",
                    "1",
                    "--batch-size",
                    "128",
                    "--seq-len",
                    "16384",
                ]
            ),
            "tpu",
            50_304,
        )
        self.assertEqual(custom_official.eval_batches, 5)
        self.assertEqual(custom_official.val_probe_batches, 5)
        with self.assertRaisesRegex(ValueError, "training token budget"):
            trainer.resolve_config(
                parser.parse_args(
                    [
                        "--profile", "official",
                        "--batch-size", "128",
                        "--seq-len", "16384",
                    ]
                ),
                "tpu",
                50_304,
            )
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            trainer.resolve_config(
                parser.parse_args(
                    [
                        "--profile",
                        "dev",
                        "--val-every",
                        "1",
                        "--val-probe-batches",
                        "9",
                    ]
                ),
                "cpu",
                256,
            )

    def test_probe_schedule_excludes_final_step(self) -> None:
        config = SimpleNamespace(val_every=3, steps=9)
        selected = [
            step
            for step in range(1, config.steps + 1)
            if trainer.should_run_validation_probe(step, config)
        ]
        self.assertEqual(selected, [3, 6])
        self.assertFalse(
            trainer.should_run_validation_probe(
                1, SimpleNamespace(val_every=0, steps=10)
            )
        )

    def test_diagnostic_schedule_includes_first_cadence_and_final(self) -> None:
        config = SimpleNamespace(diagnostics_every=3, steps=8)
        selected = [
            step
            for step in range(1, config.steps + 1)
            if trainer.should_run_diagnostics(step, config)
        ]
        self.assertEqual(selected, [1, 3, 6, 8])
        self.assertFalse(
            trainer.should_run_diagnostics(
                1, SimpleNamespace(diagnostics_every=0, steps=1)
            )
        )

    def test_validation_prefix_always_starts_at_batch_zero(self) -> None:
        class Dataset:
            def __init__(self) -> None:
                self.indices: list[int] = []

            def validation_batch(
                self, index: int, batch_size: int, seq_len: int, vocab_size: int
            ) -> tuple[np.ndarray, np.ndarray]:
                del vocab_size
                self.indices.append(index)
                values = np.full((batch_size, seq_len), index, dtype=np.int32)
                return values, values

        dataset = Dataset()
        config = SimpleNamespace(batch_size=2, seq_len=4, vocab_size=16)

        def compiled_eval(
            params: object, x: np.ndarray, y: np.ndarray, mask: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            del params, y
            loss = np.sum((x.astype(np.float32) + 1.0) * mask)
            return np.asarray(loss, dtype=np.float32), np.asarray(mask.sum(), dtype=np.float32)

        with patch.object(
            trainer.jax, "device_put", side_effect=lambda value, _sharding: value
        ):
            loss, elapsed = trainer.evaluate_validation_prefix(
                object(), dataset, compiled_eval, object(), config, 3
            )
        self.assertEqual(dataset.indices, [0, 1, 2])
        self.assertAlmostEqual(loss, 2.0)
        self.assertGreater(elapsed, 0.0)

    def test_training_csv_contains_every_step(self) -> None:
        history = np.asarray(
            [[2.0, 1.0e-3, 0.5], [1.5, 5.0e-4, 0.25]], dtype=np.float32
        )
        config = SimpleNamespace(steps=2, batch_size=4, seq_len=8)
        with tempfile.TemporaryDirectory() as directory:
            trainer.write_training_csv(
                Path(directory), history, config, flops_per_token=10
            )
            rows = (Path(directory) / trainer.TRAINING_CSV_NAME).read_text().splitlines()
        self.assertEqual(
            rows[0],
            "step,tokens_processed,cumulative_estimated_flops,train_loss,"
            "learning_rate,grad_norm",
        )
        self.assertEqual(len(rows), 3)
        self.assertTrue(rows[1].startswith("1,32,320,2.0,"))
        self.assertTrue(rows[2].startswith("2,64,640,1.5,"))

    def test_diagnostics_csv_is_long_form_and_atomic(self) -> None:
        metadata = (("overall", None, 7), ("block", 0, 4))
        shape = (len(metadata), 3, 6)
        values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        config = SimpleNamespace(steps=2, batch_size=4, seq_len=8)
        points = (
            trainer.DiagnosticPoint(1, values),
            trainer.DiagnosticPoint(2, values + 1.0),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer.write_diagnostics_csv(root, points, metadata, config, 10)
            destination = root / trainer.DIAGNOSTICS_CSV_NAME
            rows = destination.read_text(encoding="utf-8").splitlines()
            self.assertFalse((root / ".diagnostics.csv.tmp").exists())
        self.assertEqual(
            rows[0],
            "step,tokens_processed,cumulative_estimated_flops,scope,layer,"
            "family,stat,value,element_count",
        )
        self.assertEqual(len(rows), 1 + 2 * 2 * 3 * 6)
        self.assertEqual(rows[1], "1,32,320,overall,,param,l1_norm,0.0,7")
        self.assertIn("2,64,640,block,0,update,fourth_moment,", rows[-1])

    def test_diagnostic_statistics_use_postupdate_param_raw_gradient_and_signed_delta(self) -> None:
        params = {
            "token_embedding": np.asarray([[1.0, 2.0]], dtype=np.float32),
            "position_embedding": np.asarray([[3.0, 4.0]], dtype=np.float32),
            "blocks": [
                {"weight": np.asarray([-1.0, 1.0], dtype=np.float32)}
            ],
            "final_ln_scale": np.asarray([2.0], dtype=np.float32),
            "final_ln_bias": np.asarray([0.0], dtype=np.float32),
        }
        gradients = trainer.jax.tree_util.tree_map(
            lambda value: np.full_like(value, 2.0), params
        )
        after = trainer.jax.tree_util.tree_map(
            lambda value: value + np.float32(-0.25), params
        )
        values = np.asarray(trainer.diagnostic_values(params, gradients, after))
        metadata = trainer.diagnostic_scope_metadata(params)
        self.assertEqual(
            metadata,
            (("overall", None, 8), ("embeddings", None, 4),
             ("block", 0, 2), ("final_norm", None, 2)),
        )
        flattened = np.concatenate(
            [np.ravel(value) for value in trainer.jax.tree_util.tree_leaves(after)]
        ).astype(np.float32)
        expected_param = np.asarray(
            [
                np.abs(flattened).sum(),
                np.linalg.norm(flattened),
                flattened.mean(),
                flattened.std(),
                np.mean((flattened - flattened.mean()) ** 3),
                np.mean((flattened - flattened.mean()) ** 4),
            ]
        )
        np.testing.assert_allclose(values[0, 0], expected_param, rtol=1e-6)
        self.assertAlmostEqual(float(values[0, 1, 2]), 2.0)
        self.assertAlmostEqual(float(values[0, 2, 2]), -0.25)

    def test_diagnostic_executable_preserves_ordinary_optimizer_trajectory(self) -> None:
        parser = trainer.build_parser()
        config = trainer.resolve_config(
            parser.parse_args(
                [
                    "--profile", "smoke", "--steps", "2", "--diagnostics-every", "1"
                ]
            ),
            "cpu",
            256,
        )
        host_params = trainer.init_params(config, 7)
        decay_mask = trainer.weight_decay_mask(host_params)
        x = np.arange(config.batch_size * config.seq_len, dtype=np.int32).reshape(
            config.batch_size, config.seq_len
        ) % config.vocab_size
        y = (x + 1) % config.vocab_size

        ordinary = trainer.jax.jit(
            lambda p, o, bx, by: trainer.train_step(
                p, o, bx, by, config, decay_mask
            )
        )
        diagnostic = trainer.jax.jit(
            lambda p, o, bx, by: trainer.diagnostic_train_step(
                p, o, bx, by, config, decay_mask
            )
        )
        params_a = trainer.jax.tree_util.tree_map(np.copy, host_params)
        params_b = trainer.jax.tree_util.tree_map(np.copy, host_params)
        optimizer_a = trainer.init_optimizer(params_a, config.steps)
        optimizer_b = trainer.init_optimizer(params_b, config.steps)
        params_a, optimizer_a, metrics_a = ordinary(params_a, optimizer_a, x, y)
        params_b, optimizer_b, metrics_b, diagnostics = diagnostic(
            params_b, optimizer_b, x, y
        )
        trainer.sync_tree((params_a, optimizer_a, params_b, optimizer_b, diagnostics))
        for left, right in zip(
            trainer.jax.tree_util.tree_leaves((params_a, optimizer_a, metrics_a)),
            trainer.jax.tree_util.tree_leaves((params_b, optimizer_b, metrics_b)),
            strict=True,
        ):
            np.testing.assert_array_equal(np.asarray(left), np.asarray(right))

    def test_validation_csv_contains_probes_and_canonical_final_row(self) -> None:
        rows: list[trainer.ValidationRow] = [
            trainer.ValidationRow(
                250, 8_192_000, "fineweb_probe", "fineweb",
                262_144, 4.0, np.exp(4.0), 0.25, False
            ),
            trainer.ValidationRow(
                500, 16_384_000, "fineweb", "fineweb",
                10_485_760, 3.5, np.exp(3.5), 8.0, True
            ),
            trainer.ValidationRow(
                500, 16_384_000, "downstream", "science",
                8_192, 3.0, np.exp(3.0), 0.03, False
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer.write_validation_csv(root, rows)
            contents = (root / trainer.VALIDATION_CSV_NAME).read_text().splitlines()
            temporary = root / f".{trainer.VALIDATION_CSV_NAME}.tmp"
            self.assertFalse(temporary.exists())
        self.assertEqual(
            contents[0],
            "step,tokens_processed,kind,domain,validation_tokens,validation_loss,"
            "perplexity,validation_seconds,canonical",
        )
        self.assertEqual(
            contents[1],
            f"250,8192000,fineweb_probe,fineweb,262144,4.0,{np.exp(4.0)},0.25,false",
        )
        self.assertEqual(
            contents[2],
            f"500,16384000,fineweb,fineweb,10485760,3.5,{np.exp(3.5)},8.0,true",
        )
        self.assertEqual(len(contents), 4)

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "canonical"):
                trainer.write_validation_csv(Path(directory), (rows[0], rows[2]))

    def test_downstream_batches_mask_document_boundaries_and_exact_targets(self) -> None:
        domain = trainer.DownstreamDomain(
            "science",
            np.asarray([99, 10, 11, 12, 99, 20, 21], dtype=np.uint16),
            (
                trainer.DocumentSpan(0, 4, 1, 3),
                trainer.DocumentSpan(4, 3, 5, 2),
            ),
        )
        config = SimpleNamespace(batch_size=2, seq_len=2)
        batches = trainer.downstream_batches(domain, config)
        pairs = []
        for x, y, mask in batches:
            flat_x, flat_y, flat_mask = x.ravel(), y.ravel(), mask.ravel()
            pairs.extend(
                (int(flat_x[index]), int(flat_y[index]))
                for index in np.flatnonzero(flat_mask)
            )
        self.assertEqual(pairs, [(99, 10), (10, 11), (11, 12), (99, 20), (20, 21)])
        self.assertNotIn((12, 99), pairs)
        self.assertEqual(sum(int(mask.sum()) for _, _, mask in batches), 5)

    def test_repeatable_downstream_data_groups_standalone_documents(self) -> None:
        parser = trainer.build_parser()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.npy"
            second = root / "second.npy"
            np.save(first, np.asarray([99, 1, 2], dtype=np.uint16))
            np.save(second, np.asarray([99, 3, 4, 5], dtype=np.uint16))
            args = parser.parse_args(
                [
                    "--downstream-data", f"science={first}",
                    "--downstream-data", f"science={second}",
                ]
            )
            domains = trainer.load_downstream_domains(args, 256)
        self.assertEqual(len(domains), 1)
        self.assertEqual(domains[0].name, "science")
        self.assertEqual(domains[0].scored_tokens, 5)
        self.assertEqual(len(domains[0].documents), 2)

    def test_gpt2_downstream_manifest_fits_padded_model_vocabulary(self) -> None:
        parser = trainer.build_parser()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = root / "science.bin"
            header = np.zeros(256, dtype="<i4")
            header[:3] = (20_240_520, 1, 3)
            tokens = np.asarray([50_256, 1, 2], dtype="<u2")
            shard.write_bytes(header.tobytes() + tokens.tobytes())
            manifest = root / "fresh10.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "kind": "fresh10",
                        "tokenizer": {"name": "gpt2", "vocab_size": 50_257},
                        "domains": [
                            {
                                "name": "science",
                                "path": shard.name,
                                "bytes": shard.stat().st_size,
                                "tokens": 3,
                                "scored_tokens": 2,
                                "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
                                "documents": [
                                    {
                                        "token_offset": 0,
                                        "token_count": 3,
                                        "score_offset": 1,
                                        "scored_tokens": 2,
                                    }
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            args = parser.parse_args(["--downstream-manifest", str(manifest)])
            domains = trainer.load_downstream_domains(args, 50_304)
            self.assertEqual(domains[0].scored_tokens, 2)
            with self.assertRaisesRegex(ValueError, "must fit the model vocabulary"):
                trainer.load_downstream_domains(args, 50_000)

    def test_console_writes_only_to_stderr(self) -> None:
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            console = trainer.Console("never")
            console.banner()
            console.table("test", (("field", "value"),))
            console.phase("phase", "detail")
            console.step(1, 1, 1.25, 1.0e-3, 0.5, 1024.0)
            console.success(1.0, 12.5, 0.25)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("GPT TPU SPEEDRUN", stderr.getvalue())
        self.assertIn("synchronized training 12.500s", stderr.getvalue())
        self.assertIn("compilation excluded", stderr.getvalue())
        self.assertIn("validation loss", stderr.getvalue())

    def test_weight_decay_mask_selects_matrices_not_bias_or_norm(self) -> None:
        params = {
            "token_embedding": np.zeros((16, 8), dtype=np.float32),
            "blocks": [
                {
                    "qkv_w": np.zeros((8, 24), dtype=np.float32),
                    "qkv_b": np.zeros((24,), dtype=np.float32),
                    "ln1_scale": np.ones((8,), dtype=np.float32),
                }
            ],
            "final_ln_bias": np.zeros((8,), dtype=np.float32),
        }
        mask = trainer.weight_decay_mask(params)
        self.assertTrue(mask["token_embedding"])
        self.assertTrue(mask["blocks"][0]["qkv_w"])
        self.assertFalse(mask["blocks"][0]["qkv_b"])
        self.assertFalse(mask["blocks"][0]["ln1_scale"])
        self.assertFalse(mask["final_ln_bias"])

    def test_official_topology_accepts_only_single_process_v4_8(self) -> None:
        v4_devices = [FakeDevice("tpu", "TPU v4") for _ in range(4)]
        with (
            patch.object(trainer.jax, "local_devices", return_value=v4_devices),
            patch.object(trainer.jax, "process_count", return_value=1),
            patch.object(trainer.jax, "device_count", return_value=4),
        ):
            trainer.validate_official_topology("official", v4_devices)

        invalid_cases = (
            (v4_devices, v4_devices, 2, 4),
            (v4_devices[:2], v4_devices[:2], 1, 2),
            (
                [FakeDevice("cpu", "cpu") for _ in range(4)],
                [FakeDevice("cpu", "cpu") for _ in range(4)],
                1,
                4,
            ),
            (
                [FakeDevice("tpu", "TPU v5p") for _ in range(4)],
                [FakeDevice("tpu", "TPU v5p") for _ in range(4)],
                1,
                4,
            ),
        )
        for devices, local_devices, process_count, device_count in invalid_cases:
            with self.subTest(devices=devices, process_count=process_count):
                with (
                    patch.object(trainer.jax, "local_devices", return_value=local_devices),
                    patch.object(trainer.jax, "process_count", return_value=process_count),
                    patch.object(trainer.jax, "device_count", return_value=device_count),
                    self.assertRaisesRegex(RuntimeError, "one TPU v4-8"),
                ):
                    trainer.validate_official_topology("official", devices)

    def test_system_metadata_is_versioned_and_topology_aware(self) -> None:
        devices = [FakeDevice("tpu", "TPU v4") for _ in range(4)]
        with (
            patch.object(trainer.jax, "device_count", return_value=4),
            patch.object(trainer.jax, "local_device_count", return_value=4),
            patch.object(trainer.jax, "process_count", return_value=1),
        ):
            metadata = trainer.system_metadata(devices)
        self.assertEqual(metadata["platform"], "tpu")
        self.assertEqual(metadata["device_count"], 4)
        self.assertEqual(metadata["local_device_count"], 4)
        self.assertEqual(metadata["process_count"], 1)
        self.assertEqual(metadata["device_kinds"], ["TPU v4"])
        self.assertEqual(metadata["jax_version"], trainer.jax.__version__)
        self.assertEqual(metadata["jaxlib_version"], trainer.jaxlib.__version__)
        self.assertIn("libtpu_version", metadata)
        self.assertIn("python_version", metadata)
        self.assertEqual(metadata["device_ids"], [None] * 4)

    def test_compile_metric_names_are_unambiguous(self) -> None:
        source = TRAINER_PATH.read_text(encoding="utf-8")
        self.assertIn('"train_compile_seconds":', source)
        self.assertIn('"eval_compile_seconds":', source)
        self.assertIn('"total_compile_seconds":', source)
        self.assertNotIn('"compile_seconds":', source)
        self.assertEqual(source.count("compiled_eval = jax.jit("), 1)
        self.assertIn('"validation_curve": VALIDATION_CSV_NAME', source)
        self.assertIn(
            ").lower(params, sample_x, sample_y, sample_mask).compile()",
            source,
        )
        probe = source.index("if should_run_validation_probe(step_index, config):")
        synchronize = source.index(
            "sync_tree((params, optimizer, last_metrics))", probe
        )
        evaluate = source.index("evaluate_validation_prefix(", synchronize)
        self.assertLess(synchronize, evaluate)


if __name__ == "__main__":
    unittest.main()
