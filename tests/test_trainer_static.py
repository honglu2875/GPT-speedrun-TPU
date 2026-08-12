from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, replace
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
    def test_yaml_config_is_authoritative_strict_and_versioned(self) -> None:
        source = trainer.CONFIG_PATH.read_text(encoding="utf-8")
        official = trainer.load_experiment_profile("official")
        self.assertEqual(official.schema_version, 1)
        self.assertEqual(official.train_tokens, 624_984_064)
        self.assertEqual(official.attention_backend, "tpu_flash")
        self.assertEqual(official.dtype_name, "bfloat16")
        self.assertEqual(
            official.source_sha256,
            hashlib.sha256(trainer.CONFIG_PATH.read_bytes()).hexdigest(),
        )

        invalid = {
            "duplicate": source.replace(
                "schema_version: 1", "schema_version: 1\nschema_version: 1", 1
            ),
            "unknown": source + "\nunknown: true\n",
            "anchor": source.replace("schema_version: 1", "schema_version: &v 1", 1),
            "alias": source.replace(
                "schema_version: 1", "schema_version: &v 1\nextra: *v", 1
            ),
            "tag": source.replace("schema_version: 1", "schema_version: !!int 1", 1),
            "directive": "%YAML 1.2\n---\n" + source,
            "multiple documents": source + "\n---\n{}\n",
            "nonfinite": source.replace(
                "learning_rate: 0.0003", "learning_rate: .nan", 1
            ),
            "invalid unselected profile": source.replace(
                "warmup_steps: 715", "warmup_steps: -1", 1
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for label, contents in invalid.items():
                with self.subTest(label=label):
                    path = root / "config.yaml"
                    path.write_text(contents, encoding="utf-8")
                    with patch.object(trainer, "CONFIG_PATH", path), self.assertRaises(
                        ValueError
                    ):
                        trainer.load_experiment_profile("smoke")
            path = root / "config.yaml"
            path.write_bytes(b"#" * (trainer._MAX_CONFIG_BYTES + 1))
            with patch.object(trainer, "CONFIG_PATH", path), self.assertRaisesRegex(
                ValueError, "safety limit"
            ):
                trainer.load_experiment_profile("smoke")

            target = root / "target.yaml"
            target.write_text(source, encoding="utf-8")
            symlink = root / "config-link.yaml"
            symlink.symlink_to(target)
            with patch.object(trainer, "CONFIG_PATH", symlink), self.assertRaisesRegex(
                ValueError, "non-symlink"
            ):
                trainer.load_experiment_profile("smoke")

            alternate = root / "alternate.yaml"
            alternate.write_text(source, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "beside train.py"):
                trainer.load_experiment_profile("smoke", alternate)

    def test_static_cli_values_are_rejected_but_diagnostic_overrides_resolve(self) -> None:
        parser = trainer.build_parser()
        for option in (
            ("--layers", "13"),
            ("--attention-backend", "dense"),
            ("--learning-rate", "0.001"),
            ("--eval-batches", "1"),
        ):
            with self.subTest(option=option), self.assertRaisesRegex(
                ValueError, "defined by sibling config.yaml"
            ):
                trainer.resolve_config(
                    parser.parse_args(["--profile", "official", *option]),
                    "tpu",
                    50_304,
                )
        config = trainer.resolve_config(
            parser.parse_args(
                [
                    "--profile", "official", "--steps", "100",
                    "--val-every", "0", "--diagnostics-every", "0",
                    "--log-every", "100",
                ]
            ),
            "tpu",
            50_304,
        )
        self.assertEqual(config.steps, 100)
        self.assertEqual(config.val_every, 0)
        self.assertEqual(
            dict(config.config_overrides),
            {"steps": 100, "val_every": 0, "diagnostics_every": 0, "log_every": 100},
        )

    def test_train_tokens_derives_exact_steps_and_is_exclusive(self) -> None:
        parser = trainer.build_parser()
        config = trainer.resolve_config(
            parser.parse_args(
                [
                    "--profile", "official",
                    "--train-tokens", "655360",
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
                    ]
                ),
                "tpu",
                50_304,
            )
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--steps", "20", "--train-tokens", "655360"])

    def test_short_diagnostic_run_preserves_yaml_warmup(self) -> None:
        parser = trainer.build_parser()
        config = trainer.resolve_config(
            parser.parse_args(["--profile", "official", "--steps", "100"]),
            "tpu",
            50_304,
        )
        self.assertEqual(config.steps, 100)
        self.assertEqual(config.warmup_steps, 715)
        with self.assertRaisesRegex(ValueError, "defined by sibling config.yaml"):
            trainer.resolve_config(
                parser.parse_args(
                    ["--profile", "official", "--warmup-steps", "100"]
                ),
                "tpu",
                50_304,
            )

    def test_xprof_diagnostic_contract_and_capture_window(self) -> None:
        parser = trainer.build_parser()
        defaults = parser.parse_args([])
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

        with patch.object(trainer, "make_runtime_key") as make_key:
            runtime = trainer.prepare_attention_runtime(
                defaults,
                SimpleNamespace(attention_backend="dense"),
                (),
            )
        make_key.assert_not_called()
        self.assertEqual(
            trainer.attention_runtime_metadata(runtime),
            {
                "key_digest": None,
                "resolution_source": "dense",
                "tune_seconds": 0.0,
                "tiles": None,
            },
        )
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

        smoke = trainer.resolve_config(
            parser.parse_args(["--profile", "smoke"]), "cpu", 256
        )
        development = trainer.resolve_config(
            parser.parse_args(["--profile", "dev"]), "tpu", 50_304
        )
        for config in (smoke, development):
            self.assertEqual(config.val_every, 0)
            self.assertEqual(config.diagnostics_every, 0)

        overridden = trainer.resolve_config(
            parser.parse_args(
                [
                    "--profile",
                    "dev",
                    "--val-every",
                    "5",
                ]
            ),
            "tpu",
            50_304,
        )
        self.assertEqual(overridden.val_every, 5)
        self.assertEqual(overridden.val_probe_batches, 8)
        disabled = trainer.resolve_config(
            parser.parse_args(
                [
                    "--profile",
                    "official",
                    "--steps",
                    "1",
                    "--val-every",
                    "0",
                ]
            ),
            "tpu",
            50_304,
        )
        self.assertEqual(disabled.val_every, 0)
        self.assertEqual(disabled.eval_batches, 320)
        self.assertEqual(disabled.val_probe_batches, 8)
        for option in (
            ("--batch-size", "128"),
            ("--seq-len", "16384"),
            ("--val-probe-batches", "9"),
        ):
            with self.subTest(option=option), self.assertRaisesRegex(
                ValueError, "defined by sibling config.yaml"
            ):
                trainer.resolve_config(
                    parser.parse_args(["--profile", "official", *option]),
                    "tpu",
                    50_304,
                )
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            trainer.resolve_config(
                parser.parse_args(
                    ["--profile", "official", "--train-tokens", "655361"]
                ),
                "tpu",
                50_304,
            )

    def test_tiled_loss_resolves_semantic_vocab_and_counts_recompute_flops(self) -> None:
        parser = trainer.build_parser()
        dense = trainer.resolve_config(
            parser.parse_args(["--profile", "official"]), "tpu", 50_304
        )
        experiment = replace(
            trainer.load_experiment_profile("official"), loss_backend="tiled"
        )
        tiled = trainer.resolve_config(
            parser.parse_args(["--profile", "official"]),
            "tpu",
            50_304,
            experiment,
        )
        self.assertEqual(dense.semantic_vocab_size, 50_304)
        # Switching kernels alone preserves the calibrated 50,304-class
        # objective. Masking storage-only rows is an explicit algorithm choice.
        self.assertEqual(tiled.semantic_vocab_size, 50_304)
        self.assertEqual(tiled.vocab_tile_size, 2_048)
        params_total = 124_475_904
        self.assertGreater(
            trainer.estimated_flops_per_token(tiled, params_total),
            trainer.estimated_flops_per_token(dense, params_total),
        )

        with self.assertRaisesRegex(ValueError, "defined by sibling config.yaml"):
            trainer.resolve_config(
                parser.parse_args(["--profile", "official", "--loss-backend", "tiled"]),
                "tpu", 50_304,
            )

    def test_flash_flops_include_right_padding_for_odd_sequences(self) -> None:
        parser = trainer.build_parser()
        common = ["--profile", "dev"]
        base = trainer.load_experiment_profile("dev")
        dense = trainer.resolve_config(
            parser.parse_args(common), "tpu", 50_304, replace(base, seq_len=129)
        )
        flash = trainer.resolve_config(
            parser.parse_args(common),
            "tpu",
            50_304,
            replace(base, seq_len=129, attention_backend="tpu_flash"),
        )
        self.assertGreater(
            trainer.estimated_flops_per_token(flash, 1_000_000),
            trainer.estimated_flops_per_token(dense, 1_000_000),
        )

    def test_batches_reject_tokens_outside_semantic_vocabulary(self) -> None:
        tokens = np.asarray([0, 1, 2, 7, 3, 0, 1, 2], dtype=np.int32)
        dataset = trainer.TokenDataset(
            trainer.ShardedTokens((tokens,)),
            trainer.ShardedTokens((tokens,)),
            "test",
        )
        rng = np.random.default_rng(3)
        with self.assertRaisesRegex(ValueError, "do not fit"):
            dataset.batch("train", rng, 1, 7, 7)
        with self.assertRaisesRegex(ValueError, "do not fit"):
            dataset.validation_batch(0, 1, 7, 7)

    def test_trainable_flash_attention_backends_are_tpu_only(self) -> None:
        parser = trainer.build_parser()
        base = trainer.load_experiment_profile("dev")
        for backend in ("jax_flash", "tpu_flash"):
            with self.subTest(backend=backend):
                args = parser.parse_args(["--profile", "dev"])
                experiment = replace(base, attention_backend=backend)
                with self.assertRaisesRegex(ValueError, "requires a TPU"):
                    trainer.resolve_config(args, "cpu", 50_304, experiment)
                config = trainer.resolve_config(args, "tpu", 50_304, experiment)
                self.assertEqual(config.attention_backend, backend)
        for backend in ("jax_flash", "tpu_flash"):
            with self.subTest(float32_backend=backend):
                with self.assertRaisesRegex(ValueError, "requires dtype bfloat16"):
                    trainer.resolve_config(
                        parser.parse_args(["--profile", "dev"]),
                        "tpu",
                        50_304,
                        replace(
                            base, attention_backend=backend, dtype_name="float32"
                        ),
                    )

    def test_attention_autotune_cli_requires_cache_and_non_dense_backend(self) -> None:
        parser = trainer.build_parser()
        defaults = parser.parse_args([])
        self.assertIsNone(defaults.attention_backend)
        self.assertIsNone(defaults.attention_tuning_cache)
        self.assertFalse(defaults.autotune_attention)

        official = trainer.resolve_config(
            parser.parse_args(["--profile", "official"]), "tpu", 50_304
        )
        smoke = trainer.resolve_config(
            parser.parse_args(["--profile", "smoke"]), "cpu", 256
        )
        development = trainer.resolve_config(
            parser.parse_args(["--profile", "dev"]), "tpu", 50_304
        )
        self.assertEqual(official.attention_backend, "tpu_flash")
        self.assertEqual(smoke.attention_backend, "dense")
        self.assertEqual(development.attention_backend, "dense")

        with self.assertRaisesRegex(ValueError, "requires --attention-tuning-cache"):
            trainer.validate_args(
                parser.parse_args(
                    ["--profile", "official", "--autotune-attention"]
                )
            )
        with self.assertRaisesRegex(ValueError, "non-dense attention_backend"):
            trainer.validate_args(
                parser.parse_args(
                    ["--attention-tuning-cache", "attention-tuning.json"]
                )
            )

        valid = parser.parse_args(
            [
                "--profile",
                "official",
                "--attention-tuning-cache",
                "attention-tuning.json",
                "--autotune-attention",
            ]
        )
        trainer.validate_args(valid)

    def test_ordinary_attention_resolution_uses_exact_local_shape(self) -> None:
        parser = trainer.build_parser()
        args = parser.parse_args(
            [
                "--profile",
                "official",
                "--attention-tuning-cache",
                "attention-tuning.json",
            ]
        )
        config = trainer.resolve_config(args, "tpu", 50_304)
        devices = [FakeDevice("tpu", "TPU v4") for _ in range(4)]
        tiles = trainer.AttentionTiles(
            512, 512, 256, 512, 256, 512, 256, 256, 512, 256
        )
        key = SimpleNamespace(digest="a" * 64)
        resolved = SimpleNamespace(source="shipped", tiles=tiles)
        with (
            patch.object(trainer, "make_runtime_key", return_value=key) as make_key,
            patch.object(
                trainer, "resolve_attention_tile_plan", return_value=resolved
            ) as resolve,
            patch.object(trainer, "autotune_attention") as autotune,
        ):
            runtime = trainer.prepare_attention_runtime(args, config, devices)

        key_arguments = make_key.call_args.kwargs
        self.assertEqual(key_arguments["backend"], "tpu_flash")
        self.assertEqual(key_arguments["batch"], 8)
        self.assertEqual(key_arguments["heads"], 12)
        self.assertEqual(key_arguments["sequence"], 1_024)
        self.assertEqual(key_arguments["head_dim"], 64)
        self.assertEqual(key_arguments["mode"], "forward_backward")
        self.assertIs(key_arguments["device"], devices[0])
        resolve.assert_called_once_with(
            key, cache_path=Path("attention-tuning.json").resolve()
        )
        autotune.assert_not_called()
        self.assertEqual(runtime.key_digest, "a" * 64)
        self.assertEqual(runtime.resolution_source, "shipped")
        self.assertEqual(runtime.tiles, tiles)
        self.assertEqual(runtime.tune_seconds, 0.0)
        source = TRAINER_PATH.read_text(encoding="utf-8")
        self.assertLess(
            source.index(
                "attention_runtime = prepare_attention_runtime(args, config, devices)"
            ),
            source.index('mesh = Mesh(np.asarray(devices, dtype=object), ("data",))'),
        )
        self.assertLess(
            source.index('mesh = Mesh(np.asarray(devices, dtype=object), ("data",))'),
            source.index(
                "attention_fn = make_mesh_attention(config, mesh, attention_runtime.tiles)"
            ),
        )

    def test_explicit_attention_autotune_is_synthetic_and_forced(self) -> None:
        parser = trainer.build_parser()
        args = parser.parse_args(
            [
                "--profile",
                "official",
                "--attention-tuning-cache",
                "attention-tuning.json",
                "--autotune-attention",
            ]
        )
        experiment = replace(
            trainer.load_experiment_profile("official"),
            attention_backend="jax_flash",
        )
        config = trainer.resolve_config(args, "tpu", 50_304, experiment)
        devices = [FakeDevice("tpu", "TPU v4") for _ in range(4)]
        tiles = trainer.AttentionTiles(
            512, 512, 256, 512, 256, 512, 256, 256, 512, 256
        )
        key = SimpleNamespace(digest="b" * 64)
        record = SimpleNamespace(winner=tiles)
        sentinel_attention = object()
        with (
            patch.object(trainer, "make_runtime_key", return_value=key),
            patch.object(
                trainer, "autotune_attention", return_value=record
            ) as autotune,
            patch.object(
                trainer, "make_causal_attention", return_value=sentinel_attention
            ) as make_attention,
            patch.object(trainer.time, "perf_counter", side_effect=(10.0, 12.5)),
        ):
            runtime = trainer.prepare_attention_runtime(args, config, devices)
            tune_arguments = autotune.call_args.kwargs
            built_attention = tune_arguments["attention_factory"](tiles)

        self.assertIs(built_attention, sentinel_attention)
        attention_config = make_attention.call_args.args[0]
        self.assertEqual(attention_config.backend, "jax_flash")
        self.assertEqual(attention_config.tiles, tiles)
        self.assertEqual(tune_arguments["key"], key)
        self.assertEqual(
            tune_arguments["cache_path"], Path("attention-tuning.json").resolve()
        )
        self.assertIs(tune_arguments["device"], devices[0])
        self.assertTrue(tune_arguments["force"])
        self.assertEqual(runtime.key_digest, "b" * 64)
        self.assertEqual(runtime.resolution_source, "autotuned")
        self.assertEqual(runtime.tiles, tiles)
        self.assertEqual(runtime.tune_seconds, 2.5)

    def test_mesh_attention_uses_pre_resolved_exact_plan(self) -> None:
        parser = trainer.build_parser()
        config = trainer.resolve_config(
            parser.parse_args(["--profile", "official"]),
            "tpu",
            50_304,
        )
        tiles = trainer.AttentionTiles(
            512, 512, 256, 512, 256, 512, 256, 256, 512, 256
        )
        local_attention = object()
        with (
            patch.object(
                trainer, "make_causal_attention", return_value=local_attention
            ) as make_attention,
            patch.object(trainer.jax, "shard_map", return_value="mapped") as shard,
        ):
            actual = trainer.make_mesh_attention(config, object(), tiles)

        self.assertEqual(actual, "mapped")
        attention_config = make_attention.call_args.args[0]
        self.assertEqual(attention_config.backend, "tpu_flash")
        self.assertEqual(attention_config.tiles, tiles)
        self.assertIs(shard.call_args.args[0], local_attention)
        with self.assertRaisesRegex(ValueError, "resolved tile plan"):
            trainer.make_mesh_attention(config, object(), None)
        self.assertIsNone(
            trainer.make_mesh_attention(
                SimpleNamespace(attention_backend="dense"), object(), None
            )
        )

    def test_attention_tuning_metadata_is_saved_with_exact_plan(self) -> None:
        parser = trainer.build_parser()
        config = trainer.resolve_config(
            parser.parse_args(["--profile", "smoke"]), "cpu", 256
        )
        tiles = trainer.AttentionTiles(
            512, 512, 256, 512, 256, 512, 256, 256, 512, 256
        )
        runtime = trainer.AttentionRuntime("c" * 64, "cache", tiles, 0.0)
        expected = trainer.attention_runtime_metadata(runtime)
        self.assertEqual(expected["key_digest"], "c" * 64)
        self.assertEqual(expected["resolution_source"], "cache")
        self.assertEqual(len(expected["tiles"]), 10)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            trainer.save_checkpoint(
                output,
                {"weight": np.zeros((2, 2), dtype=np.float32)},
                config,
                7,
                runtime,
            )
            with np.load(output / trainer.CHECKPOINT_NAME) as checkpoint:
                metadata = json.loads(bytes(checkpoint["metadata.json"]).decode())
        self.assertEqual(metadata["model"]["attention_tuning"], expected)
        self.assertEqual(metadata["configuration"]["path"], "config.yaml")
        self.assertEqual(
            metadata["configuration"]["sha256"], config.config_sha256
        )
        self.assertEqual(
            metadata["configuration"]["resolved"]["model"]["layers"], 2
        )

        implementation = trainer.implementation_metadata(config, runtime)
        self.assertEqual(implementation["attention_tuning"], expected)
        self.assertEqual(
            implementation["configuration"],
            trainer.experiment_config_metadata(config),
        )
        rows = trainer.attention_console_rows(runtime)
        self.assertEqual(
            rows,
            (
                ("attention tuning", "cache · key cccccccccccc"),
                ("attention fwd", "q512 · kv512/256"),
                ("attention dK/dV", "q512/256 · kv512/256"),
                ("attention dQ", "q256 · kv512/256"),
            ),
        )
        self.assertNotIn("c" * 64, repr(rows))
        source = TRAINER_PATH.read_text(encoding="utf-8")
        self.assertIn('"attention_tune_seconds":', source)

    def test_kernel_provenance_does_not_change_fixed_model_contract(self) -> None:
        parser = trainer.build_parser()
        experiment = replace(
            trainer.load_experiment_profile("official"),
            loss_backend="tiled",
            semantic_vocab_size=50_257,
        )
        config = trainer.resolve_config(
            parser.parse_args(["--profile", "official"]),
            "tpu",
            50_304,
            experiment,
        )
        tiles = trainer.AttentionTiles(
            512, 512, 256, 512, 256, 512, 256, 256, 512, 256
        )
        runtime = trainer.AttentionRuntime("d" * 64, "shipped", tiles, 0.0)
        self.assertEqual(
            trainer.contract_model_metadata(config),
            {
                "layers": 12,
                "heads": 12,
                "d_model": 768,
                "mlp_mult": 4,
                "vocab_size": 50_304,
                "semantic_vocab_size": 50_257,
                "tied_embeddings": True,
            },
        )
        implementation = trainer.implementation_metadata(config, runtime)
        self.assertEqual(implementation["attention_backend"], "tpu_flash")
        self.assertEqual(implementation["loss_backend"], "tiled")
        self.assertNotIn("semantic_vocab_size", implementation)
        self.assertNotIn("attention_backend", trainer.contract_model_metadata(config))

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
        config = SimpleNamespace(
            batch_size=2, seq_len=4, vocab_size=16, semantic_vocab_size=16
        )

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
            console.table(
                "test",
                (("field", "value"), ("long field", "x" * 512)),
            )
            console.phase("phase", "detail")
            console.step(1, 1, 1.25, 1.0e-3, 0.5, 1024.0)
            console.success(1.0, 12.5, 0.25)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("GPT TPU SPEEDRUN", stderr.getvalue())
        self.assertIn("synchronized training 12.500s", stderr.getvalue())
        self.assertIn("compilation excluded", stderr.getvalue())
        self.assertIn("validation loss", stderr.getvalue())
        table_lines = [
            line for line in stderr.getvalue().splitlines() if "│" in line
        ]
        self.assertTrue(table_lines)
        self.assertLessEqual(max(map(len, table_lines)), 80)
        self.assertIn("…", stderr.getvalue())

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

    def test_official_topology_accepts_single_or_multi_host_v4(self) -> None:
        v4_devices = [FakeDevice("tpu", "TPU v4") for _ in range(4)]
        with (
            patch.object(trainer.jax, "local_devices", return_value=v4_devices),
            patch.object(trainer.jax, "process_count", return_value=1),
            patch.object(trainer.jax, "device_count", return_value=4),
        ):
            trainer.validate_official_topology("official", v4_devices)

        global_v4_devices = [FakeDevice("tpu", "TPU v4") for _ in range(8)]
        with (
            patch.object(trainer.jax, "local_devices", return_value=v4_devices),
            patch.object(trainer.jax, "process_count", return_value=2),
            patch.object(trainer.jax, "device_count", return_value=8),
        ):
            trainer.validate_official_topology("official", global_v4_devices)

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
                    self.assertRaisesRegex(RuntimeError, "4 local TPU v4"),
                ):
                    trainer.validate_official_topology("official", devices)

    def test_rank_local_slice_partitions_global_batch_without_overlap(self) -> None:
        values = np.arange(24, dtype=np.int32).reshape(8, 3)
        pieces = [trainer.rank_local_slice(values, rank, 4) for rank in range(4)]
        np.testing.assert_array_equal(np.concatenate(pieces), values)
        self.assertTrue(all(piece.flags.c_contiguous for piece in pieces))
        with self.assertRaisesRegex(ValueError, "divisible"):
            trainer.rank_local_slice(values[:7], 0, 4)

    def test_controller_hostname_is_independent_of_jax_process_index(self) -> None:
        with (
            patch.dict(
                trainer.os.environ,
                {"SPEEDRUN_CONTROLLER_HOSTNAME": "slice-w-0"},
                clear=False,
            ),
            patch.object(trainer.socket, "gethostname", return_value="slice-w-0"),
        ):
            self.assertTrue(trainer.is_controller_process(3))
        with (
            patch.dict(
                trainer.os.environ,
                {"SPEEDRUN_CONTROLLER_HOSTNAME": "slice-w-0"},
                clear=False,
            ),
            patch.object(trainer.socket, "gethostname", return_value="slice-w-2"),
        ):
            self.assertFalse(trainer.is_controller_process(0))

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
