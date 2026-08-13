from __future__ import annotations

import hashlib
import io
import json
import math
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from harness import (
    ConfigurationError,
    ReferenceContract,
    ResultValidationError,
    RunConfig,
    SubmissionError,
    load_records,
    parse_result_line,
    rank_records,
    render_leaderboard,
    run_submission,
    validate_result,
    verify_run,
)
from harness.runner import _validate_payload_identity


FAKE_TRAINER = r'''from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--seed", required=True, type=int)
parser.add_argument("--track", required=True)
parser.add_argument("--profile", required=True)
parser.add_argument("--tag", action="append", default=[])
parser.add_argument("--seeded", action="store_true")
parser.add_argument("--stderr-message", default="")
parser.add_argument("--stderr-bytes", type=int, default=0)
parser.add_argument("--sleep-after-stderr", type=float, default=0.0)
parser.add_argument("--make-cache", action="store_true")
parser.add_argument("--evaluations-json")
parser.add_argument("--omit-checkpoint", action="store_true")
args = parser.parse_args()
if args.stderr_message:
    sys.stderr.write(args.stderr_message)
if args.stderr_bytes:
    sys.stderr.buffer.write(b"x" * args.stderr_bytes)
sys.stderr.flush()
if args.sleep_after_stderr:
    time.sleep(args.sleep_after_stderr)
output = Path(args.output_dir)
if args.make_cache:
    (output / ".jax_cache").mkdir()
    (output / ".jax_cache" / "compiled.bin").write_bytes(b"temporary")
if not args.omit_checkpoint:
    (output / "model.npz").write_bytes(b"tiny checkpoint")
(output / "training.csv").write_text("step,train_loss\n1,2.5\n")
(output / "seen.json").write_text(json.dumps({"config": args.config, "seed": args.seed, "track": args.track, "profile": args.profile, "tag": args.tag, "seeded": args.seeded}))
result = {
    "schema_version": 1,
    "status": "ok",
    "track": args.track,
    "profile": args.profile,
    "seed": args.seed,
    "checkpoint": None if args.omit_checkpoint else "model.npz",
    "artifacts": {"training_curve": "training.csv"},
    "metrics": {
        "train_seconds": 0.125,
        "tokens_processed": 96,
        "validation_loss": 2.5,
        "validation_tokens": 64,
        "compile_seconds": 0.75,
        "diagnostics": {"gradient_scale": -2.0},
    },
    "contract": {
        "model_id": "tiny-gpt-v1",
        "dataset_id": "tiny-data-v1",
        "tokenizer_id": "byte-v1",
        "sequence_length": 8,
    },
    "implementation": {
        "attention_backend": "dense",
        "loss_backend": "dense",
    },
    "system": {"platform": "test", "devices": 1},
}
if args.evaluations_json is not None:
    result["evaluations"] = json.loads(args.evaluations_json)
print("human log output")
print("SPEEDRUN_RESULT=" + json.dumps(result, separators=(",", ":")))
'''


class HarnessRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        submission = self.root / "submissions" / "tiny"
        submission.mkdir(parents=True)
        (submission / "train.py").write_text(FAKE_TRAINER, encoding="utf-8")
        (submission / "config.yaml").write_text("steps: 1\n", encoding="utf-8")
        (self.root / "uv.lock").write_bytes(b"version = 1\n")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def config(self, **changes: object) -> RunConfig:
        values: dict[str, object] = {
            "repo_root": self.root,
            "submission": "tiny",
            "runs_dir": Path("runs"),
            "records_path": Path("records/runs.jsonl"),
            "python_executable": sys.executable,
            "passthrough_args": ("--tag", "one", "--tag", "two"),
        }
        values.update(changes)
        return RunConfig(**values)  # type: ignore[arg-type]

    @staticmethod
    def fresh10_evaluations(
        domain_tokens: dict[str, int] | None = None,
    ) -> dict[str, object]:
        tokens = domain_tokens or {
            name: 8_192
            for name in (
                "science",
                "medicine",
                "software",
                "history",
                "fiction",
                "government",
                "legal",
                "economics",
                "climate",
                "education",
            )
        }
        losses = {name: 2.0 + index / 10 for index, name in enumerate(tokens)}
        macro_loss = math.fsum(losses.values()) / len(losses)
        return {
            "fineweb": {
                "loss": 2.5,
                "perplexity": math.exp(2.5),
                "scored_tokens": 64,
                "seconds": 0.25,
                "canonical": True,
            },
            "fresh10": {
                "domains": {
                    name: {
                        "loss": losses[name],
                        "perplexity": math.exp(losses[name]),
                        "scored_tokens": count,
                        "seconds": 0.01 + index / 100,
                    }
                    for index, (name, count) in enumerate(tokens.items())
                },
                "macro_loss": macro_loss,
                "macro_perplexity": math.exp(macro_loss),
                "scored_tokens": sum(tokens.values()),
                "seconds": math.fsum(
                    0.01 + index / 100 for index in range(len(tokens))
                ),
            },
        }

    def test_run_captures_validates_records_and_forwards_args(self) -> None:
        evaluator_calls: list[Path] = []

        def evaluator(checkpoint: Path, payload: object) -> dict[str, float]:
            evaluator_calls.append(checkpoint)
            self.assertTrue(checkpoint.exists())
            return {"validation_loss": 2.45}

        outcome = run_submission(self.config(target_loss=2.48), evaluator=evaluator)

        self.assertEqual(json.loads((outcome.run_dir / "seen.json").read_text()), {
            "config": str(self.root / "submissions" / "tiny" / "config.yaml"),
            "seed": 1337,
            "track": "open",
            "profile": "default",
            "tag": ["one", "two"],
            "seeded": False,
        })
        self.assertEqual(evaluator_calls, [outcome.checkpoint_path])
        self.assertTrue((outcome.run_dir / "stdout.log").is_file())
        self.assertTrue((outcome.run_dir / "stderr.log").is_file())
        self.assertTrue((outcome.run_dir / "result.json").is_file())
        records = load_records(outcome.record_path)
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertTrue(record["qualified"])
        self.assertGreater(record["timing"]["observed_wall_seconds"], 0)
        self.assertEqual(record["metrics"]["train_seconds"], 0.125)
        self.assertEqual(record["target_loss"], 2.48)
        self.assertEqual(record["metrics"]["validation_loss"], 2.45)
        self.assertEqual(record["metrics"]["evaluator"]["validation_loss"], 2.45)
        self.assertEqual(record["metrics"]["compile_seconds"], 0.75)
        self.assertEqual(record["metrics"]["diagnostics"]["gradient_scale"], -2.0)
        self.assertEqual(record["system"], {"platform": "test", "devices": 1})
        self.assertEqual(
            record["implementation"],
            {"attention_backend": "dense", "loss_backend": "dense"},
        )
        self.assertEqual(record["artifacts"]["training_curve"]["path"], "training.csv")
        self.assertEqual(len(record["artifacts"]["training_curve"]["sha256"]), 64)
        self.assertEqual(len(record["checkpoint"]["sha256"]), 64)

    def test_multi_host_run_builds_controller_owned_distributed_launch(self) -> None:
        def localize(**arguments: object) -> list[str]:
            return list(arguments["command"])  # type: ignore[arg-type]

        with (
            mock.patch(
                "harness.runner.build_distributed_launch_command",
                side_effect=localize,
            ) as build,
            mock.patch("harness.runner.socket.gethostname", return_value="slice-w-0"),
        ):
            outcome = run_submission(
                self.config(
                    tpu_vm_count=4,
                    tpu_vm_hosts="slice-w-[0-3]",
                )
            )

        remote_environment = build.call_args.kwargs["environment"]
        self.assertEqual(remote_environment["SPEEDRUN_DISTRIBUTED"], "1")
        self.assertEqual(remote_environment["SPEEDRUN_PROCESS_COUNT"], "4")
        self.assertEqual(
            remote_environment["SPEEDRUN_CONTROLLER_HOSTNAME"], "slice-w-0"
        )
        self.assertEqual(
            remote_environment["JAX_COMPILATION_CACHE_DIR"],
            f"/tmp/speedrun-jax-cache-{outcome.run_id}",
        )
        self.assertEqual(outcome.record["trainer_command"], outcome.record["command"])

    def test_interrupted_multi_host_run_cleans_exact_remote_workers(self) -> None:
        with (
            mock.patch(
                "harness.runner.build_distributed_launch_command",
                return_value=["pdsh", "synthetic"],
            ),
            mock.patch("harness.runner._run_process", side_effect=KeyboardInterrupt),
            mock.patch(
                "harness.runner.terminate_distributed_workers", return_value=True
            ) as terminate,
        ):
            with self.assertRaises(KeyboardInterrupt):
                run_submission(
                    self.config(
                        tpu_vm_count=4,
                        tpu_vm_hosts="slice-w-[0-3]",
                    )
                )

        arguments = terminate.call_args.kwargs
        self.assertEqual(arguments["host_expression"], "slice-w-[0-3]")
        self.assertEqual(arguments["host_count"], 4)
        self.assertEqual(arguments["script"].name, "train.py")
        self.assertIn("runs", arguments["output_dir"].parts)

    def test_rejects_reserved_passthrough_flags_but_not_prefixes(self) -> None:
        for flag in ("--config", "--output-dir", "--seed", "--track", "--profile"):
            for arguments in ((flag, "value"), (f"{flag}=value",), ("--", flag, "value")):
                with self.subTest(arguments=arguments):
                    with self.assertRaisesRegex(ConfigurationError, "reserved flag"):
                        run_submission(self.config(passthrough_args=arguments))

        outcome = run_submission(self.config(passthrough_args=("--seeded",)))
        seen = json.loads((outcome.run_dir / "seen.json").read_text())
        self.assertTrue(seen["seeded"])

    def test_result_identity_must_exactly_match_config(self) -> None:
        trainer = self.root / "submissions" / "tiny" / "train.py"
        original = trainer.read_text(encoding="utf-8")
        variants = {
            "track": original.replace('"track": args.track,', '"track": "sample_efficiency",'),
            "profile": original.replace('"profile": args.profile,', '"profile": "wrong",'),
            "seed": original.replace('"seed": args.seed,', '"seed": True,'),
            "missing": original.replace('    "track": args.track,\n', ""),
        }
        for label, source in variants.items():
            with self.subTest(label=label):
                trainer.write_text(source, encoding="utf-8")
                with self.assertRaisesRegex(ResultValidationError, "must exactly match"):
                    run_submission(self.config())
        trainer.write_text(original, encoding="utf-8")

    def test_official_identity_requires_exact_v4_8_system(self) -> None:
        payload = {
            "track": "open",
            "profile": "official",
            "seed": 1337,
            "system": {
                "platform": "tpu",
                "device_count": 4,
                "local_device_count": 4,
                "process_count": 1,
                "device_kinds": ["TPU v4"],
            },
        }
        config = self.config(profile="official")
        _validate_payload_identity(payload, config)
        payload["system"]["device_count"] = 8
        with self.assertRaisesRegex(ResultValidationError, "device_count"):
            _validate_payload_identity(payload, config)

    def test_official_identity_accepts_configured_multi_host_v4_system(self) -> None:
        payload = {
            "track": "open",
            "profile": "official",
            "seed": 1337,
            "system": {
                "platform": "tpu",
                "device_count": 16,
                "local_device_count": 4,
                "process_count": 4,
                "device_kinds": ["TPU v4"],
            },
        }
        config = self.config(
            profile="official",
            tpu_vm_count=4,
            tpu_vm_hosts="slice-w-[0-3]",
        )
        _validate_payload_identity(payload, config)
        payload["system"]["process_count"] = 3
        with self.assertRaisesRegex(ResultValidationError, "process_count"):
            _validate_payload_identity(payload, config)

    def test_fixed_validation_prefix_count_is_enforced(self) -> None:
        outcome = run_submission(self.config(expected_validation_tokens=64))
        self.assertEqual(outcome.record["metrics"]["validation_tokens"], 64)
        with self.assertRaisesRegex(ResultValidationError, "fixed validation prefix"):
            run_submission(self.config(expected_validation_tokens=65))

    def test_fresh10_is_optional_without_expectations(self) -> None:
        outcome = run_submission(self.config())
        self.assertNotIn("evaluations", outcome.record)

    def test_fresh10_contract_is_validated_and_preserved_in_record(self) -> None:
        expected_tokens = {
            name: 8_192
            for name in (
                "science",
                "medicine",
                "software",
                "history",
                "fiction",
                "government",
                "legal",
                "economics",
                "climate",
                "education",
            )
        }
        evaluations = self.fresh10_evaluations(expected_tokens)
        outcome = run_submission(
            self.config(
                target_loss=2.6,
                expected_validation_tokens=64,
                expected_downstream_tokens=expected_tokens,
                passthrough_args=("--evaluations-json", json.dumps(evaluations)),
            )
        )

        self.assertTrue(outcome.record["qualified"])
        self.assertEqual(outcome.record["metrics"]["validation_loss"], 2.5)
        self.assertEqual(outcome.record["evaluations"], evaluations)
        self.assertEqual(load_records(outcome.record_path)[0]["evaluations"], evaluations)

        evaluations["fresh10"]["macro_loss"] = 99  # type: ignore[index]
        self.assertNotEqual(
            outcome.record["evaluations"]["fresh10"]["macro_loss"],  # type: ignore[index]
            99,
        )

    def test_fresh10_expectations_require_evaluations(self) -> None:
        expected = {f"domain-{index}": 8_192 for index in range(10)}
        with self.assertRaisesRegex(ResultValidationError, "evaluations are required"):
            run_submission(self.config(expected_downstream_tokens=expected))

    def test_invalid_fresh10_config_fails_before_launch(self) -> None:
        for expected in (
            {f"domain-{index}": 8_192 for index in range(9)},
            {**{f"domain-{index}": 8_192 for index in range(9)}, "domain-9": 0},
        ):
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(ConfigurationError, "expected_downstream_tokens"):
                    run_submission(self.config(expected_downstream_tokens=expected))

    def test_provenance_hashes_inputs_and_copies_configured_values(self) -> None:
        trainer = self.root / "submissions" / "tiny" / "train.py"
        submission_config = self.root / "submissions" / "tiny" / "config.yaml"
        configured = {"data": {"manifest": "fineweb-α", "shards": 9}}
        outcome = run_submission(self.config(provenance=configured))
        provenance = outcome.record["provenance"]

        self.assertEqual(provenance["data"], configured["data"])
        self.assertIsNot(provenance["data"], configured["data"])
        self.assertEqual(provenance["train_py"]["bytes"], trainer.stat().st_size)
        self.assertEqual(
            provenance["train_py"]["sha256"], hashlib.sha256(trainer.read_bytes()).hexdigest()
        )
        self.assertEqual(
            provenance["config_yaml"],
            {
                "path": "submissions/tiny/config.yaml",
                "sha256": hashlib.sha256(submission_config.read_bytes()).hexdigest(),
                "bytes": submission_config.stat().st_size,
            },
        )
        self.assertEqual(
            provenance["uv_lock"]["sha256"],
            hashlib.sha256((self.root / "uv.lock").read_bytes()).hexdigest(),
        )
        self.assertEqual(provenance["shared_python"]["files"], 0)
        self.assertEqual(provenance["shared_python"]["bytes"], 0)
        self.assertEqual(
            provenance["shared_python"]["sha256"], hashlib.sha256().hexdigest()
        )

        configured["data"]["shards"] = 99
        self.assertEqual(provenance["data"]["shards"], 9)

        with self.assertRaisesRegex(ConfigurationError, "harness-owned"):
            run_submission(self.config(provenance={"train_py": {"spoofed": True}}))
        with self.assertRaisesRegex(ConfigurationError, "harness-owned"):
            run_submission(self.config(provenance={"shared_python": {"spoofed": True}}))
        with self.assertRaisesRegex(ConfigurationError, "harness-owned"):
            run_submission(self.config(provenance={"config_yaml": {"spoofed": True}}))

    def test_submission_config_must_be_a_regular_sibling_file(self) -> None:
        submission = self.root / "submissions" / "tiny"
        submission_config = submission / "config.yaml"
        submission_config.unlink()
        with self.assertRaisesRegex(ConfigurationError, "configuration file not found"):
            run_submission(self.config())

        target = submission / "elsewhere.yaml"
        target.write_text("steps: 1\n", encoding="utf-8")
        submission_config.symlink_to(target.name)
        with self.assertRaisesRegex(ConfigurationError, "configuration file not found"):
            run_submission(self.config())

    def test_shared_python_provenance_changes_with_dependency_bytes(self) -> None:
        shared = self.root / "speedrun" / "kernels"
        shared.mkdir(parents=True)
        dependency = shared / "attention.py"
        dependency.write_text("PLAN = 128\n", encoding="utf-8")
        first = run_submission(self.config()).record["provenance"]["shared_python"]
        dependency.write_text("PLAN = 512\n", encoding="utf-8")
        second = run_submission(self.config()).record["provenance"]["shared_python"]
        self.assertEqual(first["files"], 1)
        self.assertNotEqual(first["sha256"], second["sha256"])
        self.assertNotEqual(
            first["entries"][0]["sha256"], second["entries"][0]["sha256"]
        )

    def test_rejects_nonfinite_declared_metrics_and_provenance(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "finite JSON"):
            run_submission(self.config(provenance={"bad": float("nan")}))

        trainer = self.root / "submissions" / "tiny" / "train.py"
        source = trainer.read_text(encoding="utf-8").replace(
            '"compile_seconds": 0.75,', '"compile_seconds": float("inf"),'
        )
        trainer.write_text(source, encoding="utf-8")
        with self.assertRaisesRegex(ResultValidationError, "finite JSON"):
            run_submission(self.config())

    def test_stderr_is_teed_live_and_captured_byte_for_byte(self) -> None:
        captured = io.StringIO()
        result: list[object] = []
        config = self.config(
            passthrough_args=(
                "--stderr-message",
                "live marker",
                "--sleep-after-stderr",
                "0.35",
            )
        )

        with mock.patch("harness.runner.sys.stderr", captured):
            thread = threading.Thread(target=lambda: result.append(run_submission(config)))
            thread.start()
            deadline = time.monotonic() + 2.0
            while "live marker" not in captured.getvalue() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertIn("live marker", captured.getvalue())
            self.assertTrue(thread.is_alive(), "stderr was not visible until after process exit")
            thread.join(3.0)

        self.assertFalse(thread.is_alive())
        outcome = result[0]
        self.assertEqual((outcome.run_dir / "stderr.log").read_bytes(), b"live marker")

    def test_large_stderr_does_not_deadlock_and_timeout_keeps_partial_log(self) -> None:
        captured = io.StringIO()
        with mock.patch("harness.runner.sys.stderr", captured):
            outcome = run_submission(
                self.config(passthrough_args=("--stderr-bytes", str(512 * 1024)))
            )
        self.assertEqual((outcome.run_dir / "stderr.log").stat().st_size, 512 * 1024)

        timeout_config = self.config(
            passthrough_args=(
                "--stderr-message",
                "before timeout",
                "--sleep-after-stderr",
                "10",
            ),
            timeout_seconds=0.1,
        )
        with mock.patch("harness.runner.sys.stderr", io.StringIO()):
            started = time.monotonic()
            with self.assertRaisesRegex(SubmissionError, "timed out"):
                run_submission(timeout_config)
            self.assertLess(time.monotonic() - started, 2.0)
        latest_run = max((self.root / "runs").iterdir(), key=lambda path: path.stat().st_mtime_ns)
        self.assertEqual((latest_run / "stderr.log").read_bytes(), b"before timeout")

    def test_discards_per_run_compilation_cache_and_rejects_bad_timeouts(self) -> None:
        outcome = run_submission(self.config(passthrough_args=("--make-cache",)))
        self.assertFalse((outcome.run_dir / ".jax_cache").exists())

        for value in (True, float("nan"), float("inf"), 10**10_000):
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaisesRegex(ConfigurationError, "timeout_seconds"):
                    run_submission(self.config(timeout_seconds=value))

    def test_sample_efficiency_requires_and_enforces_contract(self) -> None:
        with self.assertRaises(ConfigurationError):
            run_submission(self.config(track="sample_efficiency"))

        reference = ReferenceContract(
            model_id="tiny-gpt-v1",
            dataset_id="tiny-data-v1",
            tokenizer_id="byte-v1",
            sequence_length=8,
        )
        outcome = run_submission(
            self.config(track="sample_efficiency", reference_contract=reference)
        )
        self.assertEqual(outcome.record["track"], "sample_efficiency")
        self.assertEqual(outcome.record["reference_contract"], reference.as_dict())

        wrong = ReferenceContract(
            model_id="different",
            dataset_id="tiny-data-v1",
            tokenizer_id="byte-v1",
            sequence_length=8,
        )
        with self.assertRaisesRegex(ResultValidationError, "contract mismatch"):
            run_submission(self.config(track="sample_efficiency", reference_contract=wrong))

    def test_none_retention_deletes_only_after_evaluator(self) -> None:
        seen_during_validation: list[bool] = []

        def evaluator(checkpoint: Path, payload: object) -> None:
            seen_during_validation.append(checkpoint.exists())

        outcome = run_submission(
            self.config(checkpoint_retention="none-after-validation"), evaluator=evaluator
        )
        self.assertEqual(seen_during_validation, [True])
        self.assertIsNone(outcome.checkpoint_path)
        self.assertFalse((outcome.run_dir / "model.npz").exists())
        self.assertFalse(outcome.record["checkpoint"]["retained"])
        self.assertEqual(len(outcome.record["checkpoint"]["sha256"]), 64)

    def test_qualifying_retention_removes_nonqualifying_checkpoint(self) -> None:
        outcome = run_submission(self.config(target_loss=2.0))
        self.assertFalse(outcome.record["qualified"])
        self.assertIsNone(outcome.checkpoint_path)
        self.assertFalse(outcome.record["checkpoint"]["retained"])

    def test_research_run_can_explicitly_omit_checkpoint_and_be_reverified(self) -> None:
        outcome = run_submission(
            self.config(
                passthrough_args=("--omit-checkpoint",),
                profile="dev",
                require_checkpoint=False,
            )
        )
        self.assertIsNone(outcome.checkpoint_path)
        self.assertIsNone(outcome.record["checkpoint"])
        validated = verify_run(
            outcome.run_dir,
            track="open",
            require_checkpoint=False,
        )
        self.assertIsNone(validated.checkpoint_path)
        with self.assertRaisesRegex(ResultValidationError, "checkpoint is required"):
            verify_run(outcome.run_dir, track="open")

    def test_fixed_training_token_budget_is_enforced_and_recorded(self) -> None:
        outcome = run_submission(self.config(expected_training_tokens=96))
        self.assertEqual(outcome.record["constraints"]["training_tokens"], 96)
        with self.assertRaisesRegex(ResultValidationError, "training-token budget"):
            run_submission(self.config(expected_training_tokens=95))
        with self.assertRaisesRegex(ConfigurationError, "expected_training_tokens"):
            run_submission(self.config(expected_training_tokens=0))

    def test_rejects_submission_and_checkpoint_path_traversal(self) -> None:
        with self.assertRaises(ConfigurationError):
            run_submission(self.config(submission="../tiny"))

        run_dir = self.root / "manual"
        run_dir.mkdir()
        outside = self.root / "outside.npz"
        outside.write_bytes(b"outside")
        payload = {
            "schema_version": 1,
            "status": "ok",
            "checkpoint": "../outside.npz",
            "metrics": {
                "train_seconds": 1.0,
                "tokens_processed": 1,
                "validation_loss": 1.0,
            },
        }
        with self.assertRaisesRegex(ResultValidationError, "escapes"):
            validate_result(payload, run_dir=run_dir, track="open")


class ProtocolAndScoringTests(unittest.TestCase):
    def test_evaluation_schema_rejects_inconsistent_fresh10_aggregates(self) -> None:
        evaluations = HarnessRunTests.fresh10_evaluations()
        expected = {
            name: row["scored_tokens"]
            for name, row in evaluations["fresh10"]["domains"].items()  # type: ignore[index,union-attr]
        }
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "model.npz").write_bytes(b"checkpoint")
            base_payload = {
                "schema_version": 1,
                "status": "ok",
                "checkpoint": "model.npz",
                "metrics": {
                    "train_seconds": 1.0,
                    "tokens_processed": 1,
                    "validation_loss": 2.5,
                    "validation_tokens": 64,
                },
                "evaluations": evaluations,
            }
            validated = validate_result(
                base_payload,
                run_dir=run_dir,
                track="open",
                expected_validation_tokens=64,
                expected_downstream_tokens=expected,
            )
            self.assertEqual(validated.evaluations, evaluations)

            mutations = {
                "fineweb canonical": lambda value: value["evaluations"]["fineweb"].update(
                    canonical=False
                ),
                "fineweb loss": lambda value: value["evaluations"]["fineweb"].update(
                    loss=2.4
                ),
                "macro loss": lambda value: value["evaluations"]["fresh10"].update(
                    macro_loss=1.0
                ),
                "macro perplexity": lambda value: value["evaluations"]["fresh10"].update(
                    macro_perplexity=1.0
                ),
                "total tokens": lambda value: value["evaluations"]["fresh10"].update(
                    scored_tokens=81_919
                ),
                "domain tokens": lambda value: next(
                    iter(value["evaluations"]["fresh10"]["domains"].values())
                ).update(scored_tokens=8_191),
                "nonfinite seconds": lambda value: next(
                    iter(value["evaluations"]["fresh10"]["domains"].values())
                ).update(seconds=float("inf")),
                "nonpositive perplexity": lambda value: next(
                    iter(value["evaluations"]["fresh10"]["domains"].values())
                ).update(perplexity=0.0),
                "inconsistent perplexity": lambda value: next(
                    iter(value["evaluations"]["fresh10"]["domains"].values())
                ).update(perplexity=42.0),
                "inconsistent total seconds": lambda value: value["evaluations"][
                    "fresh10"
                ].update(seconds=42.0),
            }
            for label, mutate in mutations.items():
                with self.subTest(label=label):
                    candidate = json.loads(json.dumps(base_payload))
                    mutate(candidate)
                    with self.assertRaises(ResultValidationError):
                        validate_result(
                            candidate,
                            run_dir=run_dir,
                            track="open",
                            expected_validation_tokens=64,
                            expected_downstream_tokens=expected,
                        )

    def test_fresh10_domain_names_must_match_expected_mapping(self) -> None:
        evaluations = HarnessRunTests.fresh10_evaluations()
        expected = {
            name: row["scored_tokens"]
            for name, row in evaluations["fresh10"]["domains"].items()  # type: ignore[index,union-attr]
        }
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "model.npz").write_bytes(b"checkpoint")
            domains = evaluations["fresh10"]["domains"]  # type: ignore[index]
            domains["unexpected"] = domains.pop("science")  # type: ignore[union-attr]
            payload = {
                "schema_version": 1,
                "status": "ok",
                "checkpoint": "model.npz",
                "metrics": {
                    "train_seconds": 1.0,
                    "tokens_processed": 1,
                    "validation_loss": 2.5,
                },
                "evaluations": evaluations,
            }
            with self.assertRaisesRegex(ResultValidationError, "domain names"):
                validate_result(
                    payload,
                    run_dir=run_dir,
                    track="open",
                    expected_downstream_tokens=expected,
                )

    def test_empty_leaderboard_renders(self) -> None:
        rendered = render_leaderboard([], track="open")
        self.assertIn("No qualifying runs.", rendered)

    def test_result_must_be_final_line_and_finite(self) -> None:
        with self.assertRaises(ResultValidationError):
            parse_result_line('SPEEDRUN_RESULT={"schema_version":1}\nlate log\n')
        with self.assertRaises(ResultValidationError):
            parse_result_line("SPEEDRUN_RESULT={not json}\n")

    def test_optional_implementation_provenance_must_be_an_object(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "model.npz").write_bytes(b"checkpoint")
            payload = {
                "schema_version": 1,
                "status": "ok",
                "checkpoint": "model.npz",
                "implementation": ["not", "an", "object"],
                "metrics": {
                    "train_seconds": 1.0,
                    "tokens_processed": 1,
                    "validation_loss": 1.0,
                },
            }
            with self.assertRaisesRegex(ResultValidationError, "implementation"):
                validate_result(payload, run_dir=run_dir, track="open")

    def test_track_rankings_use_canonical_fields(self) -> None:
        def record(name: str, seconds: float, tokens: int) -> dict[str, object]:
            return {
                "run_id": name + "-run",
                "submission": name,
                "status": "ok",
                "qualified": True,
                "track": "sample_efficiency",
                "profile": "tiny",
                "timing": {"observed_wall_seconds": seconds / 10},
                "metrics": {"train_seconds": seconds, "tokens_processed": tokens, "validation_loss": 2.0},
            }

        ranked = rank_records(
            [record("fast-many", 1.0, 20), record("slow-few", 9.0, 10), record("fast-few", 2.0, 10)],
            track="sample_efficiency",
            profile="tiny",
        )
        self.assertEqual([item["submission"] for item in ranked], ["fast-few", "slow-few", "fast-many"])
        rendered = render_leaderboard(ranked, track="sample_efficiency")
        self.assertIn("Sample Efficiency leaderboard", rendered)
        self.assertIn("fast-few", rendered)

    def test_open_ranking_uses_synchronized_time_not_wall_time(self) -> None:
        records = [
            {
                "run_id": "a-run",
                "submission": "a",
                "status": "ok",
                "qualified": True,
                "track": "open",
                "profile": "tiny",
                "timing": {"observed_wall_seconds": 100.0},
                "metrics": {"train_seconds": 1.0, "tokens_processed": 8, "validation_loss": 2.0},
            },
            {
                "run_id": "b-run",
                "submission": "b",
                "status": "ok",
                "qualified": True,
                "track": "open",
                "profile": "tiny",
                "timing": {"observed_wall_seconds": 1.0},
                "metrics": {"train_seconds": 2.0, "tokens_processed": 8, "validation_loss": 2.0},
            },
        ]
        ranked = rank_records(records, track="open", profile="tiny")
        self.assertEqual([item["submission"] for item in ranked], ["a", "b"])

    def test_leaderboard_can_requalify_records_for_one_shared_target(self) -> None:
        records = [
            {
                "run_id": "loose-run",
                "submission": "loose",
                "status": "ok",
                "qualified": True,
                "track": "open",
                "profile": "official",
                "metrics": {
                    "train_seconds": 1.0,
                    "tokens_processed": 8,
                    "validation_loss": 4.0,
                },
            },
            {
                "run_id": "strict-run",
                "submission": "strict",
                "status": "ok",
                "qualified": False,
                "track": "open",
                "profile": "official",
                "metrics": {
                    "train_seconds": 2.0,
                    "tokens_processed": 8,
                    "validation_loss": 3.27,
                },
            },
        ]
        ranked = rank_records(
            records, track="open", profile="official", target_loss=3.28
        )
        self.assertEqual([item["submission"] for item in ranked], ["strict"])


if __name__ == "__main__":
    unittest.main()
