from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml

from speedrun.data import FRESH10_DOMAINS, manifest_digest
from speedrun.scaling import (
    DEFAULT_SUITE,
    LearningRateEdgeError,
    ScalingError,
    _fit_slice,
    _manifest_shard_contract,
    _next_adaptive_calibration,
    _public_point,
    _read_run,
    _warrants_high_side_extension,
    build_parser,
    load_suite,
    materialize_configs,
    parameter_count,
    select_learning_rate,
    trainer_command,
    validate_data_directory,
    validate_fresh10_directory,
    validate_runtime_environment,
    _runtime_inventory_in_current_process,
    variant_config_bytes,
)


TRAINER_PATH = Path(__file__).parents[1] / "submissions" / "reference" / "train.py"
TRAINER_SPEC = importlib.util.spec_from_file_location(
    "scaling_reference_train", TRAINER_PATH
)
if TRAINER_SPEC is None or TRAINER_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"could not import {TRAINER_PATH}")
reference_trainer = importlib.util.module_from_spec(TRAINER_SPEC)
sys.modules[TRAINER_SPEC.name] = reference_trainer
TRAINER_SPEC.loader.exec_module(reference_trainer)


class ScalingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.suite = load_suite(DEFAULT_SUITE)

    def test_suite_matches_exact_current_budget_and_cost(self) -> None:
        suite = self.suite
        self.assertEqual(suite["anchor"]["total_flops"], 537_549_813_420_392_448)
        self.assertEqual(
            [item["multiplier"] for item in suite["compute_slices"]],
            [0.25, 0.5, 1.0],
        )
        self.assertEqual(len(suite["fit_shapes"]), 5)
        self.assertEqual(len(suite["calibrations"]), 15)
        self.assertEqual(len(suite["variants"]), 10)
        self.assertEqual(len(suite["controls"]), 1)
        self.assertEqual(len(suite["adaptive_calibrations"]), 28)
        self.assertEqual(len(suite["optional_extensions"]), 4)
        self.assertEqual(suite["required_train_tokens"], 3_463_512_064)
        self.assertEqual(suite["validation_tokens"], 99_975_168)
        self.assertEqual(suite["dataset"]["id"], "fineweb-4b-gpt2")
        self.assertEqual(
            [item["parameters"] for item in suite["fit_shapes"]],
            [22_590_400, 30_357_504, 39_895_744, 51_500_032, 65_465_280],
        )
        for point in suite["all_variants"]:
            target = suite["slices_by_id"][point["slice"]]["target_total_flops"]
            self.assertLess(abs(point["total_flops"] / target - 1.0), 1e-4)
            self.assertEqual(point["train_tokens"], point["steps"] * 32 * 1024)
        base_cost = 15 * 0.25 + 5 * 0.5 + 5 * 1.0 + 1.0
        self.assertEqual(base_cost, 12.25)
        self.assertEqual(suite["runtime"]["device_kind"], "TPU v4")
        self.assertEqual(
            suite["dataset"]["source_inventory_sha256"],
            "02ddc6361cc2f8a3d23b0d8b823c7eb7e2b1663ad3d0eff63e83b373456fc12b",
        )
        self.assertEqual(
            suite["dataset"]["exclusion_policy_sha256"],
            "ab25cabd0781b1046b7ad7b281b4147ff6e27d36977f4e842b8c92573399ad77",
        )
        self.assertEqual(
            suite["dataset"]["preparation_core_sha256"],
            "4bbdcb76da837276f6f337b805d37a74e3272b476e01fd198f416097abe19241",
        )

    def test_parameter_formula_matches_reference_anchor(self) -> None:
        self.assertEqual(
            parameter_count(layers=12, d_model=768, vocab_size=50_304, seq_len=1024),
            124_475_904,
        )

    def test_generated_configs_are_static_strict_and_immutable(self) -> None:
        point = self.suite["calibrations"][0]
        payload = variant_config_bytes(self.suite, point)
        config = yaml.safe_load(payload)
        dev = config["profiles"]["dev"]
        self.assertEqual(dev["training"]["sampling"], "shuffled_epochs")
        self.assertEqual(dev["training"]["train_tokens"], point["train_tokens"])
        self.assertEqual(dev["model"]["d_model"], point["d_model"])
        self.assertEqual(dev["optimizer"]["learning_rate"], point["learning_rate"])
        self.assertEqual(
            config["profiles"]["official"]["training"]["sampling"],
            "random_windows",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = materialize_configs(self.suite, root, [point["id"]])
            self.assertEqual(paths[0].read_bytes(), payload)
            paths[0].write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ScalingError, "immutable file differs"):
                materialize_configs(self.suite, root, [point["id"]])

    def test_exact_builder_manifest_contract_rejects_inventory_drift(self) -> None:
        payload = _fake_manifest(self.suite)
        train, validation = _manifest_shard_contract(payload, self.suite)
        self.assertEqual(len(train), 39)
        self.assertEqual(len(validation), 1)

        wrong_name = json.loads(json.dumps(payload))
        wrong_name["name"] = "fineweb-2b-gpt2"
        with self.assertRaisesRegex(ScalingError, "manifest name"):
            _manifest_shard_contract(wrong_name, self.suite)

        missing_hash = json.loads(json.dumps(payload))
        del missing_hash["files"][7]["sha256"]
        with self.assertRaisesRegex(ScalingError, "SHA-256 is required"):
            _manifest_shard_contract(missing_hash, self.suite)

        reordered = json.loads(json.dumps(payload))
        reordered["files"][1], reordered["files"][2] = (
            reordered["files"][2],
            reordered["files"][1],
        )
        with self.assertRaisesRegex(ScalingError, "must list validation shard"):
            _manifest_shard_contract(reordered, self.suite)

    def test_data_preflight_requests_full_hashes_and_rejects_extra_binary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = _fake_manifest(self.suite)
            (root / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
            for entry in payload["files"]:
                (root / entry["path"]).touch()
            prepared = SimpleNamespace(
                name=self.suite["dataset"]["id"],
                train_files=tuple(
                    root / item["path"]
                    for item in payload["files"]
                    if item["split"] == "train"
                ),
                validation_files=(root / payload["files"][0]["path"],),
            )
            with patch(
                "speedrun.scaling.verify_dataset", return_value=prepared
            ) as verify, patch(
                "speedrun.scaling._validate_production_provenance",
                return_value=_fake_production_provenance(self.suite),
            ):
                inventory = validate_data_directory(root, self.suite)
                verify.assert_called_once_with(
                    root / "manifest.json", root, train_shards=39, verify_hash=True
                )
                self.assertEqual(len(inventory["shards"]), 40)
                self.assertEqual(
                    inventory["manifest_canonical_sha256"],
                    manifest_digest(payload),
                )

                (root / "unexpected.bin").touch()
                with self.assertRaisesRegex(ScalingError, "extra=.*unexpected.bin"):
                    validate_data_directory(root, self.suite)

    def test_trainer_command_is_consumed_by_real_parser_without_duplicate_flags(self) -> None:
        command = trainer_command(
            python_executable=sys.executable,
            trainer=TRAINER_PATH,
            config=TRAINER_PATH.with_name("config.yaml"),
            output=Path("/tmp/isoflop-output"),
            seed=1337,
            data_path=Path("/tmp/fineweb-4b"),
            dataset_id="fineweb-4b-gpt2",
            color="never",
            downstream_manifest=Path("/tmp/fresh10.json"),
            downstream_root=Path("/tmp/fresh10"),
            attention_tuning_cache=Path("/tmp/attention.json"),
            autotune_attention=True,
        )
        self.assertEqual(command.count("--dataset-id"), 1)
        parsed = reference_trainer.build_parser().parse_args(command[2:])
        self.assertEqual(parsed.dataset_id, "fineweb-4b-gpt2")
        self.assertEqual(parsed.data_path, Path("/tmp/fineweb-4b"))
        self.assertEqual(parsed.profile, "dev")
        self.assertEqual(parsed.track, "open")
        self.assertTrue(parsed.omit_checkpoint)
        self.assertEqual(parsed.downstream_manifest, Path("/tmp/fresh10.json"))
        self.assertEqual(parsed.downstream_root, Path("/tmp/fresh10"))
        self.assertEqual(parsed.attention_tuning_cache, Path("/tmp/attention.json"))
        self.assertTrue(parsed.autotune_attention)

    def test_scaling_run_requires_fresh10_paths(self) -> None:
        parser = build_parser()
        with patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                parser.parse_args(
                    [
                        "run",
                        "--data-path",
                        "/tmp/data",
                        "--confirm-execution-fingerprint",
                        "digest",
                        "--staged",
                    ]
                )

    def test_exact_v4_8_runtime_gate(self) -> None:
        devices = [
            SimpleNamespace(device_kind="TPU v4", platform="tpu", id=i, process_index=0)
            for i in range(4)
        ]
        versions = {
            "jax": "0.11.0",
            "jaxlib": "0.11.0",
            "libtpu": "0.0.44.1",
        }
        with patch(
            "speedrun.scaling.host_platform.python_version", return_value="3.12.8"
        ), patch(
            "speedrun.scaling.importlib_metadata.version",
            side_effect=lambda name: versions[name],
        ), patch("jax.devices", return_value=devices), patch(
            "jax.local_devices", return_value=devices
        ), patch("jax.process_count", return_value=1), patch(
            "jax.device_count", return_value=4
        ), patch("jax.local_device_count", return_value=4):
            runtime = _runtime_inventory_in_current_process(self.suite)
        self.assertEqual(runtime["device_ids"], [0, 1, 2, 3])
        devices[3].device_kind = "TPU v5p"
        with patch(
            "speedrun.scaling.host_platform.python_version", return_value="3.12.8"
        ), patch(
            "speedrun.scaling.importlib_metadata.version",
            side_effect=lambda name: versions[name],
        ), patch("jax.devices", return_value=devices), patch(
            "jax.local_devices", return_value=devices
        ), patch("jax.process_count", return_value=1), patch(
            "jax.device_count", return_value=4
        ), patch("jax.local_device_count", return_value=4):
            with self.assertRaisesRegex(ScalingError, "exactly one-process TPU v4-8"):
                _runtime_inventory_in_current_process(self.suite)

    def test_runtime_gate_uses_an_isolated_process(self) -> None:
        expected = {
            "python_version": "3.12.13",
            "jax_version": "0.11.0",
            "jaxlib_version": "0.11.0",
            "libtpu_version": "0.0.44.1",
            "platform": "tpu",
            "device_count": 4,
            "local_device_count": 4,
            "process_count": 1,
            "device_kinds": ["TPU v4"],
            "device_ids": [0, 1, 2, 3],
            "process_indices": [0, 0, 0, 0],
        }
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(expected), stderr=""
        )
        with patch(
            "speedrun.scaling._runtime_inventory_in_current_process",
            side_effect=AssertionError("runtime discovery must stay out of the parent"),
        ), patch("speedrun.scaling.subprocess.run", return_value=completed) as run:
            self.assertEqual(validate_runtime_environment(self.suite), expected)
        command = run.call_args.args[0]
        self.assertEqual(
            command,
            [
                sys.executable,
                "-m",
                "speedrun.scaling",
                "--suite",
                str(self.suite["path"]),
                "--internal-runtime-probe",
            ],
        )
        self.assertEqual(run.call_args.kwargs["cwd"], Path(__file__).parents[1])
        self.assertIs(run.call_args.kwargs["stdout"], subprocess.PIPE)
        self.assertIs(run.call_args.kwargs["stderr"], subprocess.PIPE)
        self.assertIs(run.call_args.kwargs["text"], True)
        self.assertIs(run.call_args.kwargs["check"], False)
        self.assertEqual(run.call_args.kwargs["timeout"], 60.0)

    def test_fresh10_preflight_hash_verifies_exact_ten_domains(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            domains = []
            for name, entry in zip(
                FRESH10_DOMAINS,
                self.suite["fresh10"]["payload"]["domains"],
                strict=True,
            ):
                path = root / entry["path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
                domains.append(
                    SimpleNamespace(
                        name=name,
                        path=path,
                        scored_tokens=8_192,
                        token_count=entry["tokens"],
                        sha256=entry["sha256"],
                        documents=(1, 2, 3, 4),
                    )
                )
            prepared = SimpleNamespace(
                name=self.suite["fresh10"]["name"],
                domains=tuple(domains),
                scored_tokens=81_920,
                manifest_sha256=self.suite["fresh10"][
                    "manifest_canonical_sha256"
                ],
            )
            with patch(
                "speedrun.scaling.verify_fresh10", return_value=prepared
            ) as verify:
                inventory = validate_fresh10_directory(
                    self.suite["fresh10"]["manifest_path"], root, self.suite
                )
            verify.assert_called_once_with(
                root,
                manifest=self.suite["fresh10"]["manifest_path"],
                verify_hash=True,
            )
            self.assertEqual(len(inventory["domains"]), 10)
            self.assertEqual(inventory["scored_tokens"], 81_920)

    def test_quadratic_fit_brackets_interior_and_refuses_edge_minimum(self) -> None:
        parameters = [22e6, 30e6, 40e6, 52e6, 65e6]

        def points(losses: list[float]) -> list[dict[str, float | int | str]]:
            return [
                {
                    "id": f"p{index}",
                    "parameters": int(parameters[index]),
                    "train_tokens": int(1e18 / parameters[index]),
                    "validation_loss": loss,
                }
                for index, loss in enumerate(losses)
            ]

        bracketed, _ = _fit_slice(
            points([4.2, 3.9, 3.7, 3.8, 4.1]),
            slice_id="test",
            target_total_flops=1,
        )
        self.assertTrue(bracketed["bracketed"])
        self.assertIsNotNone(bracketed["interpolated_optimum"])

        edge, _ = _fit_slice(
            points([4.2, 4.0, 3.8, 3.6, 3.4]),
            slice_id="test",
            target_total_flops=1,
        )
        self.assertFalse(edge["bracketed"])
        self.assertTrue(edge["observed_best_at_endpoint"])
        self.assertIsNone(edge["interpolated_optimum"])
        self.assertTrue(_warrants_high_side_extension(edge))

        low_edge, _ = _fit_slice(
            points([3.4, 3.6, 3.8, 4.0, 4.2]),
            slice_id="test",
            target_total_flops=1,
        )
        self.assertFalse(_warrants_high_side_extension(low_edge))

    def test_learning_rate_selection_uses_lowest_loss_and_is_immutable(self) -> None:
        shape_id = self.suite["fit_shapes"][0]["shape_id"]
        candidates = [
            item for item in self.suite["calibrations"] if item["shape_id"] == shape_id
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for candidate, loss in zip(candidates, (3.2, 3.1, 3.3), strict=True):
                _write_fake_run(root, self.suite, candidate, loss)
            selected = select_learning_rate(
                self.suite, shape_id=shape_id, runs_path=root
            )
            self.assertEqual(selected["selected_learning_rate"], 0.0003)
            selection_path = root / "learning-rate-selections" / f"{shape_id}.json"
            before = selection_path.read_bytes()
            selected_again = select_learning_rate(
                self.suite, shape_id=shape_id, runs_path=root
            )
            self.assertEqual(selected_again, selected)
            self.assertEqual(selection_path.read_bytes(), before)

    def test_learning_rate_selection_refuses_mixed_dataset_hashes(self) -> None:
        shape_id = self.suite["fit_shapes"][0]["shape_id"]
        candidates = [
            item for item in self.suite["calibrations"] if item["shape_id"] == shape_id
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, candidate in enumerate(candidates):
                provenance = _fake_dataset_provenance(self.suite)
                if index == 2:
                    provenance["shards"][4]["sha256"] = "b" * 64
                    provenance["manifest_canonical_sha256"] = "c" * 64
                _write_fake_run(
                    root,
                    self.suite,
                    candidate,
                    3.0 + index / 10,
                    provenance=provenance,
                )
            with self.assertRaisesRegex(ScalingError, "one canonical dataset"):
                select_learning_rate(self.suite, shape_id=shape_id, runs_path=root)

    def test_learning_rate_selection_refuses_an_edge_winner(self) -> None:
        shape_id = self.suite["fit_shapes"][0]["shape_id"]
        candidates = [
            item for item in self.suite["calibrations"] if item["shape_id"] == shape_id
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for candidate, loss in zip(candidates, (3.0, 3.1, 3.2), strict=True):
                _write_fake_run(root, self.suite, candidate, loss)
            with self.assertRaises(LearningRateEdgeError) as raised:
                select_learning_rate(self.suite, shape_id=shape_id, runs_path=root)
            self.assertEqual(raised.exception.side, "lower")

    def test_bounded_lr_expansion_selects_next_geometric_point(self) -> None:
        shape_id = self.suite["fit_shapes"][0]["shape_id"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial = [
                item
                for item in self.suite["calibrations"]
                if item["shape_id"] == shape_id
            ]
            for point in initial:
                result = root / point["id"] / "artifacts" / "result.json"
                result.parent.mkdir(parents=True)
                result.touch()
            lower = _next_adaptive_calibration(
                self.suite, shape_id, "lower", root
            )
            upper = _next_adaptive_calibration(
                self.suite, shape_id, "upper", root
            )
            self.assertAlmostEqual(lower["learning_rate"], 0.0002 / 1.5)
            self.assertAlmostEqual(upper["learning_rate"], 0.00045 * 1.5)
            result = root / lower["id"] / "artifacts" / "result.json"
            result.parent.mkdir(parents=True)
            result.touch()
            next_lower = _next_adaptive_calibration(
                self.suite, shape_id, "lower", root
            )
            self.assertAlmostEqual(next_lower["learning_rate"], 0.0002 / 2.25)

    def test_result_fresh10_counts_are_release_gating(self) -> None:
        point = self.suite["calibrations"][0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_fake_run(root, self.suite, point, 3.1)
            result_path = root / point["id"] / "artifacts" / "result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["evaluations"]["fresh10"]["domains"][FRESH10_DOMAINS[0]][
                "scored_tokens"
            ] = 8_191
            result_path.write_text(json.dumps(result), encoding="utf-8")
            with self.assertRaisesRegex(ScalingError, "scored token count differs"):
                _read_run(self.suite, point, root)


def _fake_manifest(suite: dict) -> dict:
    files = [
        {
            "path": "fineweb_val_000000.bin",
            "split": "validation",
            "tokens": 100_000_000,
            "bytes": 200_001_024,
            "sha256": f"{0:064x}",
        }
    ]
    files.extend(
        {
            "path": f"fineweb_train_{index:06d}.bin",
            "split": "train",
            "tokens": 100_000_000,
            "bytes": 200_001_024,
            "sha256": f"{index:064x}",
        }
        for index in range(1, 40)
    )
    return {
        "schema_version": 1,
        "name": suite["dataset"]["id"],
        "source": {
            "dataset": suite["dataset"]["source_repository"],
            "revision": suite["dataset"]["source_revision"],
        },
        "tokenizer": {
            "name": "gpt2",
            "implementation": "tiktoken",
            "implementation_version": suite["dataset"]["tokenizer_version"],
            "document_prefix_token": 50_256,
            "vocab_size": 50_257,
        },
        "format": {
            "name": "llm.c-gpt2-v1",
            "header_bytes": 1_024,
            "header_dtype": "little-endian int32",
            "magic": 20_240_520,
            "version": 1,
            "token_dtype": "little-endian uint16",
        },
        "default_train_shards": 39,
        "validation_prefix_tokens": 100_000_000,
        "files": files,
    }


def _fake_dataset_provenance(suite: dict) -> dict:
    usable_per_shard = ((100_000_000 - 1) // 1024) * 1024
    manifest = _fake_manifest(suite)
    return {
        "name": suite["dataset"]["id"],
        "root": "/tmp/fineweb-4b",
        "manifest_path": "/tmp/fineweb-4b/manifest.json",
        "manifest_raw_sha256": "d" * 64,
        "manifest_canonical_sha256": "e" * 64,
        "usable_train_tokens": 39 * usable_per_shard,
        "usable_validation_tokens": usable_per_shard,
        "production": _fake_production_provenance(suite),
        "shards": [
            {
                key: entry[key]
                for key in ("path", "split", "tokens", "bytes", "sha256")
            }
            for entry in manifest["files"]
        ],
    }


def _fake_production_provenance(suite: dict) -> dict:
    return {
        "source_inventory_sha256": suite["dataset"]["source_inventory_sha256"],
        "source_inventory_raw_sha256": "1" * 64,
        "exclusion_policy_sha256": suite["dataset"]["exclusion_policy_sha256"],
        "exclusion_policy_raw_sha256": "2" * 64,
        "preparation_core_sha256": suite["dataset"]["preparation_core_sha256"],
        "build_plan_raw_sha256": "3" * 64,
        "builder_module_sha256": "4" * 64,
        "entrypoint_sha256": "5" * 64,
        "source_date_before": "2024-04-01",
        "validation_train_document_disjoint": True,
        "validation_boundary_discarded_tokens": 17,
        "validation_boundary_document_id_sha256": "6" * 64,
    }


def _fake_runtime(suite: dict) -> dict:
    return {
        "python_version": f"{suite['runtime']['python_major_minor']}.9",
        "jax_version": suite["runtime"]["jax_version"],
        "jaxlib_version": suite["runtime"]["jaxlib_version"],
        "libtpu_version": suite["runtime"]["libtpu_version"],
        "platform": "tpu",
        "device_count": 4,
        "local_device_count": 4,
        "process_count": 1,
        "device_kinds": ["TPU v4"],
        "device_ids": [0, 1, 2, 3],
        "process_indices": [0, 0, 0, 0],
    }


def _fake_fresh10_provenance(suite: dict) -> dict:
    return {
        "name": suite["fresh10"]["name"],
        "root": "/tmp/fresh10",
        "manifest_path": str(suite["fresh10"]["manifest_path"]),
        "manifest_raw_sha256": suite["fresh10"]["manifest_raw_sha256"],
        "manifest_canonical_sha256": suite["fresh10"][
            "manifest_canonical_sha256"
        ],
        "repository": suite["fresh10"]["repository"],
        "revision": suite["fresh10"]["revision"],
        "publication_not_before": suite["fresh10"]["publication_not_before"],
        "scored_tokens": suite["fresh10"]["scored_tokens"],
        "domains": [
            {
                "name": expected_name,
                "path": entry["path"],
                "tokens": entry["tokens"],
                "scored_tokens": 8_192,
                "bytes": entry["bytes"],
                "sha256": entry["sha256"],
                "documents": 4,
            }
            for expected_name, entry in zip(
                FRESH10_DOMAINS, suite["fresh10"]["payload"]["domains"], strict=True
            )
        ],
    }


def _write_fake_run(
    root: Path,
    suite: dict,
    point: dict,
    loss: float,
    *,
    provenance: dict | None = None,
) -> None:
    point_root = root / point["id"]
    work = point_root / "work"
    repo = Path(__file__).parents[1]
    source_targets = {
        "train.py": repo / "submissions" / "reference" / "train.py",
        **{
            relative: repo / relative
            for relative in suite["source_snapshot"]
            if relative == "speedrun/__init__.py"
            or relative.startswith("speedrun/kernels/")
        },
    }
    for relative, source in source_targets.items():
        destination = work / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    config_path = work / "config.yaml"
    config_path.write_bytes(variant_config_bytes(suite, point))
    work_snapshot = {
        relative: suite["source_snapshot"][
            "submissions/reference/train.py" if relative == "train.py" else relative
        ]
        for relative in source_targets
    }
    work_snapshot["config.yaml"] = point["config_sha256"]
    result = point_root / "artifacts" / "result.json"
    result.parent.mkdir(parents=True)
    result.write_text(
        json.dumps(_fake_result(suite, point, loss)), encoding="utf-8"
    )
    run_manifest = {
        "schema_version": 3,
        "classification": "diagnostic_noncompetition_isoflop",
        "suite_id": suite["suite_id"],
        "suite_sha256": suite["suite_sha256"],
        "execution_fingerprint": suite["execution_fingerprint"],
        "template_sha256": suite["template_sha256"],
        "source_snapshot": suite["source_snapshot"],
        "work_snapshot": work_snapshot,
        "point": _public_point(point),
        "trainer_sha256": suite["source_snapshot"]["submissions/reference/train.py"],
        "config_sha256": point["config_sha256"],
        "checkpoint_policy": "omit_research_checkpoint",
        "dataset": provenance or _fake_dataset_provenance(suite),
        "fresh10": _fake_fresh10_provenance(suite),
        "runtime": _fake_runtime(suite),
        "seed": 1337,
    }
    (point_root / "run-manifest.json").write_text(
        json.dumps(run_manifest), encoding="utf-8"
    )


def _fake_result(suite: dict, point: dict, loss: float) -> dict:
    downstream_loss = 4.0
    domain_seconds = 0.1
    fresh_domains = {
        name: {
            "loss": downstream_loss,
            "perplexity": math.exp(downstream_loss),
            "scored_tokens": 8_192,
            "seconds": domain_seconds,
        }
        for name in FRESH10_DOMAINS
    }
    usable_train = _fake_dataset_provenance(suite)["usable_train_tokens"]
    return {
        "schema_version": 1,
        "status": "ok",
        "profile": "dev",
        "track": "open",
        "seed": suite["seed"],
        "checkpoint": None,
        "system": {**_fake_runtime(suite), "controller_process_index": 0},
        "contract": {
            "model_id": "reference-gpt-v1",
            "dataset_id": suite["dataset"]["id"],
            "tokenizer_id": "gpt2",
            "sequence_length": suite["sequence_length"],
            "model": {
                "layers": point["layers"],
                "heads": point["heads"],
                "d_model": point["d_model"],
                "mlp_mult": 4,
                "vocab_size": suite["vocab_size"],
                "semantic_vocab_size": suite["vocab_size"],
                "tied_embeddings": True,
            },
        },
        "implementation": {
            "attention_backend": "tpu_flash",
            "loss_backend": "dense",
            "vocab_tile_size": 2_048,
            "configuration": {
                "sha256": point["config_sha256"],
                "resolved": {
                    "training": {
                        "steps": point["steps"],
                        "train_tokens": point["train_tokens"],
                        "batch_size": suite["batch_size"],
                        "seq_len": suite["sequence_length"],
                        "sampling": "shuffled_epochs",
                        "dtype": "bfloat16",
                    },
                    "model": {
                        "layers": point["layers"],
                        "heads": point["heads"],
                        "d_model": point["d_model"],
                        "mlp_mult": 4,
                        "vocab_size": suite["vocab_size"],
                        "semantic_vocab_size": suite["vocab_size"],
                        "tied_embeddings": True,
                    },
                    "kernels": {
                        "attention_backend": "tpu_flash",
                        "loss_backend": "dense",
                        "vocab_tile_size": 2_048,
                    },
                    "optimizer": {
                        "learning_rate": point["learning_rate"],
                        "min_lr_ratio": suite["optimizer"]["min_lr_ratio"],
                        "warmup_steps": point["warmup_steps"],
                        "weight_decay": suite["optimizer"]["weight_decay"],
                        "beta1": suite["optimizer"]["beta1"],
                        "beta2": suite["optimizer"]["beta2"],
                        "grad_clip": suite["optimizer"]["grad_clip"],
                    },
                    "evaluation": {
                        "eval_batches": suite["validation_batches"],
                        "val_every": point["val_every"],
                        "val_probe_batches": 8,
                    },
                    "logging": {
                        "diagnostics_every": point["diagnostics_every"],
                        "log_every": point["log_every"],
                    },
                },
                "schema_version": 1,
                "path": "config.yaml",
                "profile": "dev",
                "overrides": {},
            }
        },
        "evaluations": {
            "fineweb": {
                "loss": loss,
                "perplexity": math.exp(loss),
                "scored_tokens": suite["validation_tokens"],
                "seconds": 1.0,
                "canonical": True,
            },
            "fresh10": {
                "domains": fresh_domains,
                "macro_loss": downstream_loss,
                "macro_perplexity": math.exp(downstream_loss),
                "scored_tokens": 81_920,
                "seconds": math.fsum(
                    row["seconds"] for row in fresh_domains.values()
                ),
            },
        },
        "metrics": {
            "tokens_processed": point["train_tokens"],
            "training_token_budget": point["train_tokens"],
            "training_steps": point["steps"],
            "training_usable_tokens_per_epoch": usable_train,
            "parameters": point["parameters"],
            "flops_per_token": point["flops_per_token"],
            "estimated_total_flops": point["total_flops"],
            "training_sampling": "shuffled_epochs",
            "training_data_epochs": point["train_tokens"] / usable_train,
            "validation_tokens": suite["validation_tokens"],
            "validation_loss": loss,
            "train_seconds": 1.0,
        },
    }


if __name__ == "__main__":
    unittest.main()
