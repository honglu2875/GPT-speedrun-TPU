from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
import importlib.util
from io import StringIO
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
    def test_training_csv_contains_every_step(self) -> None:
        history = np.asarray(
            [[2.0, 1.0e-3, 0.5], [1.5, 5.0e-4, 0.25]], dtype=np.float32
        )
        config = SimpleNamespace(steps=2, batch_size=4, seq_len=8)
        with tempfile.TemporaryDirectory() as directory:
            trainer.write_training_csv(Path(directory), history, config)
            rows = (Path(directory) / trainer.TRAINING_CSV_NAME).read_text().splitlines()
        self.assertEqual(
            rows[0], "step,tokens_processed,train_loss,learning_rate,grad_norm"
        )
        self.assertEqual(len(rows), 3)
        self.assertTrue(rows[1].startswith("1,32,2.0,"))
        self.assertTrue(rows[2].startswith("2,64,1.5,"))

    def test_console_writes_only_to_stderr(self) -> None:
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            console = trainer.Console("never")
            console.banner()
            console.table("test", (("field", "value"),))
            console.phase("phase", "detail")
            console.step(1, 1, 1.25, 1.0e-3, 0.5, 1024.0)
            console.success(1.0, 0.25)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("GPT TPU SPEEDRUN", stderr.getvalue())
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


if __name__ == "__main__":
    unittest.main()
