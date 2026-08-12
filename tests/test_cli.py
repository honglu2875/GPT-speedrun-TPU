from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from speedrun import cli
from speedrun.config import ConfigError, LocalConfig
from speedrun.data import PreparedDataset


class CliTests(unittest.TestCase):
    def test_reserved_trainer_arguments_cannot_override_harness(self) -> None:
        for arguments in (
            ["--seed", "7"],
            ["--profile=smoke"],
            ["--output-dir", "/tmp/elsewhere"],
            ["--train-data", "other.bin"],
            ["--color=always"],
            ["--out", "/tmp/elsewhere"],
            ["--prof=smoke"],
            ["--data-f", "raw"],
        ):
            with self.subTest(arguments=arguments), self.assertRaisesRegex(
                ConfigError, "harness-controlled|controlled by the harness"
            ):
                cli._reject_reserved_trainer_args(arguments)

        cli._reject_reserved_trainer_args(["--steps", "20", "--batch-size=32"])

    def test_non_run_unknown_arguments_are_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            cli.main(["doctor", "--not-a-doctor-flag"])

    def test_official_target_is_versioned_and_can_only_be_tightened(self) -> None:
        self.assertEqual(
            cli._effective_target_loss(
                "official", requested=None, development_default=99.0
            ),
            3.28,
        )
        self.assertEqual(
            cli._effective_target_loss(
                "official", requested=3.26, development_default=99.0
            ),
            3.26,
        )
        with self.assertRaisesRegex(ConfigError, "may not be easier"):
            cli._effective_target_loss(
                "official", requested=3.29, development_default=3.28
            )
        self.assertEqual(
            cli._effective_target_loss("dev", requested=None, development_default=4.0),
            4.0,
        )

    def test_wizard_accepts_defaults_and_returns_complete_config(self) -> None:
        defaults = LocalConfig()
        # Two path prompts, five menu prompts, one target prompt, and four
        # confirmations. Empty input accepts every displayed default.
        with patch("builtins.input", side_effect=[""] * 12):
            result, diagnostics, require_tpu, download, save = cli._prepare_wizard(
                defaults,
                run_diagnostics=True,
                require_tpu=True,
                download=True,
                save=True,
            )
        self.assertEqual(result, defaults)
        self.assertTrue(diagnostics)
        self.assertTrue(require_tpu)
        self.assertTrue(download)
        self.assertTrue(save)

    def test_dataset_provenance_uses_stable_names_not_cache_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "data" / "manifests" / "tiny.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("{}", encoding="utf-8")
            prepared = PreparedDataset(
                name="tiny",
                root=Path("/dev/shm"),
                manifest_path=manifest,
                manifest_sha256="a" * 64,
                train_files=(Path("/dev/shm/train-1.bin"),),
                validation_files=(Path("/dev/shm/val.bin"),),
                train_tokens=100,
                validation_tokens=20,
                validation_prefix_tokens=16,
            )
            provenance = cli._data_provenance(
                prepared, profile="dev", integrity="sha256", repo=root
            )["dataset"]
        self.assertEqual(provenance["manifest"]["path"], "data/manifests/tiny.json")
        self.assertEqual(
            provenance["manifest"]["sha256"],
            "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
        )
        self.assertEqual(provenance["manifest"]["canonical_sha256"], "a" * 64)
        self.assertEqual(provenance["train_files"], ["train-1.bin"])
        self.assertEqual(provenance["validation_prefix_tokens"], 16)
        self.assertNotIn("/dev/shm", str(provenance))


if __name__ == "__main__":
    unittest.main()
