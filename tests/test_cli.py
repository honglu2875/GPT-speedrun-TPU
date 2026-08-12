from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from speedrun import cli
from speedrun.config import ConfigError, LocalConfig
from speedrun.data import Fresh10Domain, PreparedDataset, PreparedFresh10
from speedrun.report import REPORT_ADMISSION_QUALIFICATION_LOSS


class CliTests(unittest.TestCase):
    def test_official_open_budget_preserves_calibrated_baseline(self) -> None:
        budget = cli.OFFICIAL_OPEN_TRAINING_TOKENS
        self.assertEqual(budget, 624_984_064)
        self.assertEqual(budget // (32 * 1024), 19_073)
        self.assertEqual(budget % (32 * 1024), 0)

    def test_sample_efficiency_contract_pins_semantic_vocabulary(self) -> None:
        smoke = cli._reference_contract("smoke")
        self.assertEqual(smoke.dataset_id, "smoke")
        self.assertEqual(smoke.tokenizer_id, "synthetic-byte-v1")
        contract = cli._reference_contract("official")
        self.assertEqual(contract.extra["model"]["vocab_size"], 50_304)
        self.assertEqual(
            contract.extra["model"]["semantic_vocab_size"], 50_304
        )

    def test_reserved_trainer_arguments_cannot_override_harness(self) -> None:
        for arguments in (
            ["--seed", "7"],
            ["--config", "/tmp/other.yaml"],
            ["--conf=/tmp/other.yaml"],
            ["--profile=smoke"],
            ["--output-dir", "/tmp/elsewhere"],
            ["--train-data", "other.bin"],
            ["--color=always"],
            ["--out", "/tmp/elsewhere"],
            ["--prof=smoke"],
            ["--data-f", "raw"],
            ["--downstream-manifest", "other.json"],
            ["--down", "other.json"],
        ):
            with self.subTest(arguments=arguments), self.assertRaisesRegex(
                ConfigError, "harness-controlled|controlled by the harness"
            ):
                cli._reject_reserved_trainer_args(arguments)

        cli._reject_reserved_trainer_args(["--steps", "20", "--batch-size=32"])

    def test_clone_copies_submission_config_byte_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "submissions" / "source"
            source.mkdir(parents=True)
            (source / "train.py").write_text("print('train')\n", encoding="utf-8")
            config_bytes = b"steps: 20\r\nlearning_rate: 3.0e-4\r\n"
            (source / "config.yaml").write_bytes(config_bytes)
            (source / "README.md").write_text("# Source\n", encoding="utf-8")
            args = cli.build_parser().parse_args(["clone", "source", "variant"])

            with patch("speedrun.cli.repo_root", return_value=root):
                self.assertEqual(cli.command_clone(args), 0)

            destination = root / "submissions" / "variant"
            self.assertEqual((destination / "config.yaml").read_bytes(), config_bytes)
            self.assertEqual(
                (destination / "train.py").read_text(encoding="utf-8"),
                "print('train')\n",
            )
            self.assertTrue((destination / "README.md").is_file())

    def test_clone_requires_config_before_creating_destination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "submissions" / "source"
            source.mkdir(parents=True)
            (source / "train.py").write_text("print('train')\n", encoding="utf-8")
            args = cli.build_parser().parse_args(["clone", "source", "variant"])

            with patch("speedrun.cli.repo_root", return_value=root):
                with self.assertRaisesRegex(ConfigError, "configuration does not exist"):
                    cli.command_clone(args)

            self.assertFalse((root / "submissions" / "variant").exists())

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

    def test_report_admission_has_no_customization_surface(self) -> None:
        self.assertEqual(REPORT_ADMISSION_QUALIFICATION_LOSS, 3.76)
        report = cli.build_parser().parse_args(["report"])
        self.assertFalse(hasattr(report, "admission_loss"))
        self.assertFalse(hasattr(LocalConfig(), "report_admission_loss"))
        prepare = cli.build_parser().parse_args(["prepare"])
        target_action = next(
            action for action in cli.build_parser()._subparsers._group_actions[0]
            .choices["prepare"]._actions
            if action.dest == "target_loss"
        )
        self.assertIn("smoke/development", target_action.help)
        self.assertIsNone(prepare.target_loss)
        with self.assertRaises(SystemExit):
            cli.build_parser().parse_args(["report", "--admission-loss", "3.9"])

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

    def test_fresh10_provenance_records_stable_domain_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "data" / "manifests" / "fresh10.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("{}", encoding="utf-8")
            prepared = PreparedDataset(
                name="tiny",
                root=Path("/dev/shm"),
                manifest_path=manifest,
                manifest_sha256="a" * 64,
                train_files=(Path("/dev/shm/train.bin"),),
                validation_files=(Path("/dev/shm/val.bin"),),
                train_tokens=10,
                validation_tokens=10,
                validation_prefix_tokens=8,
            )
            domain = Fresh10Domain(
                name="science",
                path=Path("/dev/shm/fresh10-science.bin"),
                token_count=8196,
                scored_tokens=8192,
                sha256="b" * 64,
                documents=(),
            )
            fresh10 = PreparedFresh10(
                name="fresh10-v1",
                root=Path("/dev/shm"),
                manifest_path=manifest,
                manifest_sha256="c" * 64,
                domains=(domain,),
            )
            provenance = cli._data_provenance(
                prepared,
                profile="official",
                integrity="sha256",
                repo=root,
                fresh10=fresh10,
            )["fresh10"]
        self.assertEqual(provenance["scored_tokens"], 8192)
        self.assertEqual(provenance["domains"]["science"]["sha256"], "b" * 64)
        self.assertNotIn("/dev/shm", str(provenance))

    def test_verify_recovers_recorded_fresh10_contract(self) -> None:
        record = {
            "provenance": {
                "fresh10": {
                    "domains": {
                        "science": {"scored_tokens": 8_192},
                        "legal": {"scored_tokens": 4_096},
                    }
                }
            }
        }
        self.assertEqual(
            cli._recorded_downstream_tokens(record),
            {"science": 8_192, "legal": 4_096},
        )
        self.assertIsNone(cli._recorded_downstream_tokens({"provenance": {}}))
        with self.assertRaisesRegex(cli.HarnessError, "invalid domain row"):
            cli._recorded_downstream_tokens(
                {"provenance": {"fresh10": {"domains": {"science": {"scored_tokens": 0}}}}}
            )

    def test_verify_recovers_training_budget_without_retroactive_default(self) -> None:
        self.assertIsNone(cli._recorded_training_tokens({}))
        self.assertIsNone(
            cli._recorded_training_tokens(
                {"constraints": {"training_tokens": None}}
            )
        )
        self.assertEqual(
            cli._recorded_training_tokens(
                {"constraints": {"training_tokens": 624_984_064}}
            ),
            624_984_064,
        )
        with self.assertRaisesRegex(cli.HarnessError, "training-token"):
            cli._recorded_training_tokens(
                {"constraints": {"training_tokens": 0}}
            )


if __name__ == "__main__":
    unittest.main()
