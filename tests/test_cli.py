from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from speedrun import cli
from speedrun.config import ConfigError, LocalConfig
from speedrun.data import DataError, Fresh10Domain, PreparedDataset, PreparedFresh10
from speedrun.doctor import check_prepared_data
from speedrun.report import REPORT_ADMISSION_QUALIFICATION_LOSS


class CliTests(unittest.TestCase):
    def test_requested_data_diagnostics_fail_when_cache_is_missing(self) -> None:
        with patch(
            "speedrun.doctor.verify_dataset",
            side_effect=DataError("missing dataset shard"),
        ):
            result = check_prepared_data(Path("/dev/shm"), "official")
        self.assertEqual(result.status, "error")
        self.assertIn("make prepare", result.hint or "")

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
            ["--train-tokens", "100"],
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
        # Two path prompts, one host-count prompt, five menu prompts, one loss
        # target, one token budget, and three confirmations. Dataset
        # preparation is automatic.
        with patch("builtins.input", side_effect=[""] * 13) as prompt:
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
        self.assertEqual(prompt.call_count, 13)

    def test_prepare_training_budget_is_explicit_and_positive(self) -> None:
        parser = cli.build_parser()
        prepared = parser.parse_args(["prepare", "--training-tokens", "1250000000"])
        self.assertEqual(prepared.training_tokens, 1_250_000_000)
        with self.assertRaises(SystemExit):
            parser.parse_args(["prepare", "--training-tokens", "0"])
        self.assertFalse(
            hasattr(parser.parse_args(["run", "reference"]), "training_tokens")
        )
        action = next(
            item
            for item in parser._subparsers._group_actions[0].choices["prepare"]._actions
            if item.dest == "training_tokens"
        )
        self.assertIn("prepare only", action.help)
        self.assertIn("fixed official run contract", action.help)

    def test_prepare_routes_scaled_data_to_dedicated_subfolder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "cache"
            scaled = base / "fineweb-scaled" / "2B"
            manifest = root / "trusted-2B.json"
            prepared = PreparedDataset(
                name="fineweb-2b-gpt2",
                root=scaled,
                manifest_path=manifest,
                manifest_sha256="a" * 64,
                train_files=(scaled / "fineweb_train_000001.bin",),
                validation_files=(scaled / "fineweb_val_000000.bin",),
                train_tokens=1_900_000_000,
                validation_tokens=100_000_000,
                validation_prefix_tokens=100_000_000,
            )
            fresh10 = PreparedFresh10(
                name="fresh10-v1",
                root=base,
                manifest_path=root / "fresh10.json",
                manifest_sha256="b" * 64,
                domains=(),
            )
            args = cli.build_parser().parse_args(
                [
                    "prepare",
                    "--non-interactive",
                    "--no-save",
                    "--check-only",
                    "--path",
                    str(base),
                    "--profile",
                    "official",
                    "--training-tokens",
                    "1900000000",
                ]
            )
            with (
                patch("speedrun.cli.repo_root", return_value=root),
                patch(
                    "speedrun.cli.resolve_preparation_manifest",
                    return_value=manifest,
                ),
                patch("speedrun.cli.environment_checks", return_value=[]) as checks,
                patch("speedrun.cli.run_doctor", return_value=[]),
                patch("speedrun.cli.doctor_ok", return_value=True),
                patch("speedrun.cli.verify_dataset", return_value=prepared) as verify,
                patch("speedrun.cli.verify_fresh10", return_value=fresh10) as fresh,
            ):
                self.assertEqual(cli.command_prepare(args), 0)
            self.assertFalse(checks.call_args.kwargs["check_data"])
            verify.assert_called_once_with(manifest, scaled, train_shards=19)
            fresh.assert_called_once_with(base)

    def test_remote_prepare_forwards_corpus_budget_to_every_peer(self) -> None:
        config = LocalConfig(
            training_tokens=3_900_000_000,
            tpu_vm_count=2,
            tpu_vm_hosts="slice-w-[0-1]",
        ).validate()
        inventory = cli.ClusterInventory(
            host_expression="slice-w-[0-1]",
            hosts=("slice-w-0", "slice-w-1"),
            remote_hosts=("slice-w-1",),
            local_host="slice-w-0",
            reported_hostnames={
                "slice-w-0": "slice-w-0",
                "slice-w-1": "slice-w-1",
            },
        )
        args = cli.build_parser().parse_args(["prepare", "--non-interactive"])
        with patch("speedrun.cli.run_pdsh") as run:
            cli._run_cluster_prepare(config, args, inventory, root=Path("/repo"))
        remote = run.call_args.args[1]
        self.assertIn("--training-tokens 3900000000", remote)
        self.assertIn("--profile official", remote)
        self.assertEqual(run.call_args.kwargs["timeout"], cli._remote_prepare_timeout(config, args))

    def test_remote_prepare_timeout_scales_with_routed_corpus_bytes(self) -> None:
        args = cli.build_parser().parse_args(["prepare", "--non-interactive"])
        classic = cli._remote_prepare_timeout(LocalConfig(), args)
        two_b = cli._remote_prepare_timeout(
            LocalConfig(training_tokens=1_000_000_000), args
        )
        eight_b = cli._remote_prepare_timeout(
            LocalConfig(training_tokens=4_000_000_000), args
        )
        hero = cli._remote_prepare_timeout(
            LocalConfig(training_tokens=8_000_000_000), args
        )
        self.assertLess(classic, two_b)
        self.assertLess(two_b, eight_b)
        self.assertLess(eight_b, hero)
        self.assertGreaterEqual(hero, 6 * 3600)

    def test_unsupported_prepare_budget_is_not_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = cli.build_parser().parse_args(
                [
                    "prepare",
                    "--non-interactive",
                    "--no-doctor",
                    "--no-download",
                    "--path",
                    str(root / "cache"),
                    "--profile",
                    "official",
                    "--training-tokens",
                    "74900000001",
                ]
            )
            with patch("speedrun.cli.repo_root", return_value=root):
                with self.assertRaisesRegex(DataError, "largest prepared corpus"):
                    cli.command_prepare(args)
            self.assertFalse((root / ".speedrun.toml").exists())

    def test_scaled_budget_rejects_manual_shard_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = cli.build_parser().parse_args(
                [
                    "prepare",
                    "--non-interactive",
                    "--no-doctor",
                    "--no-download",
                    "--no-save",
                    "--path",
                    str(root / "cache"),
                    "--profile",
                    "official",
                    "--training-tokens",
                    "1000000000",
                    "--train-shards",
                    "1",
                ]
            )
            with patch("speedrun.cli.repo_root", return_value=root):
                with self.assertRaisesRegex(ConfigError, "cannot truncate"):
                    cli.command_prepare(args)

    def test_wizard_infers_cloud_tpu_host_expression(self) -> None:
        answers = ["", "", "", "4"] + [""] * 10
        with (
            patch("builtins.input", side_effect=answers),
            patch("speedrun.cli.infer_host_expression", return_value="slice-w-[0-3]"),
        ):
            result, *_ = cli._prepare_wizard(
                LocalConfig(),
                run_diagnostics=True,
                require_tpu=True,
                download=True,
                save=True,
            )
        self.assertEqual(result.tpu_vm_count, 4)
        self.assertEqual(result.tpu_vm_hosts, "slice-w-[0-3]")

    def test_cluster_prepare_downloads_on_controller_and_every_peer(self) -> None:
        args = cli.build_parser().parse_args(["prepare", "--non-interactive"])
        config = LocalConfig(
            tpu_vm_count=4,
            tpu_vm_hosts="slice-w-[0-3]",
        )
        inventory = cli.ClusterInventory(
            host_expression=config.tpu_vm_hosts,
            hosts=tuple(f"slice-w-{index}" for index in range(4)),
            remote_hosts=tuple(f"slice-w-{index}" for index in range(1, 4)),
            local_host="slice-w-0",
            reported_hostnames={
                f"slice-w-{index}": f"slice-w-{index}" for index in range(4)
            },
        )
        with patch("speedrun.cli.run_pdsh") as run:
            cli._run_cluster_prepare(config, args, inventory, root=Path("/repo"))

        self.assertEqual(run.call_args.args[0], inventory.hosts)
        remote = run.call_args.args[1]
        self.assertIn("SPEEDRUN_CLUSTER_WORKER=1", remote)
        self.assertIn("--profile official", remote)
        self.assertIn("--path /repo/shm", remote)
        self.assertIn("sudo -n chown -R", remote)
        self.assertIn("/dev/shm/.speedrun-cache", remote)

    def test_ram_cache_path_detection_does_not_follow_existing_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "shm").symlink_to("/dev/shm")
            self.assertTrue(cli._uses_repo_shm_cache("shm", root))
            self.assertTrue(cli._uses_repo_shm_cache("/dev/shm", root))
            self.assertFalse(cli._uses_repo_shm_cache("data", root))

    def test_profile_launches_every_configured_host_with_controller_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            submission = root / "submissions" / "variant"
            submission.mkdir(parents=True)
            (submission / "train.py").write_text("pass\n", encoding="utf-8")
            (submission / "config.yaml").write_text("schema_version: 1\n", encoding="utf-8")
            (root / ".speedrun.toml").write_text("[speedrun]\n", encoding="utf-8")
            config = LocalConfig(
                data_path="shm",
                default_profile="dev",
                tpu_vm_count=4,
                tpu_vm_hosts="slice-w-[0-3]",
            )
            inventory = cli.ClusterInventory(
                host_expression=config.tpu_vm_hosts,
                hosts=tuple(f"slice-w-{index}" for index in range(4)),
                remote_hosts=tuple(f"slice-w-{index}" for index in range(1, 4)),
                local_host="slice-w-0",
                reported_hostnames={
                    f"slice-w-{index}": f"slice-w-{index}" for index in range(4)
                },
            )
            prepared = PreparedDataset(
                name="dev",
                root=Path("/dev/shm"),
                manifest_path=root / "manifest.json",
                manifest_sha256="a" * 64,
                train_files=(Path("/dev/shm/train.bin"),),
                validation_files=(Path("/dev/shm/val.bin"),),
                train_tokens=100,
                validation_tokens=20,
            )
            args = cli.build_parser().parse_args(
                [
                    "profile",
                    "variant",
                    "--output-dir",
                    "profiles/test",
                    "--steps",
                    "20",
                ]
            )
            with (
                patch("speedrun.cli.repo_root", return_value=root),
                patch("speedrun.cli.load_config", return_value=config),
                patch("speedrun.cli.data_selection", return_value=("dev", 1)),
                patch("speedrun.cli.verify_dataset", return_value=prepared),
                patch("speedrun.cli._probe_configured_cluster", return_value=inventory),
                patch("speedrun.cli.sync_workspace") as sync,
                patch("speedrun.cli.run_pdsh") as run,
            ):
                self.assertEqual(cli.command_profile(args), 0)

            self.assertEqual(run.call_args.args[0], inventory.hosts)
            remote = run.call_args.args[1]
            self.assertIn("SPEEDRUN_DISTRIBUTED=1", remote)
            self.assertIn("SPEEDRUN_CONTROLLER_HOSTNAME=slice-w-0", remote)
            self.assertIn("--profile dev", remote)
            self.assertIn("--xprof-dir", remote)
            self.assertIn("profiles/test/xprof", remote)
            sync.assert_called_once()

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
