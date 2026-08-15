"""Friendly command-line front end for preparation, runs, and leaderboards."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from typing import Any, Callable, Iterable, Sequence

from .harness import (
    HarnessError,
    normalize_run_name,
    ReferenceContract,
    RunConfig,
    load_records,
    rank_records,
    render_leaderboard,
    run_submission,
    verify_run,
)
from .harness.cluster import (
    ClusterError,
    ClusterInventory,
    bootstrap_uv,
    infer_host_expression,
    prepare_ram_cache,
    probe_cluster,
    run_pdsh,
    seal_ram_cache_command,
    sync_workspace,
)

from .config import (
    ConfigError,
    LocalConfig,
    config_path,
    load_config,
    repo_root,
    resolve_path,
    save_config,
    with_overrides,
)
from .data import (
    DataError,
    FRESH10_DOMAINS,
    PreparedFresh10,
    PreparedDataset,
    prepare as prepare_data,
    prepare_fresh10,
    sha256_file,
    verify_dataset,
    verify_fresh10,
)
from .data_routing import preparation_route, resolve_preparation_manifest
from .doctor import (
    data_selection,
    doctor_ok,
    environment_checks,
    render_doctor,
    run_doctor,
)
from .report import build_report


_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_TRACKS = ("open", "sample_efficiency")
_PROFILES = ("smoke", "dev", "official")
_TIERS = ("60m", "125m", "250m", "500m", "1b")
_RETENTION = ("all", "qualifying", "none-after-validation")
_COLORS = ("auto", "always", "never")
OFFICIAL_TARGET_LOSS = 3.28
# Legacy v1 calibration fallback used only when re-verifying an old record that
# did not capture its token constraint. New family runs record their own horizon.
OFFICIAL_OPEN_TRAINING_TOKENS = 624_984_064
_CLUSTER_WORKER_ENV = "RIG_CLUSTER_WORKER"
_CONTROLLER_HOST_ENV = "RIG_CONTROLLER_HOSTNAME"
_DISTRIBUTED_ENV = "RIG_DISTRIBUTED"
_PROCESS_COUNT_ENV = "RIG_PROCESS_COUNT"


class Style:
    CODES = {
        "reset": "\033[0m",
        "bold": "\033[1m",
        "dim": "\033[2m",
        "cyan": "\033[38;5;81m",
        "blue": "\033[38;5;75m",
        "green": "\033[38;5;114m",
        "yellow": "\033[38;5;221m",
        "magenta": "\033[38;5;176m",
        "red": "\033[38;5;203m",
    }

    def __init__(self, mode: str = "auto") -> None:
        self.enabled = mode == "always" or (
            mode == "auto" and sys.stdout.isatty() and "NO_COLOR" not in os.environ
        )

    def text(self, value: object, *styles: str) -> str:
        raw = str(value)
        if not self.enabled or not styles:
            return raw
        return "".join(self.CODES[item] for item in styles) + raw + self.CODES["reset"]

    def banner(self, subtitle: str) -> None:
        print(
            f"\n  {self.text('◆', 'magenta', 'bold')}"
            f"{self.text(' GPT TPU SPEEDRUN ', 'bold')}"
            f"{self.text(subtitle, 'cyan')}\n",
            flush=True,
        )

    def heading(self, value: str) -> None:
        print(f"\n  {self.text('●', 'magenta')} {self.text(value, 'bold')}", flush=True)

    def ok(self, value: str) -> None:
        print(f"  {self.text('✓', 'green', 'bold')} {value}", flush=True)

    def note(self, value: str) -> None:
        print(f"  {self.text('→', 'cyan')} {value}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rig",
        description="Prepare, run, and score single-entry JAX trainers on TPU v4 slices.",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser(
        "prepare",
        help="interactive machine, cache, and personal-default setup",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    prepare.add_argument("--path", type=Path, help="exact dataset cache root (for example shm/)")
    prepare.add_argument("--profile", choices=_PROFILES, help="dataset profile to prepare")
    prepare.add_argument("--artifacts", type=Path, help="persistent run artifact directory")
    prepare.add_argument(
        "--tpu-vm-count",
        type=_positive_int,
        help="number of TPU VM hosts participating in one JAX job",
    )
    prepare.add_argument(
        "--tpu-vm-hosts",
        help="pdsh expression containing every TPU VM host",
    )
    prepare.add_argument("--track", choices=_TRACKS, help="default competition track")
    prepare.add_argument("--run-profile", choices=_PROFILES, help="default run profile")
    prepare.add_argument("--checkpoints", choices=_RETENTION, help="checkpoint retention policy")
    prepare.add_argument("--cluster", help="named cluster profile from .rig.toml")
    prepare.add_argument("--color", choices=_COLORS, help="terminal color preference")
    prepare.add_argument(
        "--target-loss",
        type=_nonnegative_float,
        help="default qualification target for smoke/development runs only",
    )
    prepare.add_argument(
        "--training-tokens",
        type=_positive_int,
        help=(
            "corpus capacity to prepare and use for non-smoke runs: official "
            "routes through 900M to classic, through 1.9B/3.9B/7.9B/74.9B "
            "to 2B/4B/8B/hero"
        ),
    )
    prepare.add_argument("--train-shards", type=_positive_int, help="override train shard count")
    prepare.add_argument("--offline", action="store_true", help="forbid network access")
    prepare.add_argument("--check-only", action="store_true", help="verify without mutation")
    prepare.add_argument("--force", action="store_true", help="replace invalid cached shards")
    prepare.add_argument(
        "--timeout", type=_positive_float, default=60.0, help="per-request network timeout"
    )
    prepare.add_argument("--non-interactive", action="store_true", help="use flags/current defaults")
    prepare.add_argument("--yes", action="store_true", help="accept defaults and run non-interactively")
    prepare.add_argument("--no-doctor", action="store_true", help="skip environment diagnostics")
    prepare.add_argument("--no-download", action="store_true", help="save settings without data work")
    prepare.add_argument("--no-save", action="store_true", help="do not write .rig.toml")

    doctor = commands.add_parser(
        "doctor",
        help="validate Python, JAX, TPU topology, storage, and cached data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    doctor.add_argument("--path", type=Path, help="dataset cache root")
    doctor.add_argument("--profile", choices=_PROFILES, help="data/run profile")
    doctor.add_argument(
        "--require-tpu",
        action="store_true",
        help="require the configured TPU v4 topology",
    )
    doctor.add_argument("--quick", action="store_true", help="skip compile/collective probe")
    doctor.add_argument("--skip-data", action="store_true", help="skip dataset integrity scan")
    doctor.add_argument(
        "--training-tokens",
        type=_positive_int,
        help=argparse.SUPPRESS,
    )
    doctor.add_argument("--cluster", help="named cluster profile from .rig.toml")
    doctor.add_argument("--color", choices=_COLORS)

    run = commands.add_parser(
        "run",
        help="execute, validate, and record one submission",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    run.add_argument("submission", help="folder name beneath submissions/")
    run.add_argument("--track", choices=_TRACKS)
    run.add_argument("--profile", choices=_PROFILES)
    run.add_argument("--tier", choices=_TIERS, default="125m")
    run.add_argument(
        "--tokens-per-parameter",
        type=_positive_float,
        help="research budget, rounded by the trainer to a complete global step",
    )
    run.add_argument(
        "--base-learning-rate",
        type=_positive_float,
        help="research override for the family's transferable base learning rate",
    )
    run.add_argument(
        "--study-batch-size",
        type=_positive_int,
        help="research-only global batch override",
    )
    run.add_argument(
        "--name",
        help=(
            "short label folded into the run directory name; prompted for when "
            "omitted on a terminal"
        ),
    )
    run.add_argument("--data-path", type=Path)
    run.add_argument("--seed", type=_nonnegative_int, default=1337)
    run.add_argument("--target-loss", type=_nonnegative_float)
    run.add_argument("--timeout", type=_positive_float, help="whole-process timeout in seconds")
    run.add_argument("--checkpoints", choices=_RETENTION)
    run.add_argument("--cluster", help="named cluster profile from .rig.toml")
    run.add_argument("--color", choices=_COLORS)
    run.add_argument("--skip-data-check", action="store_true")
    run.add_argument(
        "--omit-checkpoint",
        action="store_true",
        help="open/dev research only: retain metrics and curves without model weights",
    )
    run.add_argument("--study-id", help=argparse.SUPPRESS)
    run.add_argument("--study-point", help=argparse.SUPPRESS)
    run.add_argument("--study-suite-sha256", help=argparse.SUPPRESS)

    profile = commands.add_parser(
        "profile",
        help="capture a bounded XProf trace from a distributed submission",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    profile.add_argument(
        "submission", nargs="?", default="reference", help="folder name beneath submissions/"
    )
    profile.add_argument("--profile", choices=_PROFILES, help="saved profile override")
    profile.add_argument("--tier", choices=_TIERS, default="125m")
    profile.add_argument("--data-path", type=Path, help="dataset cache root override")
    profile.add_argument("--output-dir", type=Path, required=True)
    profile.add_argument("--steps", type=_positive_int, default=100)
    profile.add_argument("--xprof-start-step", type=_positive_int, default=11)
    profile.add_argument("--xprof-steps", type=_positive_int, default=10)
    profile.add_argument("--seed", type=_nonnegative_int, default=1337)
    profile.add_argument("--timeout", type=_positive_float, default=7200.0)
    profile.add_argument("--cluster", help="named cluster profile from .rig.toml")
    profile.add_argument("--color", choices=_COLORS)

    verify = commands.add_parser("verify", help="re-validate a captured run and checkpoint")
    verify.add_argument("run", help="run ID or path")
    verify.add_argument("--track", choices=_TRACKS)
    verify.add_argument("--profile", choices=_PROFILES)

    leaderboard = commands.add_parser("leaderboard", help="render recorded qualifying scores")
    leaderboard.add_argument("--track", choices=_TRACKS)
    leaderboard.add_argument("--profile", choices=_PROFILES, default="official")
    leaderboard.add_argument("--target-loss", type=_nonnegative_float)
    leaderboard.add_argument("--all-submissions", action="store_true")
    leaderboard.add_argument("--color", choices=_COLORS)

    report = commands.add_parser(
        "report",
        help="build a self-contained HTML comparison of completed run logs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    report.add_argument("--runs", type=Path, default=Path("runs"), help="run log directory")
    report.add_argument(
        "--output", type=Path, default=Path("report.html"), help="standalone HTML destination"
    )
    report.add_argument(
        "--layer-snapshots",
        type=_nonnegative_int,
        default=0,
        help=(
            "recorded steps kept per layer-snapshot chart; 0 (the default) keeps "
            "every recorded step, a positive value thins them to shrink the file"
        ),
    )
    report.add_argument(
        "--max-points",
        type=_positive_int,
        default=1_400,
        help="maximum embedded points per run and scalar series",
    )

    clone = commands.add_parser("clone", help="clone one submission into a new algorithm folder")
    clone.add_argument("source", nargs="?", default="reference")
    clone.add_argument("name")

    settings = commands.add_parser("settings", help="show resolved local preferences")
    settings.add_argument("--json", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    arguments = list(argv) if argv is not None else sys.argv[1:]
    args, unknown = parser.parse_known_args(arguments)
    if unknown and args.command != "run":
        parser.error("unrecognized arguments: " + " ".join(unknown))
    # Unknown arguments are legal only for `run`; conventionally they follow
    # `--`, but parse_known_args also makes common direct flags ergonomic.
    args.trainer_args = unknown if args.command == "run" else []
    try:
        if args.command == "prepare":
            return command_prepare(args)
        if args.command == "doctor":
            return command_doctor(args)
        if args.command == "run":
            return command_run(args)
        if args.command == "profile":
            return command_profile(args)
        if args.command == "verify":
            return command_verify(args)
        if args.command == "leaderboard":
            return command_leaderboard(args)
        if args.command == "report":
            return command_report(args)
        if args.command == "clone":
            return command_clone(args)
        if args.command == "settings":
            return command_settings(args)
        parser.error(f"unknown command {args.command!r}")
    except (ClusterError, ConfigError, DataError, HarnessError, OSError, ValueError) as exc:
        style = Style(getattr(args, "color", None) or "auto")
        print(f"\n  {style.text('error:', 'red', 'bold')} {exc}\n", file=sys.stderr)
        return 1
    return 0


def command_prepare(args: argparse.Namespace) -> int:
    root = repo_root()
    current = load_config(root, cluster=getattr(args, "cluster", None))
    proposed = with_overrides(
        current,
        {
            "data_path": str(args.path) if args.path is not None else None,
            "artifacts_path": str(args.artifacts) if args.artifacts is not None else None,
            "active_cluster": getattr(args, "cluster", None),
            "tpu_vm_count": args.tpu_vm_count,
            "tpu_vm_hosts": args.tpu_vm_hosts,
            "data_profile": args.profile,
            "default_profile": args.run_profile,
            "default_track": args.track,
            "checkpoint_retention": args.checkpoints,
            "color": args.color,
            "target_loss": args.target_loss,
            "training_tokens": args.training_tokens,
        },
    )
    interactive = not (args.non_interactive or args.yes)
    if interactive and not sys.stdin.isatty():
        raise ConfigError(
            "prepare needs a terminal for its wizard; pass --non-interactive with explicit flags"
        )
    run_diagnostics = not args.no_doctor
    data_work = not args.no_download
    require_tpu = proposed.default_profile == "official"
    save = not args.no_save
    if interactive:
        proposed, run_diagnostics, require_tpu, data_work, save = _prepare_wizard(
            proposed,
            run_diagnostics=run_diagnostics,
            require_tpu=require_tpu,
            download=data_work,
            save=save,
        )

    style = Style(proposed.color)
    style.banner("prepare")
    data_path = resolve_path(proposed.data_path, root)
    artifacts_path = resolve_path(proposed.artifacts_path, root)
    _ensure_artifacts_inside_repo(artifacts_path, root)
    route = preparation_route(proposed.data_profile, proposed.training_tokens)
    if route.is_scaled and args.train_shards is not None:
        raise ConfigError(
            "--train-shards cannot truncate a budget-selected scaled dataset; "
            "choose a smaller --training-tokens budget instead"
        )
    route_root = route.data_root(data_path)
    route_manifest = (
        resolve_preparation_manifest(route) if data_work else route.manifest
    )
    style.note(route.summary(proposed.training_tokens))
    if route.is_scaled:
        style.note(f"scaled shards use the dedicated cache {route_root}")
    if args.check_only and save:
        style.note("check-only mode does not write .rig.toml")
        save = False
    if save:
        destination = save_config(proposed, root)
        style.ok(f"saved personal defaults to {destination.relative_to(root)}")
    else:
        style.note("settings are temporary (--no-save)")

    cluster_controller = proposed.tpu_vm_count > 1 and not _is_cluster_worker()
    inventory: ClusterInventory | None = None
    if cluster_controller:
        inventory = _prepare_cluster(
            proposed,
            args,
            root=root,
            artifacts_path=artifacts_path,
            style=style,
        )

    if run_diagnostics and not cluster_controller:
        style.heading("Machine diagnostics")
        results = run_doctor(
            environment_checks(
                data_path=data_path,
                profile=proposed.data_profile,
                require_tpu=require_tpu,
                expected_process_count=_expected_process_count(proposed),
                accelerator=proposed.accelerator,
                chips_per_host=proposed.chips_per_host,
                training_tokens=proposed.training_tokens,
                check_data=args.check_only and not route.is_scaled,
                compile_probe=True,
            )
        )
        print(_indent(render_doctor(results, color=style.enabled)))
        if not doctor_ok(results):
            raise ConfigError("machine diagnostics failed; resolve the errors above")

    if not data_work:
        style.note("dataset preparation explicitly skipped (--no-download)")
    elif inventory is not None:
        style.heading("Dataset caches")
        style.note(
            f"preparing datasets and validations concurrently on "
            f"{len(inventory.hosts)} TPU VMs"
        )
        _run_cluster_prepare(proposed, args, inventory, root=root)
    else:
        style.heading("Dataset cache")
        shards = args.train_shards or route.train_shards
        if args.check_only:
            prepared = verify_dataset(route_manifest, route_root, train_shards=shards)
        else:
            progress = _progress_reporter(style)
            prepared = prepare_data(
                route_root,
                route_manifest,
                train_shards=shards,
                offline=args.offline,
                force=args.force,
                progress=progress,
                timeout=args.timeout,
            )
        _print_prepared(prepared, style)
        if proposed.data_profile == "official":
            style.heading("Fresh-domain diagnostic")
            if args.check_only:
                fresh10 = verify_fresh10(data_path)
            else:
                fresh10 = prepare_fresh10(
                    data_path,
                    offline=args.offline,
                    force=args.force,
                    progress=_progress_reporter(style),
                    timeout=args.timeout,
                )
            _print_fresh10(fresh10, style)

    if inventory is not None:
        if run_diagnostics:
            style.heading("Distributed machine diagnostics")
            _run_cluster_doctor(
                proposed,
                inventory,
                profile=proposed.data_profile,
                data_path=proposed.data_path,
                require_tpu=require_tpu,
                # Each remote scaled prepare already verifies its routed
                # manifest and nested cache; avoid hashing all 40 shards twice
                # in the same preparation transaction.
                check_data=(data_work or args.check_only) and not route.is_scaled,
                quick=False,
                color=proposed.color,
                root=root,
            )
    return 0


def command_doctor(args: argparse.Namespace) -> int:
    config = load_config(cluster=getattr(args, 'cluster', None))
    profile = args.profile or config.default_profile
    training_tokens = args.training_tokens or config.training_tokens
    path = resolve_path(args.path or config.data_path)
    color = args.color or config.color
    if config.tpu_vm_count > 1 and not _is_cluster_worker():
        inventory = _probe_configured_cluster(config)
        return _run_cluster_doctor(
            config,
            inventory,
            profile=profile,
            data_path=str(args.path or config.data_path),
            require_tpu=args.require_tpu or profile == "official",
            check_data=not args.skip_data,
            quick=args.quick,
            color=color,
            root=repo_root(),
            cluster=getattr(args, "cluster", None),
        )

    process_index = _initialize_distributed_worker(config)
    is_controller = _is_controller_process(process_index)
    style = Style(color)
    if is_controller:
        style.banner("doctor")
    results = run_doctor(
        environment_checks(
            data_path=path,
            profile=profile,
            require_tpu=args.require_tpu or profile == "official",
            expected_process_count=_expected_process_count(config),
            accelerator=config.accelerator,
            chips_per_host=config.chips_per_host,
            training_tokens=training_tokens,
            check_data=not args.skip_data,
            compile_probe=not args.quick,
        )
    )
    healthy = doctor_ok(results)
    if is_controller:
        print(_indent(render_doctor(results, color=style.enabled)))
    elif not healthy:
        print(render_doctor(results, color=False), file=sys.stderr)
    return 0 if healthy else 1


def command_run(args: argparse.Namespace) -> int:
    config = load_config(cluster=getattr(args, 'cluster', None))
    root = repo_root()
    track = args.track or config.default_track
    profile = args.profile or config.default_profile
    color = args.color or config.color
    style = Style(color)
    # Resolved first: an interactive prompt must not wait behind dataset
    # verification and cluster synchronization.
    run_name = _resolve_run_name(args, style)
    trainer_color = "always" if style.enabled else "never"
    target_loss = _effective_target_loss(
        profile, requested=args.target_loss, development_default=config.target_loss
    )
    configured_data_path = str(args.data_path or config.data_path)
    if config.tpu_vm_count > 1:
        configured_data_path = _cluster_data_argument(configured_data_path, root)
    data_path = resolve_path(configured_data_path, root)
    artifacts = resolve_path(config.artifacts_path, root)
    _ensure_artifacts_inside_repo(artifacts, root)
    if profile == "official" and args.skip_data_check:
        raise ConfigError("official runs require full dataset SHA-256 verification")
    # A profile controls runtime/evaluation policy.  The saved preparation
    # budget controls which immutable FineWeb prefix backs non-smoke runs.
    # Keeping those axes separate lets a short dev study consume the same
    # rank-disjoint corpus as the eventual official family run.
    run_data_profile = "smoke" if profile == "smoke" else config.data_profile
    route = preparation_route(run_data_profile, config.training_tokens)
    manifest = resolve_preparation_manifest(route)
    route_root = route.data_root(data_path)
    shards = route.train_shards
    style.heading("Verifying cached data")
    prepared = verify_dataset(
        manifest,
        route_root,
        train_shards=shards,
        verify_hash=not args.skip_data_check,
    )
    if not args.skip_data_check:
        style.ok(
            f"{prepared.name}: {prepared.train_tokens:,} train / "
            f"{prepared.validation_tokens:,} validation tokens"
        )
    else:
        style.note("SHA-256 scan skipped; headers and exact shard selection still checked")

    fresh10: PreparedFresh10 | None = None
    if profile == "official":
        fresh10 = verify_fresh10(data_path, verify_hash=True)
        style.ok(
            f"{fresh10.name}: {len(fresh10.domains)} domains / "
            f"{fresh10.scored_tokens:,} scored tokens"
        )

    if config.tpu_vm_count > 1:
        style.heading("Synchronizing TPU VM cluster")
        inventory = _probe_configured_cluster(config)
        sync_workspace(
            root,
            inventory,
            artifacts_path=artifacts,
            data_path=data_path,
        )
        style.ok(f"current source synchronized to {len(inventory.remote_hosts)} peer VMs")

    dataset_id, tokenizer_id = _data_identity(
        run_data_profile, prepared_name=prepared.name
    )
    passthrough = [
        "--tier",
        args.tier,
        "--data-format",
        "llmc",
        "--dataset-id",
        dataset_id,
        "--tokenizer-id",
        tokenizer_id,
        "--color",
        trainer_color,
    ]
    if args.tokens_per_parameter is not None:
        passthrough.extend(
            ("--tokens-per-parameter", str(args.tokens_per_parameter))
        )
    if args.base_learning_rate is not None:
        passthrough.extend(
            ("--base-learning-rate", str(args.base_learning_rate))
        )
    if args.study_batch_size is not None:
        passthrough.extend(("--study-batch-size", str(args.study_batch_size)))
    for train_file in prepared.train_files:
        passthrough.extend(("--train-data", str(train_file)))
    for validation_file in prepared.validation_files:
        passthrough.extend(("--val-data", str(validation_file)))
    if fresh10 is not None:
        passthrough.extend(("--downstream-manifest", str(fresh10.manifest_path)))
        passthrough.extend(("--downstream-root", str(fresh10.root)))
    forwarded = list(args.trainer_args)
    if forwarded and forwarded[0] == "--":
        forwarded.pop(0)
    _reject_reserved_trainer_args(forwarded)
    passthrough.extend(forwarded)
    timeout = args.timeout or {"smoke": 300.0, "dev": 3600.0, "official": 21600.0}[profile]
    retention = args.checkpoints or config.checkpoint_retention
    if args.omit_checkpoint and (track != "open" or profile != "dev"):
        raise ConfigError("--omit-checkpoint is restricted to open/dev research runs")
    if args.omit_checkpoint:
        passthrough.append("--omit-checkpoint")
    study_values = (args.study_id, args.study_point, args.study_suite_sha256)
    if any(value is not None for value in study_values) and not all(
        value is not None for value in study_values
    ):
        raise ConfigError(
            "--study-id, --study-point, and --study-suite-sha256 must be supplied together"
        )
    if args.study_id is not None and (
        not _NAME.fullmatch(args.study_id) or not _NAME.fullmatch(args.study_point)
    ):
        raise ConfigError("study and point IDs must be simple filesystem-safe names")
    if args.study_suite_sha256 is not None and not re.fullmatch(
        r"[0-9a-f]{64}", args.study_suite_sha256
    ):
        raise ConfigError("study suite SHA-256 must be 64 lowercase hexadecimal digits")
    reference = (
        _reference_contract(
            profile,
            tier=args.tier,
            dataset_id=dataset_id,
            tokenizer_id=tokenizer_id,
        )
        if track == "sample_efficiency"
        else None
    )
    style.banner(f"run / {track} / {profile}")
    outcome = run_submission(
        RunConfig(
            repo_root=root,
            submission=args.submission,
            name=run_name,
            runs_dir=artifacts,
            records_path=artifacts / "records.jsonl",
            track=track,
            profile=profile,
            seed=args.seed,
            timeout_seconds=timeout,
            target_loss=target_loss,
            expected_training_tokens=None,
            expected_validation_tokens=(
                prepared.validation_prefix_tokens if profile == "official" else None
            ),
            expected_downstream_tokens=(
                {domain.name: domain.scored_tokens for domain in fresh10.domains}
                if fresh10 is not None
                else None
            ),
            passthrough_args=tuple(passthrough),
            reference_contract=reference,
            checkpoint_retention=retention,
            environment={},
            provenance={
                **_data_provenance(
                    prepared,
                    profile=run_data_profile,
                    integrity="headers+size" if args.skip_data_check else "sha256",
                    repo=root,
                    fresh10=fresh10,
                ),
                "cluster": {
                    "tpu_vm_count": config.tpu_vm_count,
                    "tpu_vm_hosts": config.tpu_vm_hosts,
                },
                **(
                    {
                        "study": {
                            "study_id": args.study_id,
                            "point_id": args.study_point,
                            "suite_sha256": args.study_suite_sha256,
                        }
                    }
                    if args.study_id is not None
                    else {}
                ),
            },
            tpu_vm_count=config.tpu_vm_count,
            tpu_vm_hosts=config.tpu_vm_hosts,
            require_checkpoint=not args.omit_checkpoint,
        )
    )
    metrics = outcome.record["metrics"]
    qualified = bool(outcome.record["qualified"])
    marker = style.text("QUALIFIED", "green", "bold") if qualified else style.text("NOT QUALIFIED", "yellow", "bold")
    style.heading("Recorded result")
    print(f"  {marker}  loss {metrics['validation_loss']:.4f}  target ≤ {target_loss:.4f}")
    print(f"  train {metrics['train_seconds']:.3f}s  tokens {metrics['tokens_processed']:,}")
    evaluations = outcome.record.get("evaluations")
    if isinstance(evaluations, dict):
        fresh = evaluations.get("fresh10")
        if isinstance(fresh, dict):
            print(
                f"  fresh10 macro loss {float(fresh['macro_loss']):.4f}  "
                f"ppl {float(fresh['macro_perplexity']):.2f}"
            )
    print(f"  run {outcome.run_id}\n")
    return 0


def command_profile(args: argparse.Namespace) -> int:
    """Run one bounded diagnostic on the configured JAX process topology."""

    root = repo_root()
    if not config_path(root).is_file():
        raise ConfigError("no saved default profile found; run `make prepare` first")
    config = load_config(root, cluster=getattr(args, "cluster", None))
    profile = args.profile or config.default_profile
    color = args.color or config.color
    style = Style(color)
    if args.xprof_start_step + args.xprof_steps - 1 > args.steps:
        raise ConfigError("the XProf capture window must fit inside --steps")

    if not _NAME.fullmatch(args.submission):
        raise ConfigError(
            "submission names may contain only letters, digits, '.', '_' and '-'"
        )
    submissions_root = (root / "submissions").resolve()
    submission_dir = (submissions_root / args.submission).resolve()
    try:
        submission_dir.relative_to(submissions_root)
    except ValueError as exc:
        raise ConfigError("submission path escapes submissions directory") from exc
    trainer = submission_dir / "train.py"
    experiment_config = submission_dir / "config.yaml"
    if not trainer.is_file() or trainer.is_symlink():
        raise ConfigError(f"submission entry script not found: {trainer}")
    if not experiment_config.is_file() or experiment_config.is_symlink():
        raise ConfigError(f"submission configuration file not found: {experiment_config}")

    configured_data_path = str(args.data_path or config.data_path)
    if config.tpu_vm_count > 1:
        configured_data_path = _cluster_data_argument(configured_data_path, root)
    data_path = resolve_path(configured_data_path, root)
    run_data_profile = "smoke" if profile == "smoke" else config.data_profile
    route = preparation_route(run_data_profile, config.training_tokens)
    manifest = resolve_preparation_manifest(route)
    route_root = route.data_root(data_path)
    shards = route.train_shards
    style.heading("Verifying cached profile data")
    prepared = verify_dataset(manifest, route_root, train_shards=shards)
    style.ok(
        f"{prepared.name}: {prepared.train_tokens:,} train / "
        f"{prepared.validation_tokens:,} validation tokens"
    )

    output_dir = resolve_path(args.output_dir, root)
    xprof_dir = output_dir / "xprof"
    dataset_id, tokenizer_id = _data_identity(
        run_data_profile, prepared_name=prepared.name
    )
    trainer_color = "always" if style.enabled else "never"
    trainer_command = [
        str(root / ".venv" / "bin" / "python"),
        str(trainer),
        "--config",
        str(experiment_config),
        "--output-dir",
        str(output_dir),
        "--seed",
        str(args.seed),
        "--track",
        config.default_track,
        "--profile",
        profile,
        "--tier",
        args.tier,
        "--steps",
        str(args.steps),
        "--val-every",
        "0",
        "--diagnostics-every",
        "0",
        "--log-every",
        str(args.steps),
        "--data-format",
        "llmc",
        "--dataset-id",
        dataset_id,
        "--tokenizer-id",
        tokenizer_id,
    ]
    for train_file in prepared.train_files:
        trainer_command.extend(("--train-data", str(train_file)))
    for validation_file in prepared.validation_files:
        trainer_command.extend(("--val-data", str(validation_file)))
    trainer_command.extend(
        (
            "--xprof-dir",
            str(xprof_dir),
            "--xprof-start-step",
            str(args.xprof_start_step),
            "--xprof-steps",
            str(args.xprof_steps),
            "--no-final-validation",
            "--no-checkpoint",
            "--color",
            trainer_color,
        )
    )

    style.banner(f"profile / {args.submission} / {profile}")
    if config.tpu_vm_count > 1:
        inventory = _probe_configured_cluster(config)
        style.note(
            f"synchronizing source and launching all {len(inventory.hosts)} TPU VMs"
        )
        sync_workspace(
            root,
            inventory,
            artifacts_path=output_dir,
            data_path=data_path,
        )
        remote_environment = {
            _CLUSTER_WORKER_ENV: "1",
            _CONTROLLER_HOST_ENV: inventory.reported_hostnames[inventory.local_host],
            _DISTRIBUTED_ENV: "1",
            _PROCESS_COUNT_ENV: str(config.tpu_vm_count),
            "JAX_COMPILATION_CACHE_DIR": f"/tmp/rig-profile-cache-{os.getpid()}",
            "PYTHONUNBUFFERED": "1",
        }
        assignments = " ".join(
            f"{key}={shlex.quote(value)}" for key, value in remote_environment.items()
        )
        remote = (
            f"cd {shlex.quote(str(submission_dir))} && "
            f"env {assignments} {shlex.join(trainer_command)}"
        )
        run_pdsh(
            inventory.hosts,
            remote,
            labels=True,
            timeout=float(args.timeout),
            # A partially launched JAX collective must be torn down, not
            # replayed while surviving workers may still be waiting.
            retry_transport=False,
        )
    else:
        try:
            completed = subprocess.run(
                trainer_command,
                cwd=submission_dir,
                check=False,
                timeout=float(args.timeout),
            )
        except subprocess.TimeoutExpired as exc:
            raise ConfigError(
                f"profile trainer timed out after {float(args.timeout):g}s"
            ) from exc
        if completed.returncode != 0:
            raise ConfigError(
                f"profile trainer exited with status {completed.returncode}"
            )
    style.ok(f"worker 0 XProf trace saved to {xprof_dir}")
    return 0


def command_verify(args: argparse.Namespace) -> int:
    config = load_config(cluster=getattr(args, 'cluster', None))
    root = repo_root()
    artifacts = resolve_path(config.artifacts_path, root)
    candidate = Path(args.run).expanduser()
    run_dir = candidate.resolve() if candidate.exists() else (artifacts / args.run).resolve()
    records = load_records(artifacts / "records.jsonl")
    record = next(
        (item for item in reversed(records) if item.get("run_id") == run_dir.name), None
    )
    track = args.track or (str(record["track"]) if record is not None else config.default_track)
    profile = args.profile or (
        str(record["profile"]) if record is not None else config.default_profile
    )
    recorded_contract = record.get("contract") if isinstance(record, dict) else None
    recorded_model = (
        recorded_contract.get("model")
        if isinstance(recorded_contract, dict)
        else None
    )
    recorded_tier = (
        recorded_model.get("tier") if isinstance(recorded_model, dict) else "125m"
    )
    recorded_dataset = (
        recorded_contract.get("dataset_id")
        if isinstance(recorded_contract, dict)
        else None
    )
    recorded_tokenizer = (
        recorded_contract.get("tokenizer_id")
        if isinstance(recorded_contract, dict)
        else None
    )
    reference = (
        _reference_contract(
            profile,
            tier=str(recorded_tier),
            dataset_id=(str(recorded_dataset) if recorded_dataset is not None else None),
            tokenizer_id=(
                str(recorded_tokenizer) if recorded_tokenizer is not None else None
            ),
        )
        if track == "sample_efficiency"
        else None
    )
    expected_downstream = (
        _recorded_downstream_tokens(record)
        if profile == "official" and record is not None
        else None
    )
    result = verify_run(
        run_dir,
        track=track,
        reference_contract=reference,
        expected_training_tokens=(
            _recorded_training_tokens(record)
            if record is not None
            else (
                OFFICIAL_OPEN_TRAINING_TOKENS
                if profile == "official" and track == "open"
                else None
            )
        ),
        expected_validation_tokens=10_485_760 if profile == "official" else None,
        expected_downstream_tokens=expected_downstream,
        require_checkpoint=(
            record is None or isinstance(record.get("checkpoint"), dict)
        ),
    )
    if record is not None:
        stdout_sha256 = sha256_file(run_dir / "stdout.log")
        expected_stdout = record.get("logs", {}).get("stdout_sha256")
        if stdout_sha256 != expected_stdout:
            raise HarnessError("captured stdout hash no longer matches its immutable record")
        recorded_checkpoint = record.get("checkpoint")
        expected_checkpoint = (
            recorded_checkpoint.get("sha256")
            if isinstance(recorded_checkpoint, dict)
            else None
        )
        if result.checkpoint_sha256 != expected_checkpoint:
            raise HarnessError("checkpoint hash no longer matches its immutable record")
        recorded_artifacts = record.get("artifacts", {})
        if set(result.artifacts) != set(recorded_artifacts):
            raise HarnessError("run artifacts no longer match their immutable record")
        for name, path in result.artifacts.items():
            expected_artifact = recorded_artifacts.get(name, {}).get("sha256")
            if sha256_file(path) != expected_artifact:
                raise HarnessError(
                    f"artifact {name!r} hash no longer matches its immutable record"
                )
    checkpoint_label = (
        result.checkpoint_sha256[:12]
        if result.checkpoint_sha256 is not None
        else "omitted"
    )
    print(
        f"verified {run_dir.name} ({track}/{profile}): loss={result.validation_loss:.4f}, "
        f"tokens={result.tokens_processed:,}, checkpoint={checkpoint_label}"
    )
    return 0


def command_leaderboard(args: argparse.Namespace) -> int:
    config = load_config(cluster=getattr(args, 'cluster', None))
    track = args.track or config.default_track
    artifacts = resolve_path(config.artifacts_path)
    records = load_records(artifacts / "records.jsonl")
    target_loss = _effective_target_loss(
        args.profile, requested=args.target_loss, development_default=config.target_loss
    )
    ranked = rank_records(
        records,
        track=track,
        profile=args.profile,
        target_loss=target_loss,
        best_per_submission=not args.all_submissions,
    )
    style = Style(args.color or config.color)
    print(f"target validation loss ≤ {target_loss:.4f}\n")
    print(render_leaderboard(ranked, track=track, color=style.enabled))
    return 0


def command_report(args: argparse.Namespace) -> int:
    root = repo_root()
    runs = args.runs if args.runs.is_absolute() else root / args.runs
    output = args.output if args.output.is_absolute() else root / args.output
    summary = build_report(
        runs,
        output,
        max_chart_points=args.max_points,
        layer_snapshots=args.layer_snapshots,
    )
    relative = (
        summary.output_path.relative_to(root)
        if summary.output_path.is_relative_to(root)
        else summary.output_path
    )
    print(
        f"report {relative}: {len(summary.included)} run(s) plotted, "
        f"{len(summary.skipped)} skipped"
    )
    for run_id, reason in summary.skipped.items():
        print(f"  skipped {run_id}: {reason}")
    return 0


def command_clone(args: argparse.Namespace) -> int:
    if not _NAME.fullmatch(args.source) or not _NAME.fullmatch(args.name):
        raise ConfigError("submission names may contain only letters, digits, '.', '_' and '-'")
    root = repo_root()
    source = root / "submissions" / args.source
    destination = root / "submissions" / args.name
    if not (source / "train.py").is_file():
        raise ConfigError(f"source submission does not exist: {source}")
    source_config = source / "config.yaml"
    if not source_config.is_file() or source_config.is_symlink():
        raise ConfigError(f"source submission configuration does not exist: {source_config}")
    if destination.exists():
        raise ConfigError(f"destination already exists: {destination}")
    destination.mkdir(parents=True)
    shutil.copy2(source / "train.py", destination / "train.py")
    shutil.copy2(source_config, destination / "config.yaml")
    if (source / "README.md").is_file():
        shutil.copy2(source / "README.md", destination / "README.md")
    print(f"cloned {args.source} -> {args.name} ({destination})")
    return 0


def command_settings(args: argparse.Namespace) -> int:
    config = load_config(cluster=getattr(args, 'cluster', None))
    payload = asdict(config)
    root = repo_root()
    payload["data_path_resolved"] = str(resolve_path(config.data_path, root))
    payload["artifacts_path_resolved"] = str(resolve_path(config.artifacts_path, root))
    payload["config_path"] = str(config_path())
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        width = max(len(key) for key in payload)
        for key, value in payload.items():
            print(f"{key:<{width}}  {value}")
    return 0


def _prepare_cluster(
    config: LocalConfig,
    args: argparse.Namespace,
    *,
    root: Path,
    artifacts_path: Path,
    style: Style,
) -> ClusterInventory:
    style.heading("TPU VM cluster")
    inventory = probe_cluster(config.tpu_vm_hosts, config.tpu_vm_count)
    style.ok(
        f"passwordless SSH ready on {len(inventory.hosts)} hosts "
        f"({len(inventory.remote_hosts)} peer VMs)"
    )
    if _uses_repo_shm_cache(config.data_path, root):
        style.note("checking RAM-backed /dev/shm and configuring the shm cache link")
        prepare_ram_cache(root, inventory, create_link=not args.check_only)
        style.ok("shm points to writable RAM-backed storage on every TPU VM")
    if args.check_only:
        style.note("check-only mode does not synchronize source or environments")
    else:
        style.note("incrementally synchronizing source and personal settings to peer VMs")
        sync_workspace(
            root,
            inventory,
            artifacts_path=artifacts_path,
            data_path=resolve_path(config.data_path, root),
        )
        style.note("synchronizing the frozen uv environment on peer VMs")
        bootstrap_uv(root, inventory.remote_hosts, offline=args.offline)
    return inventory


def _run_cluster_prepare(
    config: LocalConfig,
    args: argparse.Namespace,
    inventory: ClusterInventory,
    *,
    root: Path,
) -> None:
    if not inventory.hosts:
        return
    data_argument = _cluster_data_argument(config.data_path, root)
    command = [
        str(root / ".venv" / "bin" / "python"),
        "-m",
        "rig",
        "prepare",
        "--non-interactive",
        "--no-save",
        "--no-doctor",
        "--path",
        data_argument,
        "--profile",
        config.data_profile,
        "--artifacts",
        config.artifacts_path,
        "--tpu-vm-count",
        str(config.tpu_vm_count),
        "--tpu-vm-hosts",
        config.tpu_vm_hosts,
        "--track",
        config.default_track,
        "--run-profile",
        config.default_profile,
        "--checkpoints",
        config.checkpoint_retention,
        "--color",
        "never",
        "--target-loss",
        str(config.target_loss),
        "--training-tokens",
        str(config.training_tokens),
        "--timeout",
        str(args.timeout),
    ]
    # Peers inherit the mirrored .rig.toml, but an explicit selection on the
    # controller must reach them or they resolve a different cluster -- or, if
    # the file has no active cluster, refuse to resolve one at all.
    if getattr(args, "cluster", None):
        command.extend(("--cluster", args.cluster))
    if args.train_shards is not None:
        command.extend(("--train-shards", str(args.train_shards)))
    if args.offline:
        command.append("--offline")
    if args.check_only:
        command.append("--check-only")
    if args.force:
        command.append("--force")
    worker_command = f"env {_CLUSTER_WORKER_ENV}=1 {shlex.join(command)}"
    if _uses_repo_shm_cache(config.data_path, root) and not args.check_only:
        worker_command = f"{worker_command} && {seal_ram_cache_command()}"
    remote = f"cd {shlex.quote(str(root.resolve()))} && {worker_command}"
    run_pdsh(
        inventory.hosts,
        remote,
        labels=True,
        timeout=_remote_prepare_timeout(config, args),
    )


def _remote_prepare_timeout(config: LocalConfig, args: argparse.Namespace) -> float:
    """Return a whole-peer cap distinct from the per-request HTTP timeout.

    Budget routing determines the nominal shard bytes each peer must install.
    Estimate transfer plus verification at 10 MiB/s, add 50% for contention and
    retries, then add 30 minutes for fixed setup.  This gives hero roughly 6.5
    hours while retaining a 15-minute absolute floor.  Offline and check-only
    paths still use the same safe upper bound; it is a deadline, not an expected
    duration.
    """

    route = preparation_route(config.data_profile, config.training_tokens)
    token_bytes = 2 * (
        (route.train_capacity or route.train_shards * 100_000_000)
        + (100_000_000 if config.data_profile != "smoke" else 0)
    )
    transfer_and_verify = token_bytes / (10 * 1024**2)
    route_deadline = transfer_and_verify * 1.5 + 30 * 60.0
    return max(900.0, float(args.timeout) * 20.0, route_deadline)


def _uses_repo_shm_cache(value: str, root: Path) -> bool:
    """Recognize the conventional checkout ``shm`` path without following links."""

    configured = Path(value).expanduser()
    if not configured.is_absolute():
        configured = root / configured
    lexical = Path(os.path.abspath(configured))
    return lexical in {root.resolve() / "shm", Path("/dev/shm")}


def _cluster_data_argument(value: str, root: Path) -> str:
    """Route conventional multi-host shm use through the protected cache link."""

    if _uses_repo_shm_cache(value, root):
        return str(root.resolve() / "shm")
    return value


def _probe_configured_cluster(config: LocalConfig) -> ClusterInventory:
    if config.tpu_vm_count <= 1:
        raise ConfigError("multi-host operation requires tpu_vm_count greater than 1")
    return probe_cluster(config.tpu_vm_hosts, config.tpu_vm_count)


def _run_cluster_doctor(
    config: LocalConfig,
    inventory: ClusterInventory,
    *,
    profile: str,
    data_path: str,
    require_tpu: bool,
    check_data: bool,
    quick: bool,
    color: str,
    root: Path,
    cluster: str | None = None,
) -> int:
    data_path = _cluster_data_argument(data_path, root)
    command = [
        str(root / ".venv" / "bin" / "python"),
        "-m",
        "rig",
        "doctor",
        "--path",
        data_path,
        "--profile",
        profile,
        "--training-tokens",
        str(config.training_tokens),
        "--color",
        color,
    ]
    # Peers must evaluate the same cluster contract; otherwise a controller
    # invoked with --cluster would check itself against one profile and its
    # peers against whatever their file happens to make active.
    if cluster:
        command.extend(("--cluster", cluster))
    if require_tpu:
        command.append("--require-tpu")
    if not check_data:
        command.append("--skip-data")
    if quick:
        command.append("--quick")
    remote = (
        f"cd {shlex.quote(str(root.resolve()))} && env "
        f"{_CLUSTER_WORKER_ENV}=1 "
        f"{_CONTROLLER_HOST_ENV}={shlex.quote(inventory.reported_hostnames[inventory.local_host])} "
        f"{_DISTRIBUTED_ENV}=1 "
        f"{_PROCESS_COUNT_ENV}={config.tpu_vm_count} {shlex.join(command)}"
    )
    run_pdsh(
        inventory.hosts,
        remote,
        labels=True,
        timeout=900.0,
    )
    return 0


def _is_cluster_worker() -> bool:
    return os.environ.get(_CLUSTER_WORKER_ENV) == "1"


def _expected_process_count(config: LocalConfig) -> int:
    raw = os.environ.get(_PROCESS_COUNT_ENV)
    if raw is None:
        return config.tpu_vm_count
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{_PROCESS_COUNT_ENV} must be a positive integer") from exc
    if value <= 0:
        raise ConfigError(f"{_PROCESS_COUNT_ENV} must be a positive integer")
    return value


def _is_controller_process(process_index: int) -> bool:
    configured = os.environ.get(_CONTROLLER_HOST_ENV)
    if configured is None:
        return process_index == 0
    local = os.uname().nodename.strip().split(".", 1)[0]
    expected = configured.strip().split(".", 1)[0]
    if not expected:
        raise ConfigError(f"{_CONTROLLER_HOST_ENV} may not be empty")
    return local == expected


def _initialize_distributed_worker(config: LocalConfig) -> int:
    expected = _expected_process_count(config)
    if not _is_cluster_worker() or expected <= 1:
        return 0
    import jax

    jax.distributed.initialize()
    actual = int(jax.process_count())
    if actual != expected:
        raise ConfigError(
            f"JAX discovered {actual} processes, but prepare configured {expected} TPU VM hosts"
        )
    return int(jax.process_index())


def _prepare_wizard(
    config: LocalConfig,
    *,
    run_diagnostics: bool,
    require_tpu: bool,
    download: bool,
    save: bool,
) -> tuple[LocalConfig, bool, bool, bool, bool]:
    style = Style(config.color)
    style.banner("interactive preparation")
    print("  Choose personal defaults. Official data/model rules remain versioned in Git.\n")
    data_path = _ask("Data cache root", config.data_path, style)
    data_profile = _choose(
        "Dataset to prepare",
        _PROFILES,
        config.data_profile,
        style,
        descriptions={
            "smoke": "tiny generated CI data",
            "dev": "one 100M-token FineWeb train shard",
            "official": (
                "classic 900M train corpus, or the smallest published scaled "
                "prefix selected by the preparation budget"
            ),
        },
    )
    artifacts = _ask("Persistent run/artifact directory", config.artifacts_path, style)
    tpu_vm_count = _ask_int("TPU VM hosts in this JAX job", config.tpu_vm_count, style)
    if tpu_vm_count > 1:
        inferred_hosts = infer_host_expression(tpu_vm_count)
        default_hosts = config.tpu_vm_hosts if config.tpu_vm_hosts else inferred_hosts
        if not default_hosts:
            default_hosts = f"tpu-worker-[0-{tpu_vm_count - 1}]"
        tpu_vm_hosts = _ask("pdsh host expression", default_hosts, style)
    else:
        tpu_vm_hosts = ""
    track = _choose("Default track", _TRACKS, config.default_track, style)
    run_profile = _choose("Default run profile", _PROFILES, data_profile, style)
    retention = _choose(
        "Checkpoint retention",
        _RETENTION,
        config.checkpoint_retention,
        style,
        descriptions={
            "all": "keep every checkpoint",
            "qualifying": "keep checkpoints at or below the target",
            "none-after-validation": "remove after harness validation",
        },
    )
    color = _choose("Terminal colors", _COLORS, config.color, style)
    target = _ask_float(
        "Smoke/development qualification target", config.target_loss, style
    )
    training_tokens = _ask_int(
        "Non-smoke corpus capacity (≤900M classic; max 74.9B)",
        config.training_tokens,
        style,
    )
    run_diagnostics = _confirm("Run environment diagnostics now", run_diagnostics, style)
    if run_diagnostics:
        require_tpu = _confirm(
            "Require a healthy Cloud TPU v4 topology on every configured VM",
            require_tpu,
            style,
        )
    save = _confirm("Save these personal defaults", save, style)
    resolved = LocalConfig(
        data_path=data_path,
        artifacts_path=artifacts,
        tpu_vm_count=tpu_vm_count,
        tpu_vm_hosts=tpu_vm_hosts,
        data_profile=data_profile,
        default_profile=run_profile,
        default_track=track,
        checkpoint_retention=retention,
        color=color,
        target_loss=target,
        training_tokens=training_tokens,
    ).validate()
    return resolved, run_diagnostics, require_tpu, download, save


def _resolve_run_name(args: argparse.Namespace, style: Style) -> str:
    """Return the run label, prompting when one was not supplied.

    Naming runs is worth a deliberate keystroke, so an interactive invocation
    always asks. Everything non-interactive -- a study loop, a pdsh worker, a
    piped shell -- silently keeps the unnamed default, because a prompt nobody
    can answer is a hang.
    """

    if args.name is not None:
        name = normalize_run_name(args.name)
        if not name:
            raise ConfigError(
                f"--name {args.name!r} contains no letters or digits to name a run by"
            )
        return name
    if _is_cluster_worker() or not sys.stdin.isatty():
        return ""
    while True:
        answer = input(
            f"  {style.text('Run name', 'bold')} "
            f"{style.text('[enter for unnamed]', 'dim')}: "
        ).strip()
        if not answer:
            return ""
        name = normalize_run_name(answer)
        if name:
            if name != answer:
                style.note(f"using {name}")
            return name
        style.note("a name needs at least one letter or digit; enter to skip")


def _ask(prompt: str, default: str, style: Style) -> str:
    while True:
        rendered = style.text(prompt, "bold")
        answer = input(f"  {rendered} {style.text(f'[{default}]', 'dim')}: ").strip()
        value = answer or default
        if value:
            return value


def _ask_float(prompt: str, default: float, style: Style) -> float:
    while True:
        raw = _ask(prompt, str(default), style)
        try:
            value = float(raw)
        except ValueError:
            print("  Enter a number.")
            continue
        if value >= 0 and value < float("inf"):
            return value
        print("  Enter a finite non-negative number.")


def _ask_int(prompt: str, default: int, style: Style) -> int:
    while True:
        raw = _ask(prompt, str(default), style)
        try:
            value = int(raw)
        except ValueError:
            print("  Enter a positive integer.")
            continue
        if value > 0:
            return value
        print("  Enter a positive integer.")


def _choose(
    prompt: str,
    choices: Sequence[str],
    default: str,
    style: Style,
    descriptions: dict[str, str] | None = None,
) -> str:
    descriptions = descriptions or {}
    print(f"  {style.text(prompt, 'bold')}")
    for index, choice in enumerate(choices, 1):
        selected = style.text("●", "cyan") if choice == default else "○"
        detail = f" — {descriptions[choice]}" if choice in descriptions else ""
        print(f"    {selected} {index}. {choice}{style.text(detail, 'dim')}")
    while True:
        answer = input(f"    {style.text(f'[{default}]', 'dim')}: ").strip()
        if not answer:
            return default
        if answer.isdigit() and 1 <= int(answer) <= len(choices):
            return choices[int(answer) - 1]
        if answer in choices:
            return answer
        print(f"    Choose 1-{len(choices)} or enter a listed name.")


def _confirm(prompt: str, default: bool, style: Style) -> bool:
    marker = "Y/n" if default else "y/N"
    while True:
        answer = input(f"  {style.text(prompt, 'bold')} {style.text(f'[{marker}]', 'dim')}: ").strip().lower()
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("  Enter y or n.")


def _progress_reporter(style: Style) -> Callable[[str, int, int], None]:
    last: dict[str, int] = {}

    def report(name: str, completed: int, total: int) -> None:
        percent = 100 if total <= 0 else min(100, int(completed * 100 / total))
        bucket = percent // 10
        previous = last.get(name)
        if completed != total and previous == bucket:
            return
        last[name] = bucket
        if sys.stdout.isatty():
            width = 20
            filled = int(width * percent / 100)
            bar = style.text("━" * filled, "green") + style.text("─" * (width - filled), "dim")
            print(
                f"\r  {name:<30} {bar} {percent:3d}% "
                f"{completed / 2**20:7.1f}/{total / 2**20:.1f} MiB",
                end="\n" if completed == total else "",
                flush=True,
            )
        elif completed == total or bucket != previous:
            print(f"  {name}: {percent}%")

    return report


def _print_prepared(prepared: PreparedDataset, style: Style) -> None:
    style.ok(f"cache ready at {prepared.root}")
    print(f"  manifest       sha256:{prepared.manifest_sha256[:12]}")
    print(f"  training       {len(prepared.train_files)} shard(s), {prepared.train_tokens:,} tokens")
    print(f"  validation     {len(prepared.validation_files)} shard(s), {prepared.validation_tokens:,} tokens")
    print(f"  fixed prefix   {prepared.validation_prefix_tokens:,} validation predictions")


def _print_fresh10(prepared: PreparedFresh10, style: Style) -> None:
    style.ok(f"fresh10 ready at {prepared.root}")
    print(f"  manifest       sha256:{prepared.manifest_sha256[:12]}")
    print(f"  domains        {len(prepared.domains)} ({', '.join(FRESH10_DOMAINS)})")
    print(f"  scored tokens  {prepared.scored_tokens:,} total")


def _reference_contract(
    profile: str,
    *,
    tier: str = "125m",
    dataset_id: str | None = None,
    tokenizer_id: str | None = None,
) -> ReferenceContract:
    default_dataset_id, default_tokenizer_id = _data_identity(profile)
    shapes = {
        "60m": (12, 6, 384),
        "125m": (12, 10, 640),
        "250m": (16, 14, 896),
        "500m": (19, 20, 1280),
        "1b": (21, 28, 1792),
    }
    if tier not in shapes:
        raise ConfigError(f"unknown model tier in reference contract: {tier!r}")
    if profile == "smoke":
        model = {
            "layers": 2,
            "heads": 2,
            "d_model": 64,
            "mlp_mult": 4,
            "normalization": "rms_norm",
            "position_encoding": "rope_base_10000",
            "mlp_activation": "gelu",
            "vocab_size": 256,
            "semantic_vocab_size": 256,
            "tied_embeddings": True,
            "tier": "smoke",
            "parameterization": "standard",
        }
        sequence = 32
    else:
        layers, heads, width = shapes[tier]
        model = {
            "layers": layers,
            "heads": heads,
            "d_model": width,
            "mlp_mult": 4,
            "normalization": "rms_norm",
            "position_encoding": "rope_base_10000",
            "mlp_activation": "gelu",
            "vocab_size": 50_304,
            "semantic_vocab_size": 50_304,
            "tied_embeddings": False,
            "tier": tier,
            "parameterization": "complete_d_p",
        }
        sequence = 1024
    return ReferenceContract(
        model_id="reference-gpt-v3-family",
        dataset_id=dataset_id or default_dataset_id,
        tokenizer_id=tokenizer_id or default_tokenizer_id,
        sequence_length=sequence,
        extra={"model": model},
    )


def _data_identity(
    profile: str, *, prepared_name: str | None = None
) -> tuple[str, str]:
    if profile == "smoke":
        return "smoke", "synthetic-byte-v1"
    return prepared_name or "fineweb10b-gpt2", "gpt2"


def _effective_target_loss(
    profile: str,
    *,
    requested: float | None,
    development_default: float,
) -> float:
    if profile == "official":
        if requested is not None and requested > OFFICIAL_TARGET_LOSS:
            raise ConfigError(
                f"official target may not be easier than {OFFICIAL_TARGET_LOSS:.2f}"
            )
        return requested if requested is not None else OFFICIAL_TARGET_LOSS
    return requested if requested is not None else development_default


def _data_provenance(
    prepared: PreparedDataset,
    *,
    profile: str,
    integrity: str,
    repo: Path,
    fresh10: PreparedFresh10 | None = None,
) -> dict[str, Any]:
    try:
        manifest_path = prepared.manifest_path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:
        manifest_path = str(prepared.manifest_path.resolve())
    manifest_file_sha256 = (
        sha256_file(prepared.manifest_path) if prepared.manifest_path.is_file() else None
    )
    result: dict[str, Any] = {
        "dataset": {
            "name": prepared.name,
            "profile": profile,
            "manifest": {
                "path": manifest_path,
                "sha256": manifest_file_sha256,
                "canonical_sha256": prepared.manifest_sha256,
            },
            "integrity": integrity,
            "train_files": [path.name for path in prepared.train_files],
            "validation_files": [path.name for path in prepared.validation_files],
            "train_tokens_available": prepared.train_tokens,
            "validation_tokens_available": prepared.validation_tokens,
            "validation_prefix_tokens": prepared.validation_prefix_tokens,
        }
    }
    if fresh10 is not None:
        try:
            fresh_manifest_path = (
                fresh10.manifest_path.resolve().relative_to(repo.resolve()).as_posix()
            )
        except ValueError:
            fresh_manifest_path = str(fresh10.manifest_path.resolve())
        result["fresh10"] = {
            "name": fresh10.name,
            "manifest": {
                "path": fresh_manifest_path,
                "sha256": sha256_file(fresh10.manifest_path),
                "canonical_sha256": fresh10.manifest_sha256,
            },
            "integrity": "sha256",
            "scored_tokens": fresh10.scored_tokens,
            "domains": {
                domain.name: {
                    "file": domain.path.name,
                    "sha256": domain.sha256,
                    "scored_tokens": domain.scored_tokens,
                }
                for domain in fresh10.domains
            },
        }
    return result


def _recorded_downstream_tokens(record: dict[str, Any]) -> dict[str, int] | None:
    """Recover the Fresh10 identity/count contract captured for a prior run."""

    provenance = record.get("provenance")
    fresh10 = provenance.get("fresh10") if isinstance(provenance, dict) else None
    domains = fresh10.get("domains") if isinstance(fresh10, dict) else None
    if domains is None:
        return None
    if not isinstance(domains, dict):
        raise HarnessError("recorded Fresh10 provenance has invalid domains")
    result: dict[str, int] = {}
    for name, row in domains.items():
        count = row.get("scored_tokens") if isinstance(row, dict) else None
        if (
            not isinstance(name, str)
            or not name
            or name.strip() != name
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
        ):
            raise HarnessError("recorded Fresh10 provenance has an invalid domain row")
        result[name] = count
    return result


def _recorded_training_tokens(record: dict[str, Any]) -> int | None:
    """Recover a token constraint without retroactively invalidating old runs."""

    constraints = record.get("constraints")
    if constraints is None:
        return None
    if not isinstance(constraints, dict):
        raise HarnessError("recorded run constraints are invalid")
    value = constraints.get("training_tokens")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HarnessError("recorded training-token constraint is invalid")
    return value


def _ensure_artifacts_inside_repo(path: Path, root: Path) -> None:
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ConfigError(
            f"artifact directory must stay on persistent storage inside the repository: {path}"
        ) from exc


def _reject_reserved_trainer_args(arguments: Sequence[str]) -> None:
    reserved = {
        "--config",
        "--output-dir",
        "--seed",
        "--track",
        "--profile",
        "--smoke",
        "--data",
        "--data-path",
        "--train-data",
        "--val-data",
        "--data-format",
        "--dataset-id",
        "--tokenizer-id",
        "--downstream-manifest",
        "--downstream-root",
        "--downstream-data",
        "--train-tokens",
        "--tier",
        "--tokens-per-parameter",
        "--base-learning-rate",
        "--study-batch-size",
        "--omit-checkpoint",
        "--study-id",
        "--study-point",
        "--study-suite-sha256",
        "--color",
    }
    for argument in arguments:
        option = argument.split("=", 1)[0]
        if option in reserved:
            raise ConfigError(
                f"{option} is controlled by the harness; set it before `--` "
                "or choose a trainer-specific option"
            )
        if option.startswith("--"):
            matches = sorted(flag for flag in reserved if flag.startswith(option))
            if matches:
                raise ConfigError(
                    f"abbreviated option {option} could override harness-controlled "
                    f"{matches[0]}; use the trainer option's complete name"
                )


def _indent(value: str) -> str:
    return "\n".join("  " + line for line in value.splitlines())


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed < float("inf"):
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed < float("inf"):
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
