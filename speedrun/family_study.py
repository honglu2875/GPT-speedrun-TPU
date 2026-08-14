"""Resumable CSV-first learning-rate studies for tiered model families."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import yaml

from .config import load_config, resolve_path
from .data_routing import preparation_route, resolve_preparation_manifest


class StudyError(ValueError):
    """A study definition or recorded point is inconsistent."""


_TIERS = ("60m", "125m", "250m", "500m", "1b")
_FIELDS = (
    "study_id",
    "suite_sha256",
    "point_id",
    "status",
    "tier",
    "declared_parameters",
    "actual_parameters",
    "tokens_per_parameter",
    "planned_train_tokens",
    "actual_train_tokens",
    "batch_size",
    "sequence_length",
    "planned_steps",
    "base_learning_rate",
    "effective_global_peak_lr",
    "effective_hidden_peak_lr",
    "width_multiplier",
    "depth_multiplier",
    "data_multiplier",
    "seed",
    "run_id",
    "validation_loss",
    "validation_perplexity",
    "train_seconds",
    "tokens_per_second",
    "training_steps",
    "dataset_name",
    "dataset_manifest_sha256",
    "config_sha256",
    "result_sha256",
    "error",
)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(k, str) for k in value):
        raise StudyError(f"{label} must be a string-keyed mapping")
    return value


def _positive_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StudyError(f"{label} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise StudyError(f"{label} must be a finite positive number")
    return result


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise StudyError(f"{label} must be a positive integer")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_suite(path: Path, repo: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    try:
        payload = _mapping(yaml.safe_load(raw), "study suite")
    except yaml.YAMLError as exc:
        raise StudyError(f"invalid study YAML: {exc}") from exc
    required = {
        "schema_version",
        "study_id",
        "kind",
        "submission",
        "profile",
        "tiers",
        "admission_tiers",
        "tokens_per_parameter",
        "batch_size",
        "learning_rates",
        "seeds",
    }
    if set(payload) != required:
        missing = sorted(required - set(payload))
        unknown = sorted(set(payload) - required)
        raise StudyError(
            f"study suite keys differ (missing={missing}, unknown={unknown})"
        )
    if payload["schema_version"] != 1:
        raise StudyError("study schema_version must be 1")
    if payload["kind"] != "learning_rate_transfer":
        raise StudyError("only learning_rate_transfer suites are supported")
    for key in ("study_id", "submission"):
        value = payload[key]
        if (
            not isinstance(value, str)
            or not value
            or any(not (character.isalnum() or character in "-_") for character in value)
        ):
            raise StudyError(f"{key} must be a simple identifier")
    if payload["profile"] != "dev":
        raise StudyError("learning-rate studies must use the dev research profile")
    tiers = tuple(payload["tiers"]) if isinstance(payload["tiers"], list) else ()
    admission = (
        tuple(payload["admission_tiers"])
        if isinstance(payload["admission_tiers"], list)
        else ()
    )
    if not tiers or len(set(tiers)) != len(tiers) or any(t not in _TIERS for t in tiers):
        raise StudyError("tiers must be a non-empty unique subset of known tiers")
    if not admission or any(t not in tiers for t in admission):
        raise StudyError("admission_tiers must be a non-empty subset of tiers")
    learning_rates = (
        tuple(payload["learning_rates"])
        if isinstance(payload["learning_rates"], list)
        else ()
    )
    if len(learning_rates) < 3:
        raise StudyError("learning_rates must contain at least three points")
    rates = tuple(
        _positive_number(value, f"learning_rates[{index}]")
        for index, value in enumerate(learning_rates)
    )
    if tuple(sorted(rates)) != rates or len(set(rates)) != len(rates):
        raise StudyError("learning_rates must be strictly increasing")
    seeds = tuple(payload["seeds"]) if isinstance(payload["seeds"], list) else ()
    if not seeds or any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds
    ):
        raise StudyError("seeds must contain non-negative integers")
    submission_config = repo / "submissions" / str(payload["submission"]) / "config.yaml"
    if not submission_config.is_file() or submission_config.is_symlink():
        raise StudyError(f"submission config is unavailable: {submission_config}")
    try:
        family_config = _mapping(
            yaml.safe_load(submission_config.read_text(encoding="utf-8")),
            "submission config",
        )
        family = _mapping(family_config["family"], "submission family")
        family_tiers = _mapping(family["tiers"], "submission family tiers")
        profiles = _mapping(family_config["profiles"], "submission profiles")
        selected_profile = _mapping(
            profiles[payload["profile"]], f"submission profile {payload['profile']}"
        )
        training = _mapping(selected_profile["training"], "submission training")
    except (KeyError, yaml.YAMLError) as exc:
        raise StudyError("submission config does not define a tiered family") from exc
    declared: dict[str, int] = {}
    for tier in tiers:
        item = _mapping(family_tiers.get(tier), f"family tier {tier}")
        declared[tier] = _positive_integer(item.get("parameters"), f"{tier}.parameters")
    return {
        **payload,
        "tiers": tiers,
        "admission_tiers": admission,
        "learning_rates": rates,
        "seeds": seeds,
        "tokens_per_parameter": _positive_number(
            payload["tokens_per_parameter"], "tokens_per_parameter"
        ),
        "batch_size": _positive_integer(payload["batch_size"], "batch_size"),
        "sequence_length": _positive_integer(
            training.get("seq_len"), "submission training.seq_len"
        ),
        "declared_parameters": declared,
        "suite_sha256": _sha256_bytes(raw),
        "path": path.resolve(),
    }


def planned_rows(suite: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tier in suite["tiers"]:
        for rate_index, rate in enumerate(suite["learning_rates"]):
            for seed in suite["seeds"]:
                point_id = f"{tier}-lr{rate_index:02d}-s{seed}"
                tokens_per_step = int(suite["batch_size"]) * int(
                    suite["sequence_length"]
                )
                ideal_tokens = (
                    int(suite["declared_parameters"][tier])
                    * float(suite["tokens_per_parameter"])
                )
                planned_steps = max(
                    1, int(math.floor(ideal_tokens / tokens_per_step + 0.5))
                )
                row = {field: "" for field in _FIELDS}
                row.update(
                    {
                        "study_id": suite["study_id"],
                        "suite_sha256": suite["suite_sha256"],
                        "point_id": point_id,
                        "status": "pending",
                        "tier": tier,
                        "declared_parameters": suite["declared_parameters"][tier],
                        "tokens_per_parameter": format(
                            suite["tokens_per_parameter"], ".12g"
                        ),
                        "planned_train_tokens": planned_steps * tokens_per_step,
                        "batch_size": suite["batch_size"],
                        "sequence_length": suite["sequence_length"],
                        "planned_steps": planned_steps,
                        "base_learning_rate": format(rate, ".12g"),
                        "seed": seed,
                    }
                )
                rows.append(row)
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _read_existing(path: Path, planned: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not path.exists():
        return [dict(row) for row in planned]
    if path.is_symlink() or not path.is_file():
        raise StudyError(f"results CSV must be a regular file: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != _FIELDS:
            raise StudyError("results CSV header does not match this study schema")
        existing = [dict(row) for row in reader]
    if [row["point_id"] for row in existing] != [row["point_id"] for row in planned]:
        raise StudyError("results CSV point order differs from the immutable plan")
    for expected, actual in zip(planned, existing, strict=True):
        for field in (
            "study_id", "suite_sha256", "point_id", "tier",
            "declared_parameters", "tokens_per_parameter", "planned_train_tokens",
            "batch_size", "sequence_length", "planned_steps", "base_learning_rate",
            "seed",
        ):
            if str(actual[field]) != str(expected[field]):
                raise StudyError(f"results CSV changed immutable field {field}")
    return existing


def _records(path: Path) -> list[Mapping[str, Any]]:
    if not path.exists():
        return []
    rows: list[Mapping[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            rows.append(_mapping(json.loads(line), f"record line {number}"))
        except json.JSONDecodeError as exc:
            raise StudyError(f"invalid records JSONL line {number}: {exc}") from exc
    return rows


def _record_for_point(
    records: Sequence[Mapping[str, Any]],
    study_id: str,
    point_id: str,
    suite_sha256: str,
) -> Mapping[str, Any] | None:
    matches = []
    for record in records:
        provenance = record.get("provenance")
        if not isinstance(provenance, Mapping):
            continue
        study = provenance.get("study")
        if (
            isinstance(study, Mapping)
            and study.get("study_id") == study_id
            and study.get("point_id") == point_id
            and study.get("suite_sha256") == suite_sha256
        ):
            matches.append(record)
    if len(matches) > 1:
        raise StudyError(f"point {point_id} has multiple accepted run records")
    return matches[0] if matches else None


def _populate(
    row: dict[str, Any], record: Mapping[str, Any], artifacts: Path
) -> None:
    metrics = _mapping(record.get("metrics"), "record metrics")
    implementation = _mapping(record.get("implementation"), "record implementation")
    configuration = _mapping(implementation.get("configuration"), "configuration")
    resolved = _mapping(configuration.get("resolved"), "resolved configuration")
    training = _mapping(resolved.get("training"), "training")
    model = _mapping(resolved.get("model"), "model")
    parameterization = _mapping(resolved.get("parameterization"), "parameterization")
    optimizer = _mapping(resolved.get("optimizer"), "optimizer")
    run_id = str(record.get("run_id"))
    result_path = artifacts / run_id / "result.json"
    if not result_path.is_file() or result_path.is_symlink():
        raise StudyError(f"accepted point {row['point_id']} has no regular result.json")
    provenance = _mapping(record.get("provenance"), "record provenance")
    dataset = _mapping(provenance.get("dataset"), "dataset provenance")
    manifest = _mapping(dataset.get("manifest"), "dataset manifest provenance")
    data_multiplier = float(parameterization["data_multiplier"])
    batch_multiplier = float(parameterization["batch_multiplier"])
    width_multiplier = float(parameterization["width_multiplier"])
    depth_multiplier = float(parameterization["depth_multiplier"])
    depth_alpha = float(parameterization["depth_alpha"])
    effective = _mapping(optimizer.get("effective"), "effective optimizer")
    global_peak = float(effective["global_peak_learning_rate"])
    hidden_peak = global_peak * width_multiplier**-1 * depth_multiplier ** (
        depth_alpha - 1.0
    )
    expected = {
        "tier": (str(model.get("tier")), str(row["tier"])),
        "parameters": (
            int(metrics["parameter_count"]),
            int(row["declared_parameters"]),
        ),
        "tokens": (
            int(metrics["tokens_processed"]),
            int(row["planned_train_tokens"]),
        ),
        "steps": (int(metrics["training_steps"]), int(row["planned_steps"])),
        "batch": (int(training["batch_size"]), int(row["batch_size"])),
    }
    for label, (actual, planned) in expected.items():
        if actual != planned:
            raise StudyError(
                f"accepted point {row['point_id']} changed {label}: "
                f"planned {planned!r}, recorded {actual!r}"
            )
    if not math.isclose(
        float(optimizer["learning_rate"]),
        float(row["base_learning_rate"]),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise StudyError(
            f"accepted point {row['point_id']} changed base learning rate"
        )
    row.update(
        {
            "status": "complete",
            "actual_parameters": metrics["parameter_count"],
            "actual_train_tokens": metrics["tokens_processed"],
            "effective_global_peak_lr": format(global_peak, ".12g"),
            "effective_hidden_peak_lr": format(hidden_peak, ".12g"),
            "width_multiplier": format(width_multiplier, ".12g"),
            "depth_multiplier": format(depth_multiplier, ".12g"),
            "data_multiplier": format(data_multiplier, ".12g"),
            "run_id": run_id,
            "validation_loss": format(float(metrics["validation_loss"]), ".12g"),
            "validation_perplexity": format(
                math.exp(float(metrics["validation_loss"])), ".12g"
            ),
            "train_seconds": format(float(metrics["train_seconds"]), ".12g"),
            "tokens_per_second": format(float(metrics["tokens_per_second"]), ".12g"),
            "training_steps": metrics["training_steps"],
            "dataset_name": dataset["name"],
            "dataset_manifest_sha256": manifest["canonical_sha256"],
            "config_sha256": configuration["sha256"],
            "result_sha256": _sha256_file(result_path),
            "error": "",
        }
    )


def run_study(
    suite: Mapping[str, Any],
    *,
    repo: Path,
    results: Path,
    color: str,
    only_point: str | None = None,
) -> None:
    planned = planned_rows(suite)
    point_ids = {str(row["point_id"]) for row in planned}
    if only_point is not None and only_point not in point_ids:
        raise StudyError(
            f"unknown point {only_point!r}; expected one of {', '.join(sorted(point_ids))}"
        )
    local = load_config(repo)
    artifacts = resolve_path(local.artifacts_path, repo)
    route = preparation_route(local.data_profile, local.training_tokens)
    # Resolve here so a missing/unpublished routed manifest fails before the
    # first accelerator process is launched.
    resolve_preparation_manifest(route)
    required_tokens = max(int(row["planned_train_tokens"]) for row in planned)
    if route.train_capacity is None or route.train_capacity < required_tokens:
        raise StudyError(
            "prepared corpus is too small for a no-replacement study: "
            f"need {required_tokens:,} train tokens, selected route provides "
            f"{(route.train_capacity or 0):,}; run `make prepare "
            f"TRAIN_TOKENS={required_tokens}` first"
        )
    rows = _read_existing(results, planned)
    _write_csv(results, rows)  # CSV exists before the first accelerator launch.
    records_path = artifacts / "records.jsonl"
    for row in rows:
        if only_point is not None and row["point_id"] != only_point:
            continue
        if row["status"] == "complete":
            continue
        recorded = _record_for_point(
            _records(records_path),
            str(suite["study_id"]),
            row["point_id"],
            str(suite["suite_sha256"]),
        )
        if recorded is not None:
            _populate(row, recorded, artifacts)
            _write_csv(results, rows)
            continue
        row["status"] = "running"
        row["error"] = ""
        _write_csv(results, rows)
        command = [
            sys.executable,
            "-m",
            "speedrun",
            "run",
            str(suite["submission"]),
            "--profile",
            str(suite["profile"]),
            "--tier",
            row["tier"],
            "--tokens-per-parameter",
            str(suite["tokens_per_parameter"]),
            "--base-learning-rate",
            row["base_learning_rate"],
            "--study-batch-size",
            str(suite["batch_size"]),
            "--seed",
            str(row["seed"]),
            "--checkpoints",
            "none-after-validation",
            "--skip-data-check",
            "--omit-checkpoint",
            "--timeout",
            "21600",
            "--study-id",
            str(suite["study_id"]),
            "--study-point",
            row["point_id"],
            "--study-suite-sha256",
            str(suite["suite_sha256"]),
            "--color",
            color,
        ]
        completed = subprocess.run(command, cwd=repo, check=False)
        if completed.returncode != 0:
            row["status"] = "failed"
            row["error"] = f"speedrun exited {completed.returncode}"
            _write_csv(results, rows)
            raise StudyError(f"{row['point_id']} failed; inspect runs/ and resume")
        recorded = _record_for_point(
            _records(records_path),
            str(suite["study_id"]),
            row["point_id"],
            str(suite["suite_sha256"]),
        )
        if recorded is None:
            raise StudyError(f"{row['point_id']} completed without an accepted record")
        _populate(row, recorded, artifacts)
        _write_csv(results, rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan, run, and resume a CSV-first model-family study."
    )
    parser.add_argument("command", choices=("plan", "run"))
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--color", choices=("auto", "always", "never"), default="auto")
    parser.add_argument(
        "--only-point",
        help="run or reconcile one exact point before resuming the full suite",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    repo = Path(__file__).resolve().parent.parent
    suite_path = args.suite if args.suite.is_absolute() else repo / args.suite
    results = args.results if args.results.is_absolute() else repo / args.results
    try:
        suite = load_suite(suite_path.resolve(), repo)
        planned = planned_rows(suite)
        if args.command == "plan":
            if args.only_point is not None:
                raise StudyError("--only-point is valid only with the run command")
            _write_csv(results.resolve(), planned)
            print(f"wrote {len(planned)} planned points to {results.resolve()}")
        else:
            run_study(
                suite,
                repo=repo,
                results=results.resolve(),
                color=args.color,
                only_point=args.only_point,
            )
            scope = f"point {args.only_point}" if args.only_point else f"{len(planned)} points"
            print(f"completed {scope} in {results.resolve()}")
    except (OSError, StudyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
