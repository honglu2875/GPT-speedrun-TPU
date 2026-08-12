"""Submission process lifecycle and immutable record construction."""

from __future__ import annotations

import codecs
import hashlib
import json
import math
import os
import re
import secrets
import selectors
import signal
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, TextIO

from .errors import ConfigurationError, ResultValidationError, SubmissionError
from .models import Evaluator, RunConfig, RunOutcome
from .records import append_record
from .validation import (
    parse_result_line,
    reference_contract_dict,
    sha256_file,
    validate_result,
)


_SUBMISSION_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_PROFILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_RESERVED_PASSTHROUGH_FLAGS = ("--output-dir", "--seed", "--track", "--profile")


def run_submission(config: RunConfig, *, evaluator: Evaluator | None = None) -> RunOutcome:
    """Run, validate, record, and apply checkpoint retention for one submission.

    This is process isolation for accidental mistakes, not a security sandbox. A
    submission is trusted local Python code and can access the invoking user's data.
    """

    checked = _validate_config(config)
    repo_root, submission_dir, runs_dir, records_path, configured_provenance = checked
    provenance = _collect_provenance(
        repo_root, submission_dir / "train.py", configured_provenance
    )
    run_id = _new_run_id(config.submission)
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=False, exist_ok=False)
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    result_path = run_dir / "result.json"

    command = [
        config.python_executable or sys.executable,
        str(submission_dir / "train.py"),
        "--output-dir",
        str(run_dir),
        "--seed",
        str(config.seed),
        "--track",
        config.track,
        "--profile",
        config.profile,
        *[str(argument) for argument in config.passthrough_args],
    ]
    environment = os.environ.copy()
    environment.update({str(key): str(value) for key, value in config.environment.items()})
    environment.update(
        {
            "SPEEDRUN_RUN_ID": run_id,
            "SPEEDRUN_OUTPUT_DIR": str(run_dir),
            "SPEEDRUN_TRACK": config.track,
            "SPEEDRUN_PROFILE": config.profile,
            # Every attempt receives a fresh persistent cache. This keeps cold
            # compilation reproducible and prevents run order from advantaging
            # later submissions.
            "JAX_COMPILATION_CACHE_DIR": str(run_dir / ".jax_cache"),
            "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
            "PYTHONUNBUFFERED": "1",
        }
    )
    started_at = datetime.now(timezone.utc)
    monotonic_start = time.perf_counter()
    with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
        return_code, timed_out = _run_process(
            command,
            cwd=submission_dir,
            environment=environment,
            stdout_handle=stdout_handle,
            stderr_handle=stderr_handle,
            timeout_seconds=float(config.timeout_seconds),
        )
    observed_seconds = time.perf_counter() - monotonic_start
    finished_at = datetime.now(timezone.utc)
    _discard_compilation_cache(run_dir / ".jax_cache")

    stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace")
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
    if timed_out:
        raise SubmissionError(
            f"submission timed out after {config.timeout_seconds:g}s; logs: {run_dir}"
        )
    if return_code != 0:
        tail = _tail(stderr_text)
        detail = f" ({tail})" if tail else ""
        raise SubmissionError(
            f"submission exited with status {return_code}{detail}; logs: {run_dir}"
        )

    payload = parse_result_line(stdout_text)
    _validate_payload_identity(payload, config)
    validated = validate_result(
        payload,
        run_dir=run_dir,
        track=config.track,
        reference_contract=config.reference_contract,
        expected_training_tokens=config.expected_training_tokens,
        expected_validation_tokens=config.expected_validation_tokens,
        expected_downstream_tokens=config.expected_downstream_tokens,
        evaluator=evaluator,
    )
    # Preserve the exact accepted payload independently from potentially noisy logs.
    result_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    relative_checkpoint = validated.checkpoint_path.relative_to(run_dir.resolve()).as_posix()
    contract = payload.get("contract") if isinstance(payload.get("contract"), dict) else None
    qualified = validated.validation_loss <= float(config.target_loss)
    recorded_metrics = dict(validated.declared_metrics)
    # These normalized values, rather than potentially surprising numeric JSON
    # representations in the raw payload, are the canonical scoring fields.
    recorded_metrics.update(
        {
            "train_seconds": validated.declared_train_seconds,
            "tokens_processed": validated.tokens_processed,
            "validation_loss": validated.validation_loss,
            "evaluator": dict(validated.evaluator_metrics),
        }
    )
    record: dict[str, Any] = {
        "record_version": 1,
        "run_id": run_id,
        "status": "ok",
        "qualified": qualified,
        "submission": config.submission,
        "track": config.track,
        "profile": config.profile,
        "seed": config.seed,
        "timestamps": {
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
        },
        "target_loss": float(config.target_loss),
        "constraints": {
            "training_tokens": config.expected_training_tokens,
            "validation_tokens": config.expected_validation_tokens,
        },
        "timing": {"observed_wall_seconds": observed_seconds},
        "metrics": recorded_metrics,
        "contract": contract,
        "system": payload.get("system"),
        "reference_contract": reference_contract_dict(config.reference_contract),
        "checkpoint": {
            "path": relative_checkpoint,
            "sha256": validated.checkpoint_sha256,
            "bytes": validated.checkpoint_bytes,
            "retained": True,
        },
        "artifacts": {
            name: {
                "path": path.relative_to(run_dir.resolve()).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for name, path in validated.artifacts.items()
        },
        "logs": {
            "stdout": "stdout.log",
            "stdout_sha256": _sha256_bytes(stdout_path.read_bytes()),
            "stderr": "stderr.log",
            "stderr_sha256": _sha256_bytes(stderr_path.read_bytes()),
        },
        "command": command,
        "provenance": provenance,
    }
    if validated.evaluations is not None:
        # Validation returns a JSON round-trip copy, so the immutable record keeps
        # the entire accepted evaluation block without retaining caller aliases.
        record["evaluations"] = dict(validated.evaluations)

    keep_checkpoint = config.checkpoint_retention == "all" or (
        config.checkpoint_retention == "qualifying" and qualified
    )
    checkpoint_path: Path | None = validated.checkpoint_path
    if not keep_checkpoint:
        validated.checkpoint_path.unlink()
        checkpoint_path = None
        record["checkpoint"]["retained"] = False
    append_record(records_path, record)
    return RunOutcome(
        run_id=run_id,
        run_dir=run_dir,
        record=record,
        record_path=records_path,
        checkpoint_path=checkpoint_path,
    )


class _LiveStderr:
    """Best-effort byte-preserving stderr output, including redirected text streams."""

    def __init__(self, stream: TextIO | None) -> None:
        self._stream = stream
        self._binary = getattr(stream, "buffer", None) if stream is not None else None
        self._decoder = (
            codecs.getincrementaldecoder("utf-8")(errors="replace")
            if stream is not None and self._binary is None
            else None
        )
        self._enabled = stream is not None

    def write(self, value: bytes) -> None:
        if not self._enabled:
            return
        try:
            if self._binary is not None:
                self._binary.write(value)
                self._binary.flush()
            elif self._decoder is not None and self._stream is not None:
                text = self._decoder.decode(value)
                if text:
                    self._stream.write(text)
                    self._stream.flush()
        except (OSError, TypeError, UnicodeError, ValueError):
            # A closed terminal or a test capture stream must not invalidate an
            # otherwise successful benchmark. The file capture remains canonical.
            self._enabled = False

    def finish(self) -> None:
        if not self._enabled or self._decoder is None or self._stream is None:
            return
        try:
            tail = self._decoder.decode(b"", final=True)
            if tail:
                self._stream.write(tail)
            self._stream.flush()
        except (OSError, TypeError, UnicodeError, ValueError):
            self._enabled = False


def _run_process(
    command: list[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    stdout_handle: BinaryIO,
    stderr_handle: BinaryIO,
    timeout_seconds: float,
) -> tuple[int | None, bool]:
    """Capture stdout and tee stderr without allowing either pipe to block the child."""

    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=environment,
        stdout=stdout_handle,
        stderr=subprocess.PIPE,
        start_new_session=True,
        bufsize=0,
    )
    assert process.stderr is not None  # stderr=subprocess.PIPE
    stderr_pipe = process.stderr
    descriptor = stderr_pipe.fileno()
    os.set_blocking(descriptor, False)
    selector = selectors.DefaultSelector()
    selector.register(stderr_pipe, selectors.EVENT_READ)
    live_stderr = _LiveStderr(sys.stderr)
    deadline = time.perf_counter() + timeout_seconds
    timed_out = False

    try:
        while True:
            return_code = process.poll()
            remaining = deadline - time.perf_counter()
            if return_code is None and remaining <= 0:
                timed_out = True
                _kill_process_group(process)
                return_code = process.wait()

            wait_seconds = 0.0 if return_code is not None else min(0.1, max(0.0, remaining))
            for _key, _events in selector.select(wait_seconds):
                _drain_stderr(descriptor, stderr_handle, live_stderr)

            return_code = process.poll()
            if return_code is not None:
                # poll() observing process exit means all writes by the direct child
                # are already available. A bounded final drain avoids hanging if a
                # stray descendant inherited the pipe and remains alive.
                _drain_stderr(descriptor, stderr_handle, live_stderr)
                return (None if timed_out else return_code), timed_out
    finally:
        if process.poll() is None:
            _kill_process_group(process)
            process.wait()
        selector.close()
        stderr_pipe.close()
        live_stderr.finish()


def _drain_stderr(
    descriptor: int,
    stderr_handle: BinaryIO,
    live_stderr: _LiveStderr,
    *,
    max_chunks: int = 64,
) -> None:
    for _ in range(max_chunks):
        try:
            chunk = os.read(descriptor, 64 * 1024)
        except BlockingIOError:
            return
        if not chunk:
            return
        stderr_handle.write(chunk)
        stderr_handle.flush()
        live_stderr.write(chunk)


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass
    if process.poll() is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass


def _validate_payload_identity(payload: Mapping[str, Any], config: RunConfig) -> None:
    expected = {"track": config.track, "profile": config.profile, "seed": config.seed}
    for name, expected_value in expected.items():
        actual = payload.get(name)
        if name not in payload or type(actual) is not type(expected_value) or actual != expected_value:
            raise ResultValidationError(
                f"result {name} must exactly match the run configuration: "
                f"expected {expected_value!r}, got {actual!r}"
            )
    if config.profile == "official":
        _validate_official_system(payload.get("system"))


def _validate_official_system(value: Any) -> None:
    if not isinstance(value, dict):
        raise ResultValidationError("official result system must be a JSON object")
    expected_scalars = {
        "platform": "tpu",
        "device_count": 4,
        "local_device_count": 4,
        "process_count": 1,
    }
    for name, expected in expected_scalars.items():
        actual = value.get(name)
        if type(actual) is not type(expected) or actual != expected:
            raise ResultValidationError(
                f"official result system.{name} must be {expected!r}; got {actual!r}"
            )
    kinds = value.get("device_kinds")
    if kinds != ["TPU v4"]:
        raise ResultValidationError(
            "official result system.device_kinds must be exactly ['TPU v4']"
        )


def _validate_config(
    config: RunConfig,
) -> tuple[Path, Path, Path, Path, dict[str, Any]]:
    if not _SUBMISSION_NAME.fullmatch(config.submission):
        raise ConfigurationError(
            "submission must be a simple name containing only letters, digits, '.', '_' or '-'"
        )
    if not _PROFILE_NAME.fullmatch(config.profile):
        raise ConfigurationError("profile must be a non-empty simple name")
    if config.track not in ("open", "sample_efficiency"):
        raise ConfigurationError("track must be 'open' or 'sample_efficiency'")
    if config.checkpoint_retention not in ("all", "qualifying", "none-after-validation"):
        raise ConfigurationError("invalid checkpoint retention policy")
    if isinstance(config.seed, bool) or not isinstance(config.seed, int) or config.seed < 0:
        raise ConfigurationError("seed must be a non-negative integer")
    timeout_seconds = _finite_config_number(config.timeout_seconds)
    if timeout_seconds is None or timeout_seconds <= 0:
        raise ConfigurationError("timeout_seconds must be greater than zero")
    if isinstance(config.passthrough_args, (str, bytes)) or not isinstance(
        config.passthrough_args, Sequence
    ):
        raise ConfigurationError("passthrough_args must be a sequence of arguments")
    target_loss = _finite_config_number(config.target_loss)
    if target_loss is None or target_loss < 0:
        raise ConfigurationError("target_loss must be a finite non-negative number")
    for argument in config.passthrough_args:
        rendered = str(argument)
        if "\x00" in rendered:
            raise ConfigurationError("passthrough arguments may not contain NUL bytes")
        for flag in _RESERVED_PASSTHROUGH_FLAGS:
            if rendered == flag or rendered.startswith(flag + "="):
                raise ConfigurationError(
                    f"passthrough arguments may not override reserved flag {flag}"
                )
    configured_provenance = _copy_finite_mapping(config.provenance, "provenance")

    repo_root = config.repo_root.resolve()
    if not repo_root.is_dir():
        raise ConfigurationError(f"repository root does not exist: {repo_root}")
    submissions_root = (repo_root / "submissions").resolve()
    submission_dir = (submissions_root / config.submission).resolve()
    try:
        submission_dir.relative_to(submissions_root)
    except ValueError as exc:  # defensive in addition to name regex
        raise ConfigurationError("submission path escapes submissions directory") from exc
    trainer = submission_dir / "train.py"
    if not trainer.is_file() or trainer.is_symlink():
        raise ConfigurationError(f"submission entry script not found: {trainer}")

    runs_dir = _resolve_managed_path(repo_root, config.runs_dir, "runs_dir", directory=True)
    records_path = _resolve_managed_path(repo_root, config.records_path, "records_path")
    runs_dir.mkdir(parents=True, exist_ok=True)
    records_path.parent.mkdir(parents=True, exist_ok=True)
    if config.track == "sample_efficiency" and config.reference_contract is None:
        raise ConfigurationError("sample_efficiency requires reference_contract")
    if config.expected_validation_tokens is not None and (
        isinstance(config.expected_validation_tokens, bool)
        or not isinstance(config.expected_validation_tokens, int)
        or config.expected_validation_tokens <= 0
    ):
        raise ConfigurationError("expected_validation_tokens must be a positive integer")
    if config.expected_training_tokens is not None and (
        isinstance(config.expected_training_tokens, bool)
        or not isinstance(config.expected_training_tokens, int)
        or config.expected_training_tokens <= 0
    ):
        raise ConfigurationError("expected_training_tokens must be a positive integer")
    if config.expected_downstream_tokens is not None:
        if not isinstance(config.expected_downstream_tokens, Mapping):
            raise ConfigurationError("expected_downstream_tokens must be a mapping")
        if len(config.expected_downstream_tokens) != 10:
            raise ConfigurationError(
                "expected_downstream_tokens must contain exactly 10 domains"
            )
        for name, count in config.expected_downstream_tokens.items():
            if not isinstance(name, str) or not name or name.strip() != name:
                raise ConfigurationError(
                    "expected_downstream_tokens keys must be non-empty, trimmed strings"
                )
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ConfigurationError(
                    f"expected_downstream_tokens[{name!r}] must be a positive integer"
                )
    # Validate the configured contract before launching an expensive TPU job.
    try:
        reference_contract_dict(config.reference_contract)
    except Exception as exc:
        raise ConfigurationError(str(exc)) from exc
    return repo_root, submission_dir, runs_dir, records_path, configured_provenance


def _copy_finite_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{label} must be a mapping")
    try:
        encoded = json.dumps(
            dict(value), sort_keys=True, ensure_ascii=False, allow_nan=False
        )
        copied = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{label} must contain only finite JSON values: {exc}") from exc
    if not isinstance(copied, dict):  # defensive: the input was already a mapping
        raise ConfigurationError(f"{label} must encode as a JSON object")
    return copied


def _collect_provenance(
    repo_root: Path, trainer: Path, configured: Mapping[str, Any]
) -> dict[str, Any]:
    owned_keys = {"train_py", "uv_lock", "git"}
    collisions = owned_keys.intersection(configured)
    if collisions:
        raise ConfigurationError(
            "provenance may not override harness-owned keys: "
            + ", ".join(sorted(collisions))
        )
    provenance: dict[str, Any] = {
        **dict(configured),
        "train_py": _file_provenance(repo_root, trainer),
        "uv_lock": None,
        "git": _git_provenance(repo_root),
    }
    lockfile = repo_root / "uv.lock"
    if lockfile.is_file():
        provenance["uv_lock"] = _file_provenance(repo_root, lockfile)
    return provenance


def _file_provenance(repo_root: Path, path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
    except OSError as exc:
        raise ConfigurationError(f"could not hash provenance file {path}: {exc}") from exc
    try:
        relative_path = path.relative_to(repo_root).as_posix()
    except ValueError:
        relative_path = str(path)
    return {"path": relative_path, "sha256": digest.hexdigest(), "bytes": size}


def _git_provenance(repo_root: Path) -> dict[str, Any] | None:
    head = _git_output(repo_root, "rev-parse", "--verify", "HEAD")
    status = _git_output(
        repo_root, "status", "--porcelain=v1", "--untracked-files=all"
    )
    if head is None and status is None:
        return None
    result: dict[str, Any] = {}
    if head is not None:
        result["head"] = head.decode("ascii", errors="replace").strip()
    if status is not None:
        result.update(
            {
                "dirty": bool(status),
                "status_porcelain": status.decode("utf-8", errors="replace"),
                "status_porcelain_sha256": _sha256_bytes(status),
            }
        )
    return result


def _git_output(repo_root: Path, *arguments: str) -> bytes | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout if completed.returncode == 0 else None


def _finite_config_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def _discard_compilation_cache(cache_path: Path) -> None:
    """Remove only the harness-owned per-run compilation cache, best effort."""

    try:
        if cache_path.is_symlink() or cache_path.is_file():
            cache_path.unlink()
        elif cache_path.is_dir():
            shutil.rmtree(cache_path)
    except OSError:
        # Cache cleanup must not replace the real submission outcome. The run
        # directory remains available for manual cleanup if the filesystem refuses.
        pass


def _resolve_managed_path(
    repo_root: Path, configured: Path, label: str, *, directory: bool = False
) -> Path:
    path = configured if configured.is_absolute() else repo_root / configured
    resolved = path.resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise ConfigurationError(f"{label} must be contained in the repository") from exc
    if not directory and resolved == repo_root:
        raise ConfigurationError(f"{label} must name a file below the repository root")
    return resolved


def _new_run_id(submission: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{timestamp}-{submission}-{secrets.token_hex(4)}"


def _tail(text: str, limit: int = 240) -> str:
    compact = " ".join(text.strip().split())
    return compact[-limit:]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
