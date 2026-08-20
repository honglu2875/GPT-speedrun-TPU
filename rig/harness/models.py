"""Dataclasses and types shared across the competition harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Literal, Mapping, Sequence


MAX_RUN_NAME = 40
_RUN_NAME_STRIP = re.compile(r"[^a-z0-9]+")

CheckpointRetention = Literal["always", "qualifying", "none"]


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Everything needed to execute one recipe reproducibly."""

    repo_root: Path
    recipe: str
    runs_dir: Path
    records_path: Path
    plan: Mapping[str, Any]
    profile: str = "default"
    seed: int = 1337
    target_loss: float = 3.28
    expected_validation_tokens: int | None = None
    expected_downstream_tokens: Mapping[str, int] | None = None
    timeout_seconds: float = 900.0
    trainer_args: Sequence[str] = ()
    cohort: Mapping[str, Any] | None = None
    checkpoint_retention: CheckpointRetention = "qualifying"
    python_executable: str | None = None
    environment: Mapping[str, str] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    name: str = ""
    tpu_vm_count: int = 1
    tpu_vm_hosts: str = ""
    # Orchestrate the slice without being in it. The artifact host then
    # writes runs/ on its own disk and it is pulled back afterwards.
    remote_controller: bool = False
    # ssh target of the artifact-owning VM, used to pull runs/ back.
    artifact_host: str = ""
    # What that VM reports as its own hostname. The trainer compares this
    # against socket.gethostname() to decide who writes artifacts, so it
    # must be the reported name, not the ssh target.
    artifact_hostname: str = ""
    require_checkpoint: bool = True


@dataclass(frozen=True)
class ValidationResult:
    """Normalized result returned by protocol validation."""

    payload: Mapping[str, Any]
    checkpoint_path: Path | None
    checkpoint_sha256: str | None
    checkpoint_bytes: int | None
    declared_train_seconds: float
    tokens_processed: int
    validation_loss: float
    declared_metrics: Mapping[str, Any]
    evaluations: Mapping[str, Any] | None
    artifacts: Mapping[str, Path]


@dataclass(frozen=True)
class RunOutcome:
    """Completed run and its persisted immutable record."""

    run_id: str
    run_dir: Path
    record: Mapping[str, Any]
    record_path: Path
    checkpoint_path: Path | None


def normalize_run_name(value: str) -> str:
    """Reduce a human-typed run name to a safe run-directory segment.

    Run IDs become directory names, are embedded in shell commands sent over
    pdsh, and are compared as record keys, so the accepted alphabet is
    deliberately narrow: lowercase alphanumerics joined by single hyphens.
    Anything unusable reduces to the empty string, which means "no name" rather
    than a silently mangled one.
    """

    if not isinstance(value, str):
        raise TypeError("run name must be a string")
    slug = _RUN_NAME_STRIP.sub("-", value.strip().lower()).strip("-")
    return slug[:MAX_RUN_NAME].strip("-")
