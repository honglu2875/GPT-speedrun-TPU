"""Dataclasses and types shared across the competition harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Callable, Literal, Mapping, Sequence


MAX_RUN_NAME = 40
_RUN_NAME_STRIP = re.compile(r"[^a-z0-9]+")

Track = Literal["open", "sample_efficiency"]
CheckpointRetention = Literal["all", "qualifying", "none-after-validation"]
Evaluator = Callable[[Path, Mapping[str, Any]], Mapping[str, Any] | None]


@dataclass(frozen=True)
class ReferenceContract:
    """Pinned model/data identity for the sample-efficiency track.

    The harness deliberately has no invented defaults. A competition owner creates
    this value from its versioned profile/configuration and passes it to each run.
    Extra fields permit a profile to pin details beyond the common four.
    """

    model_id: str
    dataset_id: str
    tokenizer_id: str
    sequence_length: int
    extra: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "model_id": self.model_id,
            "dataset_id": self.dataset_id,
            "tokenizer_id": self.tokenizer_id,
            "sequence_length": self.sequence_length,
        }
        result.update(self.extra)
        return result


@dataclass(frozen=True)
class RunConfig:
    """Everything needed to execute one recipe reproducibly."""

    repo_root: Path
    recipe: str
    runs_dir: Path
    records_path: Path
    track: Track = "open"
    profile: str = "default"
    seed: int = 1337
    target_loss: float = 3.28
    expected_training_tokens: int | None = None
    expected_validation_tokens: int | None = None
    expected_downstream_tokens: Mapping[str, int] | None = None
    timeout_seconds: float = 900.0
    passthrough_args: Sequence[str] = ()
    reference_contract: ReferenceContract | Mapping[str, Any] | None = None
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
    evaluator_metrics: Mapping[str, Any]
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
