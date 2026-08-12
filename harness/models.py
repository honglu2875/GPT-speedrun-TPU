"""Dataclasses and types shared across the competition harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence


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
    """Everything needed to execute one submission reproducibly."""

    repo_root: Path
    submission: str
    runs_dir: Path
    records_path: Path
    track: Track = "open"
    profile: str = "default"
    seed: int = 1337
    target_loss: float = 3.28
    expected_validation_tokens: int | None = None
    timeout_seconds: float = 900.0
    passthrough_args: Sequence[str] = ()
    reference_contract: ReferenceContract | Mapping[str, Any] | None = None
    checkpoint_retention: CheckpointRetention = "qualifying"
    python_executable: str | None = None
    environment: Mapping[str, str] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationResult:
    """Normalized result returned by protocol validation."""

    payload: Mapping[str, Any]
    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_bytes: int
    declared_train_seconds: float
    tokens_processed: int
    validation_loss: float
    declared_metrics: Mapping[str, Any]
    evaluator_metrics: Mapping[str, Any]
    artifacts: Mapping[str, Path]


@dataclass(frozen=True)
class RunOutcome:
    """Completed run and its persisted immutable record."""

    run_id: str
    run_dir: Path
    record: Mapping[str, Any]
    record_path: Path
    checkpoint_path: Path | None
