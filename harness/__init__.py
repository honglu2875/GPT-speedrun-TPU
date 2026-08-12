"""Public API for the TPU speedrun competition harness."""

from .doctor import CheckResult, doctor_ok, render_doctor, run_doctor
from .errors import (
    ConfigurationError,
    HarnessError,
    RecordError,
    ResultValidationError,
    SubmissionError,
)
from .models import ReferenceContract, RunConfig, RunOutcome, ValidationResult
from .records import append_record, load_records
from .runner import run_submission
from .scoring import rank_records, render_leaderboard
from .validation import RESULT_PREFIX, SCHEMA_VERSION, parse_result_line, validate_result, verify_run

__all__ = [
    "CheckResult",
    "ConfigurationError",
    "HarnessError",
    "RESULT_PREFIX",
    "RecordError",
    "ReferenceContract",
    "ResultValidationError",
    "RunConfig",
    "RunOutcome",
    "SCHEMA_VERSION",
    "SubmissionError",
    "ValidationResult",
    "append_record",
    "doctor_ok",
    "load_records",
    "parse_result_line",
    "rank_records",
    "render_doctor",
    "render_leaderboard",
    "run_doctor",
    "run_submission",
    "validate_result",
    "verify_run",
]
