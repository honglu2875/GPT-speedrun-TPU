"""Run execution, result protocol, records, and scoring.

The trainer never imports this package. Its boundary with an entry program is
the process itself: environment variables in, a final ``RIG_RESULT=`` line on
stdout back. The doctor check protocol lives in :mod:`rig.doctor` beside the
concrete checks that use it.
"""

from .errors import (
    ConfigurationError,
    HarnessError,
    RecordError,
    ResultValidationError,
    SubmissionError,
)
from .models import (
    ReferenceContract,
    RunConfig,
    RunOutcome,
    ValidationResult,
    normalize_run_name,
)
from .records import append_record, load_records
from .runner import run_submission
from .scoring import rank_records, render_leaderboard
from .validation import RESULT_PREFIX, SCHEMA_VERSION, parse_result_line, validate_result, verify_run

__all__ = [
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
    "load_records",
    "normalize_run_name",
    "parse_result_line",
    "rank_records",
    "render_leaderboard",
    "run_submission",
    "validate_result",
    "verify_run",
]
