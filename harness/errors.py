"""Errors raised by the speedrun competition harness."""

from __future__ import annotations


class HarnessError(Exception):
    """Base class for expected, user-facing harness failures."""


class ConfigurationError(HarnessError):
    """The requested run configuration is invalid."""


class SubmissionError(HarnessError):
    """The submission could not be launched or did not finish correctly."""


class ResultValidationError(HarnessError):
    """The submission result or artifact did not satisfy the protocol."""


class RecordError(HarnessError):
    """A persistent run record could not be read or written safely."""
