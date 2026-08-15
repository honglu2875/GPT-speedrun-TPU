"""Errors raised by the rig competition harness."""

from __future__ import annotations


class HarnessError(Exception):
    """Base class for expected, user-facing harness failures."""


class ConfigurationError(HarnessError):
    """The requested run configuration is invalid."""


class RecipeError(HarnessError):
    """The recipe could not be launched or did not finish correctly."""


class ResultValidationError(HarnessError):
    """The recipe result or artifact did not satisfy the protocol."""


class RecordError(HarnessError):
    """A persistent run record could not be read or written safely."""
