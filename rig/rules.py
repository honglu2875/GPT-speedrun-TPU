"""Small code-level constants from the versioned benchmark rules.

Dataset manifests identify a corpus's own validation split and file capacity.
The scoring prefix is a separate benchmark contract: every official cohort
scores the same number of predictions from its selected validation split.
"""

from __future__ import annotations


OFFICIAL_TARGET_LOSS = 3.28
OFFICIAL_VALIDATION_PREDICTIONS = 10_485_760


__all__ = (
    "OFFICIAL_TARGET_LOSS",
    "OFFICIAL_VALIDATION_PREDICTIONS",
)
