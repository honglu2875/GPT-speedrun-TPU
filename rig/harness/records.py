"""Append-only JSONL persistence for immutable competition run records."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from .errors import RecordError


def canonical_json(value: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        raise RecordError(f"record is not finite JSON: {exc}") from exc


def append_record(path: Path, record: Mapping[str, Any]) -> None:
    """Append one compact record with an advisory lock and durable flush."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (canonical_json(record) + "\n").encode("utf-8")
    flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
    descriptor = os.open(path, flags, 0o644)
    try:
        try:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_EX)
        except ImportError:  # pragma: no cover - Windows fallback
            pass
        written = 0
        while written < len(encoded):
            count = os.write(descriptor, encoded[written:])
            if count <= 0:
                raise RecordError("short write while appending run record")
            written += count
        os.fsync(descriptor)
    except OSError as exc:
        raise RecordError(f"could not append record to {path}: {exc}") from exc
    finally:
        os.close(descriptor)


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RecordError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
                if not isinstance(value, dict):
                    raise RecordError(f"record at {path}:{line_number} is not an object")
                records.append(value)
    except OSError as exc:
        raise RecordError(f"could not read {path}: {exc}") from exc
    return records

