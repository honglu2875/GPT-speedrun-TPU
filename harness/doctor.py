"""Composable doctor/prepare checks for device, libraries, and local data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Literal


CheckStatus = Literal["ok", "warning", "error"]


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: CheckStatus
    message: str
    hint: str | None = None


DoctorCheck = Callable[[], CheckResult]


def run_doctor(checks: Iterable[DoctorCheck]) -> list[CheckResult]:
    """Run infrastructure checks without coupling the harness to JAX or data code."""

    results: list[CheckResult] = []
    for check in checks:
        try:
            result = check()
        except Exception as exc:
            name = getattr(check, "__name__", type(check).__name__)
            result = CheckResult(name=name, status="error", message=str(exc))
        if not isinstance(result, CheckResult):
            raise TypeError("doctor checks must return CheckResult")
        results.append(result)
    return results


def doctor_ok(results: Iterable[CheckResult]) -> bool:
    return all(result.status != "error" for result in results)


def render_doctor(results: Iterable[CheckResult], *, color: bool = False) -> str:
    icons = {"ok": "✓", "warning": "!", "error": "✗"}
    colors = {"ok": "\x1b[32m", "warning": "\x1b[33m", "error": "\x1b[31m"}
    lines = []
    for result in results:
        icon = icons[result.status]
        if color:
            icon = f"{colors[result.status]}{icon}\x1b[0m"
        line = f"{icon} {result.name}: {result.message}"
        if result.hint:
            line += f" ({result.hint})"
        lines.append(line)
    return "\n".join(lines)
