"""Concrete environment checks used by ``speedrun doctor`` and preparation."""

from __future__ import annotations

import hashlib
from importlib import metadata
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Callable

from harness.doctor import CheckResult

from .config import repo_root
from .data import DataError, verify_dataset


_EXPECTED_RUNTIME = {"jax": "0.11.0", "jaxlib": "0.11.0", "libtpu": "0.0.44.1"}


def environment_checks(
    *,
    data_path: Path | None = None,
    profile: str | None = None,
    require_tpu: bool = False,
    check_data: bool = True,
    compile_probe: bool = True,
) -> list[Callable[[], CheckResult]]:
    checks: list[Callable[[], CheckResult]] = [
        check_python,
        check_lockfile,
        check_jax_install,
        lambda: check_devices(require_tpu=require_tpu),
    ]
    if compile_probe:
        checks.append(check_compilation)
    if data_path is not None:
        checks.append(lambda: check_storage(data_path))
        if profile and check_data:
            checks.append(lambda: check_prepared_data(data_path, profile))
    return checks


def check_python() -> CheckResult:
    version = platform.python_version()
    if sys.version_info[:2] != (3, 12):
        return CheckResult(
            "Python",
            "error",
            version,
            "run `uv sync --frozen`; this project requires Python 3.12.x",
        )
    return CheckResult("Python", "ok", f"{version} ({sys.executable})")


def check_lockfile() -> CheckResult:
    root = repo_root()
    lock = root / "uv.lock"
    if not lock.is_file():
        return CheckResult("uv lock", "error", "uv.lock is missing", "run `uv lock`")
    digest = hashlib.sha256(lock.read_bytes()).hexdigest()[:12]
    return CheckResult("uv lock", "ok", f"sha256:{digest}")


def check_jax_install() -> CheckResult:
    try:
        versions = {
            name: metadata.version(name)
            for name in ("jax", "jaxlib", "libtpu")
        }
    except metadata.PackageNotFoundError as exc:
        return CheckResult(
            "JAX runtime",
            "error",
            f"missing package {exc.name}",
            "run `uv sync --frozen`",
        )
    mismatches = [
        f"{name} expected {_EXPECTED_RUNTIME[name]}, found {version}"
        for name, version in versions.items()
        if version != _EXPECTED_RUNTIME[name]
    ]
    if mismatches:
        return CheckResult(
            "JAX runtime",
            "error",
            "; ".join(mismatches),
            "run `uv sync --frozen` to restore the locked TPU environment",
        )
    return CheckResult(
        "JAX runtime",
        "ok",
        ", ".join(f"{name} {version}" for name, version in versions.items()),
    )


def check_devices(*, require_tpu: bool) -> CheckResult:
    try:
        import jax

        devices = jax.devices()
    except Exception as exc:
        return CheckResult("accelerator", "error", f"JAX device discovery failed: {exc}")
    if not devices:
        return CheckResult("accelerator", "error", "JAX reported no devices")
    platforms = sorted({device.platform for device in devices})
    kinds = sorted({str(device.device_kind) for device in devices})
    message = f"{len(devices)} device(s), {', '.join(platforms)}; {', '.join(kinds)}"
    tpu_devices = [device for device in devices if device.platform == "tpu"]
    exact_v4 = len(tpu_devices) == 4 and all(
        str(device.device_kind).strip().lower() == "tpu v4" for device in tpu_devices
    )
    single_process = True
    try:
        single_process = jax.process_count() == 1 and jax.local_device_count() == 4
    except Exception:
        pass
    if require_tpu and (not exact_v4 or not single_process):
        return CheckResult(
            "accelerator",
            "error",
            message,
            "official runs require one process with exactly four TPU v4 chips",
        )
    if not tpu_devices:
        status = "error" if require_tpu else "warning"
        return CheckResult("accelerator", status, message, "CPU is valid only for smoke runs")
    if exact_v4 and single_process:
        return CheckResult("accelerator", "ok", message)
    return CheckResult(
        "accelerator",
        "warning",
        message,
        "this does not match the official v4-8 topology",
    )


def check_compilation() -> CheckResult:
    try:
        import jax
        import jax.numpy as jnp

        started = time.perf_counter()
        value = jnp.ones((128, 128), dtype=jnp.bfloat16)
        output = jax.jit(lambda x: x @ x)(value)
        output.block_until_ready()
        if len(jax.devices()) > 1:
            collective = jax.pmap(
                lambda x: jax.lax.psum(x, "devices"), axis_name="devices"
            )
            result = collective(jnp.arange(len(jax.devices()), dtype=jnp.float32))
            result.block_until_ready()
        elapsed = time.perf_counter() - started
    except Exception as exc:
        return CheckResult("compile probe", "error", str(exc))
    return CheckResult("compile probe", "ok", f"BF16 matmul/collective in {elapsed:.2f}s")


def check_storage(path: Path) -> CheckResult:
    try:
        resolved = path.expanduser().resolve(strict=False)
        if not resolved.exists():
            parent = next(candidate for candidate in [resolved, *resolved.parents] if candidate.exists())
            usage = shutil.disk_usage(parent)
            return CheckResult(
                "data cache",
                "warning",
                f"{resolved} does not exist; {usage.free / 2**30:.1f} GiB free at {parent}",
            )
        if not resolved.is_dir():
            return CheckResult("data cache", "error", f"not a directory: {resolved}")
        usage = shutil.disk_usage(resolved)
    except (OSError, StopIteration) as exc:
        return CheckResult("data cache", "error", str(exc))
    return CheckResult(
        "data cache",
        "ok",
        f"{resolved} ({usage.free / 2**30:.1f} GiB free)",
    )


def check_prepared_data(path: Path, profile: str) -> CheckResult:
    manifest, shards = data_selection(profile)
    try:
        prepared = verify_dataset(manifest, path, train_shards=shards, verify_hash=True)
    except (DataError, OSError) as exc:
        return CheckResult("dataset", "warning", str(exc), "run `speedrun prepare`")
    return CheckResult(
        "dataset",
        "ok",
        f"{prepared.name}: {prepared.train_tokens:,} train + "
        f"{prepared.validation_tokens:,} validation tokens",
    )


def data_selection(profile: str) -> tuple[str, int]:
    if profile == "smoke":
        return "smoke", 1
    if profile == "dev":
        return "fineweb10b-gpt2", 1
    if profile == "official":
        return "fineweb10b-gpt2", 9
    raise ValueError(f"unknown profile: {profile}")


__all__ = ["data_selection", "environment_checks"]
