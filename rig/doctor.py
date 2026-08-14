"""Environment checks used by ``rig doctor`` and preparation.

The small composable check protocol lives here beside the concrete checks that
are its only caller.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import metadata
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Callable, Iterable, Literal

from .config import repo_root
from .data import DataError, verify_dataset, verify_fresh10
from .data_routing import preparation_route, resolve_preparation_manifest


CheckStatus = Literal["ok", "warning", "error"]


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: CheckStatus
    message: str
    hint: str | None = None


DoctorCheck = Callable[[], CheckResult]


def run_doctor(checks: Iterable[DoctorCheck]) -> list[CheckResult]:
    """Run infrastructure checks without coupling execution to JAX or data code."""

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


_EXPECTED_RUNTIME = {"jax": "0.11.0", "jaxlib": "0.11.0", "libtpu": "0.0.44.1"}


def environment_checks(
    *,
    data_path: Path | None = None,
    profile: str | None = None,
    require_tpu: bool = False,
    expected_process_count: int = 1,
    training_tokens: int | None = None,
    check_data: bool = True,
    compile_probe: bool = True,
) -> list[Callable[[], CheckResult]]:
    checks: list[Callable[[], CheckResult]] = [
        check_python,
        check_lockfile,
        check_jax_install,
        lambda: check_devices(
            require_tpu=require_tpu,
            expected_process_count=expected_process_count,
        ),
    ]
    if compile_probe:
        checks.append(check_compilation)
        # Runs after the compile probe, which is what establishes that the
        # distributed backend can talk at all; a failure here is then
        # unambiguously a source mismatch rather than a broken cluster.
        checks.append(check_source_attestation)
    if data_path is not None:
        checks.append(lambda: check_storage(data_path))
        if profile and check_data:
            checks.append(
                lambda: check_prepared_data(
                    data_path, profile, training_tokens=training_tokens
                )
            )
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


def check_devices(*, require_tpu: bool, expected_process_count: int = 1) -> CheckResult:
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
    exact_v4 = len(tpu_devices) == 4 * expected_process_count and all(
        str(device.device_kind).strip().lower() == "tpu v4" for device in tpu_devices
    )
    expected_topology = True
    try:
        expected_topology = (
            jax.process_count() == expected_process_count
            and jax.local_device_count() == 4
        )
    except Exception:
        pass
    if require_tpu and (not exact_v4 or not expected_topology):
        return CheckResult(
            "accelerator",
            "error",
            message,
            "official runs require four local TPU v4 chips on every configured process",
        )
    if not tpu_devices:
        status = "error" if require_tpu else "warning"
        return CheckResult("accelerator", status, message, "CPU is valid only for smoke runs")
    if exact_v4 and expected_topology:
        return CheckResult("accelerator", "ok", message)
    return CheckResult(
        "accelerator",
        "warning",
        message,
        "this does not match the configured TPU v4 topology",
    )


def check_compilation() -> CheckResult:
    try:
        import jax
        import jax.numpy as jnp
        import numpy as np
        from jax.experimental import multihost_utils
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

        started = time.perf_counter()
        value = jnp.ones((128, 128), dtype=jnp.bfloat16)
        output = jax.jit(lambda x: x @ x)(value)
        output.block_until_ready()
        if jax.process_count() > 1:
            mesh = Mesh(np.asarray(jax.devices(), dtype=object), ("data",))
            data_sharding = NamedSharding(mesh, P("data"))
            replicated = NamedSharding(mesh, P())
            local_values = np.arange(jax.local_device_count(), dtype=np.float32)
            global_values = multihost_utils.host_local_array_to_global_array(
                local_values, mesh, P("data")
            )
            collective = jax.jit(
                lambda x: jnp.sum(x),
                in_shardings=data_sharding,
                out_shardings=replicated,
            )
            result = collective(global_values)
            result.block_until_ready()
            observed = float(result.addressable_data(0))
            expected = float(jax.process_count() * sum(range(jax.local_device_count())))
            if observed != expected:
                raise RuntimeError(
                    f"global collective returned {observed}, expected {expected}"
                )
        elif len(jax.devices()) > 1:
            collective = jax.pmap(
                lambda x: jax.lax.psum(x, "devices"), axis_name="devices"
            )
            result = collective(
                jnp.arange(jax.local_device_count(), dtype=jnp.float32)
            )
            result.block_until_ready()
        elapsed = time.perf_counter() - started
    except Exception as exc:
        return CheckResult("compile probe", "error", str(exc))
    return CheckResult("compile probe", "ok", f"BF16 matmul/collective in {elapsed:.2f}s")


def source_digest(repo: Path) -> tuple[str, int]:
    """Digest the source every process must agree on.

    Covers the shared package and every submission's entry program and config
    -- exactly the bytes the workspace synchronization mirrors onto the peers.
    Paths are relative and sorted so the digest depends only on content and
    layout, never on where the checkout lives.
    """

    paths: list[Path] = []
    package = repo / "rig"
    if package.is_dir():
        paths.extend(p for p in package.rglob("*.py") if p.is_file())
    submissions = repo / "submissions"
    if submissions.is_dir():
        for name in ("train.py", "config.yaml"):
            paths.extend(p for p in submissions.glob(f"*/{name}") if p.is_file())
    paths.sort(key=lambda item: item.relative_to(repo).as_posix())
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(repo).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest(), len(paths)


def check_source_attestation() -> CheckResult:
    """Fail the preflight unless every process launched from identical source.

    The run record's provenance -- train.py, the shared package, the lockfile,
    the config -- is all hashed on the controller. Without this check a peer
    running stale bytes would still produce a result attributed to the
    controller's hashes, because the trainer's cross-process calls are
    barriers, which synchronize without comparing anything.
    """

    name = "source attestation"
    try:
        digest, files = source_digest(repo_root())
    except OSError as exc:
        return CheckResult(name, "error", f"cannot read source tree: {exc}")
    short = digest[:12]
    try:
        import jax

        if jax.process_count() <= 1:
            return CheckResult(
                name, "ok", f"sha256:{short} over {files} files (single process)"
            )
        import numpy as np
        from jax.experimental import multihost_utils

        local = np.frombuffer(bytes.fromhex(digest), dtype=np.uint8)
        gathered = np.asarray(multihost_utils.process_allgather(local))
    except Exception as exc:
        return CheckResult(name, "error", f"could not compare across hosts: {exc}")
    if gathered.ndim != 2 or gathered.shape[0] != jax.process_count():
        return CheckResult(
            name, "error", f"unexpected attestation shape {gathered.shape}"
        )
    digests = [bytes(row).hex() for row in gathered]
    disagreeing = sorted({d for d in digests if d != digests[0]})
    if disagreeing:
        groups: dict[str, list[int]] = {}
        for index, value in enumerate(digests):
            groups.setdefault(value, []).append(index)
        detail = "; ".join(
            f"sha256:{value[:12]} on process {indices}"
            for value, indices in sorted(groups.items(), key=lambda kv: kv[1][0])
        )
        return CheckResult(
            name,
            "error",
            f"processes launched from different source: {detail}",
            hint="re-run `rig prepare` to mirror the checkout onto every TPU VM",
        )
    return CheckResult(
        name,
        "ok",
        f"sha256:{short} identical across {len(digests)} processes ({files} files)",
    )


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


def check_prepared_data(
    path: Path, profile: str, *, training_tokens: int | None = None
) -> CheckResult:
    if training_tokens is None:
        manifest, shards = data_selection(profile)
        data_root = path
    else:
        route = preparation_route(profile, training_tokens)
        manifest = resolve_preparation_manifest(route)
        shards = route.train_shards
        data_root = route.data_root(path)
    try:
        prepared = verify_dataset(
            manifest, data_root, train_shards=shards, verify_hash=True
        )
    except (DataError, OSError) as exc:
        return CheckResult("dataset", "error", str(exc), "run `make prepare`")
    fresh_detail = ""
    if profile == "official":
        try:
            downstream = verify_fresh10(path, verify_hash=True)
        except (DataError, OSError) as exc:
            return CheckResult(
                "dataset",
                "error",
                f"FineWeb is ready, but Fresh10 is not: {exc}",
                "run `make prepare` with the official data profile",
            )
        fresh_detail = (
            f" + Fresh10 {len(downstream.domains)} domains / "
            f"{downstream.scored_tokens:,} scored"
        )
    return CheckResult(
        "dataset",
        "ok",
        f"{prepared.name}: {prepared.train_tokens:,} train + "
        f"{prepared.validation_tokens:,} validation tokens{fresh_detail}",
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
