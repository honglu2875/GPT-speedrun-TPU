"""Offline/autobootstrap tile selection for TPU attention kernels.

Attention tile sizes are static compilation parameters.  Consequently, a
kernel cannot benchmark alternatives *while* XLA is compiling that same
program.  This module implements the useful two-stage version instead:

* ordinary runs resolve a deterministic result from a shipped lookup table or
  a conservative shape heuristic, both pure functions of the key, so every
  process in a multi-host job derives identical tiles without communicating; and
* an explicit offline bootstrap AOT-compiles a small set of candidates against
  synthetic tensors, warms each executable, measures synchronized executions,
  and reports the fastest robust median for promotion into the lookup table.

There is deliberately no on-disk cache. A per-host cache file was the one way
two processes could select different tiles for the same SPMD program.

No dataset or model parameters are accepted by the tuning API.  The batch in
``AutotuneKey`` is the per-device batch seen by the attention kernel, not the
global data-parallel batch.  Keys include the complete runtime and shape
Keys include the complete runtime and shape fingerprint because a result from a
different TPU generation, JAX/libtpu runtime, dtype, topology, or
forward/backward mode is not safely reusable.

The tile names intentionally follow JAX SplashAttention's public ``BlockSizes``
vocabulary, with two explicit inner-compute fields needed to represent the
older JAX FlashAttention kernel losslessly.  ``AttentionTilePlan`` remains
independent of the Pallas kernel so its CPU tests do not import TPU-only
implementation details.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Literal, Protocol


KERNEL_REVISION = "tpu_flash_attention_v1"
# Refreshed for the speedrun -> rig package rename. The only change to
# tpu_flash_attention.py was two docstring references to the package name, so
# the shipped TPU v4 tile plans below remain valid measurements.
TPU_FLASH_SOURCE_SHA256 = (
    "f437acc9aff584974ee5c681427193839236e7d1df23b791ea2f5f14f0b500a5"
)
JAX_FLASH_REVISION = "jax_flash_jax_0.11.0"
JAX_FLASH_SOURCE_SHA256 = (
    "7315cba9aa7bf9e0e9f9246f4b5e786e770fed48b1ae8e8f8ca3d682e3602c63"
)
TPU_VECTOR_LANES = 128
TPU_SUBLANES = 8
TPU_V4_VMEM_BYTES = 16 * 1024 * 1024
DEFAULT_MAX_VMEM_FRACTION = 0.625

AutotuneMode = Literal["forward", "forward_backward"]


class AutotuneError(RuntimeError):
    """Base class for attention autotuning failures."""


class AutotuneSchemaError(AutotuneError):
    """Raised when a serialized tile plan, key, or record violates the schema."""


class NoSuccessfulCandidateError(AutotuneError):
    """Raised when every candidate fails to compile or execute."""

    def __init__(
        self, message: str, measurements: tuple[CandidateMeasurement, ...] = ()
    ) -> None:
        super().__init__(message)
        self.measurements = measurements


@dataclass(frozen=True, slots=True)
class AttentionTilePlan:
    """Static forward/backward attention tiles, named like SplashAttention.

    Backward fields may be omitted for a forward-only workload.  Training
    candidates require all seven backward fields.  Sizes describe the padded
    sequence extent and are multiples of TPU's 128-wide vector dimension.
    """

    block_q: int
    block_kv: int
    block_kv_compute: int
    block_q_dkv: int | None = None
    block_q_dkv_compute: int | None = None
    block_kv_dkv: int | None = None
    block_kv_dkv_compute: int | None = None
    block_q_dq: int | None = None
    block_kv_dq: int | None = None
    block_kv_dq_compute: int | None = None

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer or None")

    @property
    def has_backward_tiles(self) -> bool:
        return all(
            value is not None
            for value in (
                self.block_q_dkv,
                self.block_q_dkv_compute,
                self.block_kv_dkv,
                self.block_kv_dkv_compute,
                self.block_q_dq,
                self.block_kv_dq,
                self.block_kv_dq_compute,
            )
        )

    def to_dict(self) -> dict[str, int | None]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AttentionTilePlan:
        expected = {
            "block_q",
            "block_kv",
            "block_kv_compute",
            "block_q_dkv",
            "block_q_dkv_compute",
            "block_kv_dkv",
            "block_kv_dkv_compute",
            "block_q_dq",
            "block_kv_dq",
            "block_kv_dq_compute",
        }
        if set(value) != expected:
            raise AutotuneSchemaError(
                "tile plan fields do not match schema: "
                f"expected {sorted(expected)}, found {sorted(value)}"
            )
        converted: dict[str, int | None] = {}
        for name in expected:
            item = value[name]
            if item is not None and (
                isinstance(item, bool) or not isinstance(item, int)
            ):
                raise AutotuneSchemaError(
                    f"tile field {name!r} must be an integer or null"
                )
            converted[name] = item
        return cls(**converted)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class AutotuneKey:
    """Stable identity of a compiled attention workload.

    ``batch`` is local/per-device.  ``backend`` names the kernel implementation
    (for example ``tpu_flash_attention_v1``), while ``platform`` and
    ``device_kind`` identify the accelerator runtime.
    """

    kernel_revision: str
    implementation_hash: str
    backend: str
    platform: str
    device_kind: str
    device_count: int
    local_device_count: int
    jax_version: str
    jaxlib_version: str
    libtpu_version: str
    dtype: str
    batch: int
    heads: int
    sequence: int
    head_dim: int
    mode: AutotuneMode
    causal: bool = True
    backward_strategy: Literal["none", "separate", "fused"] = "separate"
    q_layout: str = "head_dim_minor"
    k_layout: str = "head_dim_minor"
    v_layout: str = "head_dim_minor"
    buffer_count: int = 2
    lookahead: int = 1
    exponential: str = "native"
    conditional_rescale: bool = False

    def __post_init__(self) -> None:
        for name in (
            "kernel_revision",
            "implementation_hash",
            "backend",
            "platform",
            "device_kind",
            "jax_version",
            "jaxlib_version",
            "libtpu_version",
            "dtype",
            "q_layout",
            "k_layout",
            "v_layout",
            "exponential",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if self.mode not in ("forward", "forward_backward"):
            raise ValueError(f"unsupported autotune mode: {self.mode!r}")
        for name in (
            "device_count",
            "local_device_count",
            "batch",
            "heads",
            "sequence",
            "head_dim",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if not isinstance(self.causal, bool) or not self.causal:
            raise ValueError(
                "the current TPU Flash tuner supports causal attention only"
            )
        if self.backward_strategy not in ("none", "separate", "fused"):
            raise ValueError(
                f"unsupported backward strategy: {self.backward_strategy!r}"
            )
        if self.mode == "forward" and self.backward_strategy != "none":
            raise ValueError("forward mode requires backward_strategy='none'")
        if self.mode == "forward_backward" and self.backward_strategy == "none":
            raise ValueError("forward_backward mode requires a backward strategy")
        if (
            isinstance(self.buffer_count, bool)
            or not isinstance(self.buffer_count, int)
            or self.buffer_count <= 0
            or isinstance(self.lookahead, bool)
            or not isinstance(self.lookahead, int)
            or self.lookahead < 0
        ):
            raise ValueError("buffer_count must be positive and lookahead non-negative")
        if not self.implementation_hash:
            raise ValueError("implementation_hash must not be empty")
        if not isinstance(self.conditional_rescale, bool):
            raise ValueError("conditional_rescale must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AutotuneKey:
        expected = {
            "kernel_revision",
            "implementation_hash",
            "backend",
            "platform",
            "device_kind",
            "device_count",
            "local_device_count",
            "jax_version",
            "jaxlib_version",
            "libtpu_version",
            "dtype",
            "batch",
            "heads",
            "sequence",
            "head_dim",
            "mode",
            "causal",
            "backward_strategy",
            "q_layout",
            "k_layout",
            "v_layout",
            "buffer_count",
            "lookahead",
            "exponential",
            "conditional_rescale",
        }
        if set(value) != expected:
            raise AutotuneSchemaError(
                "autotune key fields do not match schema: "
                f"expected {sorted(expected)}, found {sorted(value)}"
            )
        try:
            return cls(**dict(value))
        except (TypeError, ValueError) as exc:
            raise AutotuneSchemaError(f"invalid autotune key: {exc}") from exc


@dataclass(frozen=True, slots=True)
class CandidateMeasurement:
    tiles: AttentionTilePlan
    status: Literal["ok", "error"]
    compile_seconds: float
    samples_seconds: tuple[float, ...]
    median_seconds: float | None
    mad_seconds: float | None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.status not in ("ok", "error"):
            raise ValueError(f"invalid candidate status: {self.status!r}")
        numeric = (self.compile_seconds, *self.samples_seconds)
        if self.median_seconds is not None:
            numeric += (self.median_seconds,)
        if self.mad_seconds is not None:
            numeric += (self.mad_seconds,)
        if any(not math.isfinite(item) or item < 0.0 for item in numeric):
            raise ValueError("timings must be finite and non-negative")
        has_summary = self.median_seconds is not None and self.mad_seconds is not None
        if self.samples_seconds != () and not has_summary:
            raise ValueError("timing samples require both median and MAD")
        if self.samples_seconds == () and (
            self.median_seconds is not None or self.mad_seconds is not None
        ):
            raise ValueError("median and MAD require timing samples")
        if self.samples_seconds:
            expected_median = statistics.median(self.samples_seconds)
            expected_mad = statistics.median(
                abs(sample - expected_median) for sample in self.samples_seconds
            )
            if (
                self.median_seconds != expected_median
                or self.mad_seconds != expected_mad
            ):
                raise ValueError("median/MAD do not match timing samples")
        if self.status == "ok" and not self.samples_seconds:
            raise ValueError("successful measurement requires timing samples")
        if self.status == "ok" and self.error is not None:
            raise ValueError("successful measurement must not contain an error")
        if self.status == "error" and not self.error:
            raise ValueError("failed measurement requires an error")

    def to_dict(self) -> dict[str, Any]:
        return {
            "tiles": self.tiles.to_dict(),
            "status": self.status,
            "compile_seconds": self.compile_seconds,
            "samples_seconds": list(self.samples_seconds),
            "median_seconds": self.median_seconds,
            "mad_seconds": self.mad_seconds,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CandidateMeasurement:
        expected = {
            "tiles",
            "status",
            "compile_seconds",
            "samples_seconds",
            "median_seconds",
            "mad_seconds",
            "error",
        }
        if set(value) != expected:
            raise AutotuneSchemaError(
                "candidate measurement fields do not match schema"
            )
        status = value["status"]
        if status not in ("ok", "error"):
            raise AutotuneSchemaError(f"invalid candidate status: {status!r}")
        try:
            samples = tuple(float(sample) for sample in value["samples_seconds"])
            compile_seconds = float(value["compile_seconds"])
            median = (
                None
                if value["median_seconds"] is None
                else float(value["median_seconds"])
            )
            mad = None if value["mad_seconds"] is None else float(value["mad_seconds"])
            error = value["error"]
            if error is not None and not isinstance(error, str):
                raise TypeError("error must be a string or null")
            return cls(
                tiles=AttentionTilePlan.from_dict(value["tiles"]),
                status=status,
                compile_seconds=compile_seconds,
                samples_seconds=samples,
                median_seconds=median,
                mad_seconds=mad,
                error=error,
            )
        except (TypeError, ValueError, KeyError) as exc:
            raise AutotuneSchemaError(f"invalid candidate measurement: {exc}") from exc


@dataclass(frozen=True, slots=True)
class AutotuneRecord:
    key: AutotuneKey
    winner: AttentionTilePlan
    measurements: tuple[CandidateMeasurement, ...]
    recorded_at: str

    def __post_init__(self) -> None:
        if not self.measurements:
            raise ValueError("autotune record requires candidate measurements")
        candidates = tuple(item.tiles for item in self.measurements)
        if len(candidates) != len(set(candidates)):
            raise ValueError("autotune record contains duplicate candidates")
        if not any(
            item.status == "ok" and item.tiles == self.winner
            for item in self.measurements
        ):
            raise ValueError("winner has no successful candidate measurement")
        try:
            timestamp = datetime.fromisoformat(self.recorded_at)
        except ValueError as exc:
            raise ValueError("recorded_at must be an ISO 8601 timestamp") from exc
        if timestamp.tzinfo is None:
            raise ValueError("recorded_at timestamp must include a timezone")

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key.to_dict(),
            "winner": self.winner.to_dict(),
            "measurements": [
                measurement.to_dict() for measurement in self.measurements
            ],
            "recorded_at": self.recorded_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AutotuneRecord:
        if set(value) != {"key", "winner", "measurements", "recorded_at"}:
            raise AutotuneSchemaError("autotune record fields do not match schema")
        if not isinstance(value["recorded_at"], str):
            raise AutotuneSchemaError("recorded_at must be a string")
        try:
            measurements = tuple(
                CandidateMeasurement.from_dict(item) for item in value["measurements"]
            )
            record = cls(
                key=AutotuneKey.from_dict(value["key"]),
                winner=AttentionTilePlan.from_dict(value["winner"]),
                measurements=measurements,
                recorded_at=value["recorded_at"],
            )
            return record
        except (TypeError, ValueError, KeyError) as exc:
            raise AutotuneSchemaError(f"invalid autotune record: {exc}") from exc


@dataclass(frozen=True, slots=True)
class ResolvedTilePlan:
    tiles: AttentionTilePlan
    source: Literal["cache", "shipped", "heuristic"]
    record: AutotuneRecord | None = None


@dataclass(frozen=True, slots=True)
class _ShippedTuning:
    kernel_revision: str
    backend: str
    implementation_hash: str
    platform: str
    device_kind: str
    device_count: int
    local_device_count: int
    jax_version: str
    jaxlib_version: str
    libtpu_version: str
    dtype: str
    batch: int
    heads: int
    sequence: int
    head_dim: int
    mode: AutotuneMode
    backward_strategy: Literal["none", "separate", "fused"]
    q_layout: str
    k_layout: str
    v_layout: str
    buffer_count: int
    lookahead: int
    exponential: str
    conditional_rescale: bool
    tiles: AttentionTilePlan

    def matches(self, key: AutotuneKey) -> bool:
        return all(
            (
                key.backend == self.backend,
                key.kernel_revision == self.kernel_revision,
                key.implementation_hash == self.implementation_hash,
                key.platform == self.platform,
                key.device_kind.casefold() == self.device_kind.casefold(),
                key.device_count == self.device_count,
                key.local_device_count == self.local_device_count,
                key.jax_version == self.jax_version,
                key.jaxlib_version == self.jaxlib_version,
                key.libtpu_version == self.libtpu_version,
                key.dtype == self.dtype,
                key.batch == self.batch,
                key.heads == self.heads,
                key.sequence == self.sequence,
                key.head_dim == self.head_dim,
                key.mode == self.mode,
                key.backward_strategy == self.backward_strategy,
                key.q_layout == self.q_layout,
                key.k_layout == self.k_layout,
                key.v_layout == self.v_layout,
                key.buffer_count == self.buffer_count,
                key.lookahead == self.lookahead,
                key.exponential == self.exponential,
                key.conditional_rescale == self.conditional_rescale,
                key.causal,
            )
        )


# The JAX Flash entry is the winner of a bounded 128--1024 tile sweep on the
# locked v4-8 runtime at the exact per-chip GPT-2-small shape.  The TPU Flash
# entry is a directly measured, numerically checked seed at that same shape,
# not yet the winner of an exhaustive custom-kernel tile sweep.  Exact source
# and runtime fingerprints make upgrades fall back to cache/bootstrap or a
# heuristic rather than silently inheriting stale results.
_SHIPPED_TUNINGS = (
    _ShippedTuning(
        kernel_revision=JAX_FLASH_REVISION,
        backend="jax_flash",
        implementation_hash=JAX_FLASH_SOURCE_SHA256,
        platform="tpu",
        device_kind="TPU v4",
        device_count=4,
        local_device_count=4,
        jax_version="0.11.0",
        jaxlib_version="0.11.0",
        libtpu_version="0.0.44.1",
        dtype="bfloat16",
        batch=8,
        heads=12,
        sequence=1024,
        head_dim=64,
        mode="forward_backward",
        backward_strategy="separate",
        q_layout="head_dim_minor",
        k_layout="head_dim_minor",
        v_layout="head_dim_minor",
        buffer_count=2,
        lookahead=1,
        exponential="native",
        conditional_rescale=False,
        tiles=AttentionTilePlan(
            block_q=512,
            block_kv=512,
            block_kv_compute=256,
            block_q_dkv=512,
            block_q_dkv_compute=256,
            block_kv_dkv=512,
            block_kv_dkv_compute=256,
            block_q_dq=256,
            block_kv_dq=512,
            block_kv_dq_compute=256,
        ),
    ),
    _ShippedTuning(
        kernel_revision=KERNEL_REVISION,
        backend="tpu_flash",
        implementation_hash=TPU_FLASH_SOURCE_SHA256,
        platform="tpu",
        device_kind="TPU v4",
        device_count=4,
        local_device_count=4,
        jax_version="0.11.0",
        jaxlib_version="0.11.0",
        libtpu_version="0.0.44.1",
        dtype="bfloat16",
        batch=8,
        heads=12,
        sequence=1024,
        head_dim=64,
        mode="forward_backward",
        backward_strategy="separate",
        q_layout="head_dim_minor",
        k_layout="head_dim_minor",
        v_layout="head_dim_minor",
        buffer_count=2,
        lookahead=1,
        exponential="native",
        conditional_rescale=False,
        tiles=AttentionTilePlan(
            block_q=512,
            block_kv=512,
            block_kv_compute=256,
            block_q_dkv=512,
            block_q_dkv_compute=256,
            block_kv_dkv=512,
            block_kv_dkv_compute=256,
            block_q_dq=256,
            block_kv_dq=512,
            block_kv_dq_compute=256,
        ),
    ),
)


def _dtype_name(dtype: Any) -> str:
    # Import lazily: cache parsing and candidate policy remain cheap, CPU-only,
    # and mockable without initializing a TPU runtime.
    import jax.numpy as jnp

    return jnp.dtype(dtype).name


def kernel_implementation_hash(
    path: str | Path | None = None, *, backend: str = KERNEL_REVISION
) -> str:
    """Hash the kernel source used to invalidate stale executable timings.

    The default path is selected by backend: JAX's installed Pallas source for
    ``jax_flash`` and the sibling ``tpu_flash_attention.py`` otherwise.
    During early TPU Flash bootstrapping (before that source exists), the
    revision string still gives a stable, deliberately non-matching hash.
    """

    normalized_backend = str(backend).strip().lower()
    if path is not None:
        source_path = Path(path)
    elif normalized_backend == "jax_flash":
        from jax.experimental.pallas.ops.tpu import flash_attention

        source = getattr(flash_attention, "__file__", None)
        if source is None:
            raise AutotuneError("cannot locate JAX FlashAttention source")
        source_path = Path(source)
    else:
        source_path = Path(__file__).with_name("tpu_flash_attention.py")
    try:
        payload = source_path.read_bytes()
    except FileNotFoundError:
        payload = f"missing:{KERNEL_REVISION}".encode("utf-8")
    except OSError as exc:
        raise AutotuneError(
            f"cannot hash kernel implementation {source_path}: {exc}"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def make_runtime_key(
    *,
    backend: str,
    dtype: Any,
    batch: int,
    heads: int,
    sequence: int,
    head_dim: int,
    mode: AutotuneMode = "forward_backward",
    device: Any | None = None,
    causal: bool = True,
    implementation_path: str | Path | None = None,
    kernel_revision: str | None = None,
    backward_strategy: Literal["none", "separate", "fused"] | None = None,
    q_layout: str = "head_dim_minor",
    k_layout: str = "head_dim_minor",
    v_layout: str = "head_dim_minor",
    buffer_count: int = 2,
    lookahead: int = 1,
    exponential: str = "native",
    conditional_rescale: bool = False,
) -> AutotuneKey:
    """Fingerprint the current JAX runtime and one local attention workload."""

    import jax
    import jaxlib

    if device is None:
        devices = jax.devices()
        if not devices:
            raise AutotuneError("JAX reported no devices")
        device = devices[0]
    try:
        libtpu_version = importlib_metadata.version("libtpu")
    except importlib_metadata.PackageNotFoundError:
        libtpu_version = "not-installed"
    normalized_backend = str(backend).strip().lower()
    if kernel_revision is None:
        kernel_revision = (
            JAX_FLASH_REVISION if normalized_backend == "jax_flash" else KERNEL_REVISION
        )
    if backward_strategy is None:
        backward_strategy = "none" if mode == "forward" else "separate"
    return AutotuneKey(
        kernel_revision=kernel_revision,
        implementation_hash=kernel_implementation_hash(
            implementation_path, backend=normalized_backend
        ),
        backend=normalized_backend,
        platform=str(device.platform).strip().lower(),
        device_kind=" ".join(str(device.device_kind).split()),
        device_count=int(jax.device_count()),
        local_device_count=int(jax.local_device_count()),
        jax_version=str(jax.__version__),
        jaxlib_version=str(jaxlib.__version__),
        libtpu_version=str(libtpu_version),
        dtype=_dtype_name(dtype),
        batch=batch,
        heads=heads,
        sequence=sequence,
        head_dim=head_dim,
        mode=mode,
        causal=causal,
        backward_strategy=backward_strategy,
        q_layout=q_layout,
        k_layout=k_layout,
        v_layout=v_layout,
        buffer_count=buffer_count,
        lookahead=lookahead,
        exponential=exponential,
        conditional_rescale=conditional_rescale,
    )


def padded_sequence_length(sequence: int) -> int:
    """Return the minimal 128-aligned sequence extent used by tile policy."""

    if sequence <= 0:
        raise ValueError(f"sequence must be positive, got {sequence}")
    return math.ceil(sequence / TPU_VECTOR_LANES) * TPU_VECTOR_LANES


def is_legal_attention_tile_plan(
    tiles: AttentionTilePlan,
    *,
    sequence: int,
    head_dim: int,
    mode: AutotuneMode,
    buffer_count: int = 2,
    vmem_bytes: int = TPU_V4_VMEM_BYTES,
    max_vmem_fraction: float = DEFAULT_MAX_VMEM_FRACTION,
) -> bool:
    """Check TPU-vector alignment and exact coverage of the padded extent."""

    if mode not in ("forward", "forward_backward"):
        return False
    if head_dim <= 0 or head_dim > TPU_VECTOR_LANES or head_dim % TPU_SUBLANES:
        return False
    if buffer_count <= 0 or vmem_bytes <= 0 or not 0.0 < max_vmem_fraction <= 1.0:
        return False
    extent = padded_sequence_length(sequence)
    forward = (tiles.block_q, tiles.block_kv, tiles.block_kv_compute)
    if any(value <= 0 or value % TPU_VECTOR_LANES for value in forward):
        return False
    if tiles.block_kv_compute > tiles.block_kv:
        return False
    if tiles.block_kv % tiles.block_kv_compute:
        return False
    if extent % tiles.block_q or extent % tiles.block_kv:
        return False
    if mode == "forward":
        return estimate_attention_vmem_bytes(
            tiles,
            head_dim=head_dim,
            mode=mode,
            buffer_count=buffer_count,
        ) <= int(vmem_bytes * max_vmem_fraction)
    if not tiles.has_backward_tiles:
        return False
    assert tiles.block_q_dkv is not None
    assert tiles.block_q_dkv_compute is not None
    assert tiles.block_kv_dkv is not None
    assert tiles.block_kv_dkv_compute is not None
    assert tiles.block_q_dq is not None
    assert tiles.block_kv_dq is not None
    assert tiles.block_kv_dq_compute is not None
    backward = (
        tiles.block_q_dkv,
        tiles.block_q_dkv_compute,
        tiles.block_kv_dkv,
        tiles.block_kv_dkv_compute,
        tiles.block_q_dq,
        tiles.block_kv_dq,
        tiles.block_kv_dq_compute,
    )
    if tiles.block_q_dkv_compute > tiles.block_q_dkv:
        return False
    if tiles.block_q_dkv % tiles.block_q_dkv_compute:
        return False
    if any(value <= 0 or value % TPU_VECTOR_LANES for value in backward):
        return False
    if tiles.block_kv_dkv_compute > tiles.block_kv_dkv:
        return False
    if tiles.block_kv_dkv % tiles.block_kv_dkv_compute:
        return False
    if tiles.block_kv_dq_compute > tiles.block_kv_dq:
        return False
    if tiles.block_kv_dq % tiles.block_kv_dq_compute:
        return False
    if not all(extent % value == 0 for value in backward if value is not None):
        return False
    return estimate_attention_vmem_bytes(
        tiles,
        head_dim=head_dim,
        mode=mode,
        buffer_count=buffer_count,
    ) <= int(vmem_bytes * max_vmem_fraction)


def estimate_attention_vmem_bytes(
    tiles: AttentionTilePlan,
    *,
    head_dim: int,
    mode: AutotuneMode,
    buffer_count: int = 2,
) -> int:
    """Conservatively model per-program live VMEM for candidate rejection.

    This is a policy guard, not an XLA allocator prediction.  It counts BF16
    q/k/v staging, FP32 score/softmax/accumulator scratch, double-buffering, and
    a 25% allowance for compiler temporaries.  Keeping the estimate below 62.5%
    of v4's 16 MiB VMEM leaves room for Pallas/XLA bookkeeping and layout casts.
    Actual resource failures are still recorded by the benchmark loop.
    """

    if head_dim <= 0 or buffer_count <= 0:
        raise ValueError("head_dim and buffer_count must be positive")
    bf16 = 2
    fp32 = 4
    forward = (
        tiles.block_q * head_dim * bf16
        + buffer_count * 2 * tiles.block_kv * head_dim * bf16
        + tiles.block_q * tiles.block_kv_compute * (fp32 + bf16)
        + tiles.block_q * head_dim * (fp32 + bf16)
        + tiles.block_q * 2 * fp32
    )
    peak = forward
    if mode == "forward_backward" and tiles.has_backward_tiles:
        assert tiles.block_q_dkv is not None
        assert tiles.block_q_dkv_compute is not None
        assert tiles.block_kv_dkv is not None
        assert tiles.block_kv_dkv_compute is not None
        assert tiles.block_q_dq is not None
        assert tiles.block_kv_dq is not None
        assert tiles.block_kv_dq_compute is not None
        backward_q = max(tiles.block_q_dkv_compute, tiles.block_q_dq)
        backward_kv = max(tiles.block_kv_dkv, tiles.block_kv_dq)
        backward_compute = max(
            tiles.block_q_dkv_compute,
            tiles.block_kv_dkv_compute,
            tiles.block_kv_dq_compute,
        )
        backward = (
            3 * backward_q * head_dim * bf16  # q/o/do
            + buffer_count * 2 * backward_kv * head_dim * bf16  # k/v
            + 3 * backward_q * backward_compute * fp32  # p/dp/ds
            + backward_q * head_dim * fp32  # dq accumulator
            + 2 * backward_kv * head_dim * fp32  # dk/dv accumulators
            + backward_q * 2 * fp32  # saved softmax statistics
        )
        peak = max(peak, backward)
    return math.ceil(peak * 1.25)


def _plan(
    block_q: int,
    block_kv: int,
    block_kv_compute: int,
    *,
    mode: AutotuneMode,
) -> AttentionTilePlan:
    if mode == "forward":
        return AttentionTilePlan(block_q, block_kv, block_kv_compute)
    return AttentionTilePlan(
        block_q=block_q,
        block_kv=block_kv,
        block_kv_compute=block_kv_compute,
        block_q_dkv=block_q,
        block_q_dkv_compute=block_kv_compute,
        block_kv_dkv=block_kv,
        block_kv_dkv_compute=block_kv_compute,
        block_q_dq=block_kv_compute,
        block_kv_dq=block_kv,
        block_kv_dq_compute=block_kv_compute,
    )


def generate_attention_tile_candidates(
    *,
    sequence: int,
    head_dim: int,
    mode: AutotuneMode = "forward_backward",
    buffer_count: int = 2,
    vmem_bytes: int = TPU_V4_VMEM_BYTES,
    max_vmem_fraction: float = DEFAULT_MAX_VMEM_FRACTION,
) -> tuple[AttentionTilePlan, ...]:
    """Generate a bounded, deterministic TPU candidate set.

    The list encodes the useful neighborhood found in the v4 sweep rather than
    a combinatorial product.  This keeps first-time bootstrap compilation
    tractable.  Shapes not divisible by 128 are treated as right-padded by the
    attention wrapper; the actual sequence length remains part of the key.
    """

    shapes = (
        (128, 128, 128),
        (128, 256, 128),
        (256, 256, 128),
        (256, 512, 128),
        (512, 512, 128),
        (512, 512, 256),
        (256, 1024, 256),
        (512, 1024, 256),
        (1024, 1024, 128),
    )
    candidates = tuple(
        candidate
        for values in shapes
        if is_legal_attention_tile_plan(
            (candidate := _plan(*values, mode=mode)),
            sequence=sequence,
            head_dim=head_dim,
            mode=mode,
            buffer_count=buffer_count,
            vmem_bytes=vmem_bytes,
            max_vmem_fraction=max_vmem_fraction,
        )
    )
    if not candidates:
        raise AutotuneError(
            "no legal attention tiles for "
            f"sequence={sequence}, head_dim={head_dim}, mode={mode}"
        )
    return candidates


def heuristic_attention_tile_plan(key: AutotuneKey) -> AttentionTilePlan:
    """Return a conservative large-tile default without running benchmarks."""

    candidates = generate_attention_tile_candidates(
        sequence=key.sequence,
        head_dim=key.head_dim,
        mode=key.mode,
        buffer_count=key.buffer_count,
    )
    target = (512, 512, 256)

    def distance(candidate: AttentionTilePlan) -> tuple[int, int, int, int]:
        # Prefer the measured v4 neighborhood, then larger compute tiles and
        # larger major blocks to reduce loop/control overhead.
        return (
            abs(candidate.block_q - target[0])
            + abs(candidate.block_kv - target[1])
            + abs(candidate.block_kv_compute - target[2]),
            -candidate.block_kv_compute,
            -candidate.block_q,
            -candidate.block_kv,
        )

    return min(candidates, key=distance)


def lookup_shipped_tuning(key: AutotuneKey) -> AttentionTilePlan | None:
    for tuning in _SHIPPED_TUNINGS:
        if tuning.matches(key):
            return tuning.tiles
    return None


def resolve_attention_tile_plan(key: AutotuneKey) -> ResolvedTilePlan:
    """Resolve the exact shipped LUT, else the shape heuristic.

    Both tiers are pure functions of the key, which is what lets every process
    in a multi-host job arrive at identical tile constants without
    communicating. A per-host cache file was the one thing that could break
    that: two hosts measuring near-equal candidates could persist different
    winners and then compile divergent HLO for the same SPMD program.
    """

    if shipped := lookup_shipped_tuning(key):
        return ResolvedTilePlan(shipped, "shipped")
    return ResolvedTilePlan(heuristic_attention_tile_plan(key), "heuristic")


class CompiledRunner(Protocol):
    def __call__(self) -> Any: ...


CompileCandidate = Callable[[AttentionTilePlan], CompiledRunner]
Synchronize = Callable[[Any], None]


def _synchronize(value: Any) -> None:
    import jax

    jax.block_until_ready(value)


def _safe_error(exc: Exception) -> str:
    message = " ".join(str(exc).split())
    return f"{type(exc).__name__}: {message}"[:1_000]


def benchmark_tile_candidates(
    candidates: Iterable[AttentionTilePlan],
    compile_candidate: CompileCandidate,
    *,
    warmup_runs: int = 2,
    measured_runs: int = 7,
    synchronize: Synchronize = _synchronize,
    clock: Callable[[], float] = time.perf_counter,
    head_dim: int = TPU_VECTOR_LANES,
    relative_noise_tolerance: float = 0.01,
) -> tuple[AttentionTilePlan, tuple[CandidateMeasurement, ...]]:
    """AOT-compile and synchronously benchmark candidates with robust medians.

    ``compile_candidate`` is deliberately injected.  Production code uses
    ``make_jax_attention_compiler`` below; tests can supply a fake compiler and
    clock without initializing Pallas or a TPU.
    """

    if warmup_runs < 0:
        raise ValueError("warmup_runs must be non-negative")
    if measured_runs <= 0:
        raise ValueError("measured_runs must be positive")
    if head_dim <= 0:
        raise ValueError("head_dim must be positive")
    if not 0.0 <= relative_noise_tolerance < 1.0:
        raise ValueError("relative_noise_tolerance must be in [0, 1)")
    unique = tuple(dict.fromkeys(candidates))
    if not unique:
        raise ValueError("at least one candidate is required")
    measurements: list[CandidateMeasurement] = []
    for tiles in unique:
        compile_started = clock()
        compile_seconds = 0.0
        samples: list[float] = []
        try:
            runner = compile_candidate(tiles)
            compile_seconds = max(0.0, clock() - compile_started)
            for _ in range(warmup_runs):
                synchronize(runner())
            for _ in range(measured_runs):
                started = clock()
                synchronize(runner())
                samples.append(max(0.0, clock() - started))
            median = statistics.median(samples)
            mad = statistics.median(abs(sample - median) for sample in samples)
            measurements.append(
                CandidateMeasurement(
                    tiles=tiles,
                    status="ok",
                    compile_seconds=compile_seconds,
                    samples_seconds=tuple(samples),
                    median_seconds=median,
                    mad_seconds=mad,
                )
            )
        except Exception as exc:
            if compile_seconds == 0.0:
                compile_seconds = max(0.0, clock() - compile_started)
            median = statistics.median(samples) if samples else None
            mad = (
                statistics.median(abs(sample - median) for sample in samples)
                if median is not None
                else None
            )
            measurements.append(
                CandidateMeasurement(
                    tiles=tiles,
                    status="error",
                    compile_seconds=compile_seconds,
                    samples_seconds=tuple(samples),
                    median_seconds=median,
                    mad_seconds=mad,
                    error=_safe_error(exc),
                )
            )
    successful = [item for item in measurements if item.status == "ok"]
    if not successful:
        errors = "; ".join(item.error or "unknown error" for item in measurements)
        raise NoSuccessfulCandidateError(
            f"all attention tile candidates failed: {errors}", tuple(measurements)
        )
    fastest = min(
        successful,
        key=lambda item: (
            item.median_seconds,
            item.mad_seconds,
            item.tiles.block_q,
            item.tiles.block_kv,
            item.tiles.block_kv_compute,
        ),
    )
    assert fastest.median_seconds is not None
    assert fastest.mad_seconds is not None
    contenders: list[CandidateMeasurement] = []
    for item in successful:
        assert item.median_seconds is not None
        assert item.mad_seconds is not None
        # Treat sub-percent differences or overlapping median/MAD bands as
        # noise.  A noisy candidate does not win merely because of one lucky
        # sample; within the band we prefer the smaller, simpler executable.
        noise_band = max(
            fastest.median_seconds * relative_noise_tolerance,
            fastest.mad_seconds,
            item.mad_seconds,
        )
        if item.median_seconds <= fastest.median_seconds + noise_band:
            contenders.append(item)

    def simplicity(item: CandidateMeasurement) -> tuple[Any, ...]:
        mode: AutotuneMode = (
            "forward_backward" if item.tiles.has_backward_tiles else "forward"
        )
        return (
            estimate_attention_vmem_bytes(
                item.tiles, head_dim=head_dim, mode=mode, buffer_count=2
            ),
            item.compile_seconds,
            item.tiles.block_q,
            item.tiles.block_kv,
            item.tiles.block_kv_compute,
            item.tiles.block_q_dkv or 0,
            item.tiles.block_q_dkv_compute or 0,
            item.tiles.block_kv_dkv or 0,
            item.tiles.block_kv_dkv_compute or 0,
            item.tiles.block_q_dq or 0,
            item.tiles.block_kv_dq or 0,
            item.tiles.block_kv_dq_compute or 0,
        )

    winner = min(contenders, key=simplicity).tiles
    return winner, tuple(measurements)


AttentionFactory = Callable[[AttentionTilePlan], Callable[[Any, Any, Any], Any]]


def make_jax_attention_compiler(
    *,
    key: AutotuneKey,
    attention_factory: AttentionFactory,
    device: Any | None = None,
    seed: int = 17,
) -> CompileCandidate:
    """Build an AOT compiler over fixed synthetic BHSD tensors.

    In ``forward_backward`` mode the timed executable returns attention output
    plus VJP gradients for q/k/v under a dense synthetic output cotangent.  This
    is closer to a transformer training step than differentiating a scalar norm
    that the compiler could fuse specially.
    """

    import jax
    import jax.numpy as jnp

    if device is None:
        devices = jax.devices(key.platform)
        if not devices:
            raise AutotuneError(f"JAX reported no {key.platform!r} devices")
        device = devices[0]
    shape = (key.batch, key.heads, key.sequence, key.head_dim)
    dtype = jnp.dtype(key.dtype)
    random_keys = jax.random.split(jax.random.key(seed), 4)

    def synthetic(random_key: Any) -> Any:
        value = jax.random.normal(random_key, shape, dtype=dtype) * jnp.asarray(
            0.02, dtype
        )
        return jax.device_put(value, device)

    q, k, v, output_cotangent = tuple(synthetic(item) for item in random_keys)
    jax.block_until_ready((q, k, v, output_cotangent))

    def compile_candidate(tiles: AttentionTilePlan) -> CompiledRunner:
        attention = attention_factory(tiles)
        if key.mode == "forward":

            def step(q_value: Any, k_value: Any, v_value: Any, unused: Any) -> Any:
                del unused
                return attention(q_value, k_value, v_value)

        else:

            def step(q_value: Any, k_value: Any, v_value: Any, cotangent: Any) -> Any:
                output, pullback = jax.vjp(attention, q_value, k_value, v_value)
                dq, dk, dv = pullback(cotangent)
                return output, dq, dk, dv

        # ``lower().compile()`` is the intentional AOT boundary.  The tile plan
        # is closed over before lowering and cannot change inside this program.
        compiled = (
            jax.jit(step, device=device).lower(q, k, v, output_cotangent).compile()
        )

        def run() -> Any:
            return compiled(q, k, v, output_cotangent)

        return run

    return compile_candidate


def autotune_attention(
    *,
    key: AutotuneKey,
    attention_factory: AttentionFactory,
    candidates: Iterable[AttentionTilePlan] | None = None,
    device: Any | None = None,
    warmup_runs: int = 2,
    measured_runs: int = 7,
    seed: int = 17,
) -> AutotuneRecord:
    """Measure one workload offline and return the winning tile plan.

    Deliberately not wired into the trainer: use it to establish a plan for a
    new topology, then promote the winner to ``_SHIPPED_TUNINGS`` so every
    process derives it identically from the key.
    """

    if candidates is None:
        candidates = generate_attention_tile_candidates(
            sequence=key.sequence,
            head_dim=key.head_dim,
            mode=key.mode,
            buffer_count=key.buffer_count,
        )
    candidates = tuple(candidates)
    illegal = [
        tiles
        for tiles in candidates
        if not is_legal_attention_tile_plan(
            tiles,
            sequence=key.sequence,
            head_dim=key.head_dim,
            mode=key.mode,
            buffer_count=key.buffer_count,
        )
    ]
    if illegal:
        raise ValueError(f"illegal attention tile candidate(s): {illegal}")
    compiler = make_jax_attention_compiler(
        key=key,
        attention_factory=attention_factory,
        device=device,
        seed=seed,
    )
    winner, measurements = benchmark_tile_candidates(
        candidates,
        compiler,
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
        head_dim=key.head_dim,
    )
    return AutotuneRecord(
        key=key,
        winner=winner,
        measurements=measurements,
        recorded_at=datetime.now(timezone.utc).isoformat(),
    )


__all__ = (
    "AttentionTilePlan",
    "AutotuneSchemaError",
    "AutotuneError",
    "AutotuneKey",
    "AutotuneMode",
    "AutotuneRecord",
    "DEFAULT_MAX_VMEM_FRACTION",
    "JAX_FLASH_REVISION",
    "JAX_FLASH_SOURCE_SHA256",
    "CandidateMeasurement",
    "KERNEL_REVISION",
    "NoSuccessfulCandidateError",
    "ResolvedTilePlan",
    "TPU_FLASH_SOURCE_SHA256",
    "TPU_V4_VMEM_BYTES",
    "autotune_attention",
    "benchmark_tile_candidates",
    "estimate_attention_vmem_bytes",
    "generate_attention_tile_candidates",
    "heuristic_attention_tile_plan",
    "is_legal_attention_tile_plan",
    "kernel_implementation_hash",
    "lookup_shipped_tuning",
    "make_jax_attention_compiler",
    "make_runtime_key",
    "padded_sequence_length",
    "resolve_attention_tile_plan",
)
