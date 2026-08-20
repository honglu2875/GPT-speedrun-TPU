"""Shape-generic attention runtime plumbing shared by training recipes.

Recipes own the model architecture and choose the attention policy.  This
module owns the systems boundary around that choice: scale resolution, static
tile-plan selection, explicit data-axis sharding, and provenance formatting.
No dataset or model parameter tree crosses this boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P

from rig.kernels import AttentionConfig, AttentionTiles, make_causal_attention
from rig.kernels.autotune import make_runtime_key, resolve_attention_tile_plan


AttentionCallable = Callable[..., jax.Array]


@dataclass(frozen=True)
class AttentionRuntime:
    """One static attention plan resolved before real-step compilation."""

    key_digest: str | None
    resolution_source: str
    tiles: AttentionTiles | None
    tune_seconds: float

    def __post_init__(self) -> None:
        if self.resolution_source not in (
            "dense",
            "cache",
            "shipped",
            "heuristic",
            "autotuned",
        ):
            raise ValueError(
                f"invalid attention resolution source: {self.resolution_source!r}"
            )
        if not math.isfinite(self.tune_seconds) or self.tune_seconds < 0.0:
            raise ValueError("attention tune seconds must be finite and nonnegative")
        if self.resolution_source == "dense":
            if self.key_digest is not None or self.tiles is not None:
                raise ValueError("dense attention must not carry a tuning key or tiles")
        elif self.key_digest is None or self.tiles is None:
            raise ValueError("non-dense attention requires a tuning key and tile plan")


def attention_softmax_scale(rule: str, head_dim: int) -> float:
    """Resolve a recipe-selected attention scaling rule."""

    if rule == "inverse_head_dim":
        return 1.0 / float(head_dim)
    if rule == "inverse_sqrt_head_dim":
        return head_dim**-0.5
    raise ValueError(f"unsupported attention scale {rule!r}")


def attention_runtime_metadata(runtime: AttentionRuntime) -> dict[str, Any]:
    """Return JSON-safe attention tile provenance for run artifacts."""

    return {
        "key_digest": runtime.key_digest,
        "resolution_source": runtime.resolution_source,
        "tune_seconds": float(runtime.tune_seconds),
        "tiles": None if runtime.tiles is None else runtime.tiles.to_dict(),
    }


def attention_console_rows(
    runtime: AttentionRuntime,
) -> tuple[tuple[str, str], ...]:
    """Return compact, terminal-safe attention provenance rows."""

    if runtime.tiles is None:
        return (
            ("attention tuning", "not applicable (dense)"),
            ("attention plan", "not applicable"),
        )
    digest = runtime.key_digest or "unknown"
    timing = f" · {runtime.tune_seconds:.3f}s" if runtime.tune_seconds > 0.0 else ""
    tiles = runtime.tiles
    assert tiles.block_q_dkv is not None
    assert tiles.block_q_dkv_compute is not None
    assert tiles.block_kv_dkv is not None
    assert tiles.block_kv_dkv_compute is not None
    assert tiles.block_q_dq is not None
    assert tiles.block_kv_dq is not None
    assert tiles.block_kv_dq_compute is not None
    return (
        (
            "attention tuning",
            f"{runtime.resolution_source}{timing} · key {digest[:12]}",
        ),
        (
            "attention fwd",
            f"q{tiles.block_q} · kv{tiles.block_kv}/{tiles.block_kv_compute}",
        ),
        (
            "attention dK/dV",
            f"q{tiles.block_q_dkv}/{tiles.block_q_dkv_compute} · "
            f"kv{tiles.block_kv_dkv}/{tiles.block_kv_dkv_compute}",
        ),
        (
            "attention dQ",
            f"q{tiles.block_q_dq} · kv{tiles.block_kv_dq}/{tiles.block_kv_dq_compute}",
        ),
    )


def resolve_attention_runtime(
    *,
    backend: str,
    dtype: Any,
    global_batch_size: int,
    heads: int,
    sequence: int,
    head_dim: int,
    devices: Sequence[jax.Device],
) -> AttentionRuntime:
    """Resolve one deterministic tile plan before constructing ``shard_map``.

    Only static shape and runtime identity reach the resolver.  Resolution is
    a pure function of the runtime key, so every process independently derives
    identical constants without measuring kernels or communicating.
    """

    if backend == "dense":
        return AttentionRuntime(None, "dense", None, 0.0)
    if not devices:
        raise ValueError("attention tile resolution requires at least one device")
    if global_batch_size % len(devices):
        raise ValueError(
            "global batch must divide the device count before attention tuning"
        )
    local_batch = global_batch_size // len(devices)
    process_index = int(jax.process_index())
    runtime_devices = tuple(
        device
        for device in devices
        if int(getattr(device, "process_index", process_index)) == process_index
    )
    if not runtime_devices:
        raise RuntimeError("JAX reported no addressable device for this process")
    key = make_runtime_key(
        backend=backend,
        dtype=dtype,
        batch=local_batch,
        heads=heads,
        sequence=sequence,
        head_dim=head_dim,
        mode="forward_backward",
        device=runtime_devices[0],
    )
    resolved = resolve_attention_tile_plan(key)
    return AttentionRuntime(key.digest, resolved.source, resolved.tiles, 0.0)


def make_mesh_attention(
    *,
    backend: str,
    mesh: Mesh,
    tiles: AttentionTiles | None,
    softmax_scale: float,
    document_masking: bool,
) -> AttentionCallable | None:
    """Build an explicitly data-sharded Pallas attention boundary.

    Parameters and optimizer state remain replicated, while the leading batch
    axis is partitioned over ``mesh['data']``.  Each Pallas invocation receives
    the local per-chip batch and performs no attention collectives.
    """

    if backend == "dense":
        return None
    if tiles is None:
        raise ValueError("non-dense attention requires a resolved tile plan")
    local_attention = make_causal_attention(
        AttentionConfig(
            backend=backend,
            tiles=tiles,
            softmax_scale=softmax_scale,
        )
    )
    batch_partition = P("data", None, None, None)
    segment_partition = P("data", None)
    in_specs = [batch_partition, batch_partition, batch_partition]
    if document_masking:
        in_specs.append(segment_partition)
    return jax.shard_map(
        local_attention,
        mesh=mesh,
        in_specs=tuple(in_specs),
        out_specs=batch_partition,
        check_vma=False,
    )


def document_segments(tokens: jax.Array, boundary_token: int) -> jax.Array:
    """Index each token by its document within the current sequence window."""

    return jnp.cumsum(tokens == boundary_token, axis=1).astype(jnp.int32)


__all__ = (
    "AttentionCallable",
    "AttentionRuntime",
    "attention_console_rows",
    "attention_runtime_metadata",
    "attention_softmax_scale",
    "document_segments",
    "make_mesh_attention",
    "resolve_attention_runtime",
)
