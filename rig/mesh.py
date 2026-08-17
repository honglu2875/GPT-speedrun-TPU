"""Distributed runtime, sharding, and machine description.

Every recipe faces the same problems here: initialize the JAX distributed
runtime before touching a device, work out which process is the controller,
cut a rank-local slice out of a global batch, place host arrays onto the mesh,
and describe the machine for the run record. None of it is specific to a model,
so none of it belongs in an entry program.

The rank is always ``jax.process_index()`` after
:func:`initialize_distributed_runtime`. There is no launcher ``RANK`` variable
to trust, and Cloud TPU's JAX rank order need not match the ``-w-N`` hostname
suffix, which is why the launcher marks its controller hostname explicitly.
"""

from __future__ import annotations

from typing import Any, Sequence
from importlib import metadata as importlib_metadata
import math
import os
import platform as host_platform
import socket

import jax
import jaxlib
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from rig.console import device_label


DISTRIBUTED_ENV = "RIG_DISTRIBUTED"
PROCESS_COUNT_ENV = "RIG_PROCESS_COUNT"
CONTROLLER_HOST_ENV = "RIG_CONTROLLER_HOSTNAME"


def validate_official_topology(profile: str, devices: Sequence[jax.Device]) -> None:
    """Require four TPU v4 chips per process on a coherent global mesh."""

    if profile != "official":
        return
    local_devices = tuple(jax.local_devices())
    process_count = int(jax.process_count())
    device_count = int(jax.device_count())
    platforms = sorted({str(device.platform) for device in local_devices})
    kinds = sorted({str(device.device_kind) for device in local_devices})
    is_tpu_v4 = all(
        str(device.platform).lower() == "tpu"
        and str(device.device_kind).strip().lower() == "tpu v4"
        for device in local_devices
    )
    if (
        process_count < 1
        or len(local_devices) != 4
        or device_count != 4 * process_count
        or len(devices) != device_count
        or not is_tpu_v4
    ):
        raise RuntimeError(
            "official profile requires exactly 4 local TPU v4 devices per JAX "
            "process and one coherent global device mesh; detected "
            f"process_count={process_count}, device_count={device_count}, "
            f"local_device_count={len(local_devices)}, platforms={platforms}, "
            f"device_kinds={kinds}"
        )


def system_metadata(devices: Sequence[jax.Device]) -> dict[str, Any]:
    platforms = sorted({str(device.platform) for device in devices})

    def optional_int(device: jax.Device, name: str) -> int | None:
        value = getattr(device, name, None)
        return int(value) if value is not None else None

    try:
        libtpu_version = importlib_metadata.version("libtpu")
    except importlib_metadata.PackageNotFoundError:
        libtpu_version = None
    return {
        "python_version": host_platform.python_version(),
        "jax_version": str(jax.__version__),
        "jaxlib_version": str(jaxlib.__version__),
        "libtpu_version": libtpu_version,
        "platform": platforms[0] if len(platforms) == 1 else ",".join(platforms),
        "device_count": int(jax.device_count()),
        "local_device_count": int(jax.local_device_count()),
        "process_count": int(jax.process_count()),
        "device_kinds": sorted({str(device.device_kind) for device in devices}),
        "device_ids": [optional_int(device, "id") for device in devices],
        "process_indices": [optional_int(device, "process_index") for device in devices],
    }


def inferred_peak_tflops(args: argparse.Namespace, devices: Sequence[jax.Device]) -> float | None:
    if args.peak_tflops is not None:
        return args.peak_tflops
    labels = " ".join(str(device.device_kind).lower() for device in devices)
    # A JAX-visible device on a v4-8 is one full 275-TFLOP/s TPU v4 chip. This
    # only feeds an explicitly labeled theoretical-utilization estimate.
    if "tpu" in labels and "v4" in labels:
        return 275.0 * len(devices)
    return None


def sync_tree(tree: Any) -> None:
    for value in jax.tree_util.tree_leaves(tree):
        if hasattr(value, "block_until_ready"):
            value.block_until_ready()


def initialize_distributed_runtime() -> tuple[int, int]:
    """Initialize JAX's Cloud TPU coordinator before the first device query."""

    distributed = os.environ.get(DISTRIBUTED_ENV) == "1"
    if distributed:
        # On Cloud TPU VMs JAX discovers the coordinator, process count, and
        # process id from TPU metadata. Every host must enter this call.
        jax.distributed.initialize()
    process_count = int(jax.process_count())
    process_index = int(jax.process_index())
    expected_raw = os.environ.get(PROCESS_COUNT_ENV)
    if expected_raw is not None:
        try:
            expected = int(expected_raw)
        except ValueError as exc:
            raise ValueError(f"{PROCESS_COUNT_ENV} must be a positive integer") from exc
        if expected <= 0:
            raise ValueError(f"{PROCESS_COUNT_ENV} must be a positive integer")
        if process_count != expected:
            raise RuntimeError(
                f"JAX discovered {process_count} processes, but the launcher expected "
                f"{expected}"
            )
    if not 0 <= process_index < process_count:
        raise RuntimeError(
            f"invalid JAX process index {process_index} for {process_count} processes"
        )
    return process_index, process_count


def is_controller_process(process_index: int) -> bool:
    """Keep artifacts on the host that owns the harness, regardless of JAX rank."""

    configured = os.environ.get(CONTROLLER_HOST_ENV)
    if configured is None:
        return process_index == 0
    local = socket.gethostname().strip().split(".", 1)[0]
    expected = configured.strip().split(".", 1)[0]
    if not expected:
        raise ValueError(f"{CONTROLLER_HOST_ENV} may not be empty")
    return local == expected


def local_batch_size(global_batch_size: int, process_count: int) -> int:
    if global_batch_size % process_count:
        raise ValueError(
            f"global batch size {global_batch_size} must be divisible by JAX "
            f"process count {process_count}"
        )
    return global_batch_size // process_count


def rank_local_slice(
    value: np.ndarray, process_index: int, process_count: int
) -> np.ndarray:
    """Return this process's contiguous part of a global leading batch axis."""

    size = local_batch_size(int(value.shape[0]), process_count)
    start = process_index * size
    return np.ascontiguousarray(value[start : start + size])


def put_host_local_array(
    value: np.ndarray,
    mesh: Mesh,
    partition_spec: P,
    sharding: NamedSharding,
    process_count: int,
) -> jax.Array:
    """Convert rank-local NumPy data into one globally sharded JAX array."""

    if process_count == 1:
        return jax.device_put(value, sharding)
    return multihost_utils.host_local_array_to_global_array(
        value, mesh, partition_spec
    )


def put_replicated_tree(tree: Any, mesh: Mesh, sharding: NamedSharding, process_count: int) -> Any:
    """Create identical global replicas from the same host value on every rank."""

    if process_count == 1:
        return jax.device_put(tree, sharding)
    return multihost_utils.host_local_array_to_global_array(tree, mesh, P())


def local_device_get(tree: Any) -> Any:
    """Copy replicated values through one addressable shard on multi-host JAX."""

    def fetch(value: Any) -> Any:
        if isinstance(value, jax.Array) and not value.is_fully_addressable:
            return jax.device_get(value.addressable_data(0))
        return jax.device_get(value)

    return jax.tree_util.tree_map(fetch, tree)


def finite_metric(name: str, value: float, *, positive: bool = False) -> float:
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0.0) or (not positive and value < 0.0):
        qualifier = "finite and positive" if positive else "finite and nonnegative"
        raise FloatingPointError(f"{name} must be {qualifier}, got {value!r}")
    return value
