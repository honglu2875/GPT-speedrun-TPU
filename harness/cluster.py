"""Small, explicit ``pdsh`` helpers for multi-host TPU VM runs.

The harness remains the sole controller for run directories and records.  It
uses ``pdsh`` only as a process launcher: one identical trainer process runs on
each TPU VM and JAX supplies the process index/topology after distributed
initialization.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import socket
import subprocess
import tarfile
import tempfile
from typing import Mapping, Sequence


SSH_SETUP_GUIDANCE = (
    "Cannot reach every configured TPU VM with non-interactive SSH. Add a public "
    "key from this controller to the same user's ~/.ssh/authorized_keys on every "
    "TPU VM, verify that `pdsh -R ssh -w HOSTS hostname` succeeds, and rerun "
    "`make prepare`; speedrun never creates or distributes SSH keys."
)

_SSH_OPTIONS = (
    "BatchMode=yes",
    "ConnectTimeout=8",
    "StrictHostKeyChecking=accept-new",
)
_DEFAULT_SSH_ARGS = " ".join(f"-o {option}" for option in _SSH_OPTIONS)
_COMMON_ARCHIVE_EXCLUDES = {
    ".git",
    ".venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "profiles",
    "report.html",
}


class ClusterError(RuntimeError):
    """A configured cluster could not be inspected, synchronized, or launched."""


class ClusterAccessError(ClusterError):
    """The controller cannot authenticate to every configured host."""


@dataclass(frozen=True)
class ClusterInventory:
    """Resolved targets and their reported machine hostnames."""

    host_expression: str
    hosts: tuple[str, ...]
    remote_hosts: tuple[str, ...]
    local_host: str
    reported_hostnames: Mapping[str, str]


def infer_host_expression(host_count: int, hostname: str | None = None) -> str:
    """Infer Cloud TPU's ``...-w-[0-N]`` expression from the local hostname."""

    if host_count <= 1:
        return ""
    name = (hostname or socket.gethostname()).strip()
    marker = name.rfind("-w-")
    if marker >= 0 and name[marker + 3 :].isdigit():
        return f"{name[: marker + 3]}[0-{host_count - 1}]"
    return ""


def pdsh_environment(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return a launcher environment with safe non-interactive SSH defaults."""

    environment = dict(os.environ)
    if base is not None:
        environment.update(base)
    environment["PDSH_RCMD_TYPE"] = "ssh"
    existing = environment.get("PDSH_SSH_ARGS_APPEND", "").strip()
    if _DEFAULT_SSH_ARGS not in existing:
        existing = " ".join(item for item in (existing, _DEFAULT_SSH_ARGS) if item)
    environment["PDSH_SSH_ARGS_APPEND"] = existing
    return environment


def expand_host_expression(
    host_expression: str,
    *,
    environment: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Ask pdsh's exec backend to expand its native host-list syntax."""

    _require_program("pdsh")
    expression = _validate_host_expression(host_expression)
    try:
        completed = subprocess.run(
            ["pdsh", "-R", "exec", "-w", expression, "/bin/echo"],
            env=pdsh_environment(environment),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=15.0,
        )
    except subprocess.TimeoutExpired as exc:
        raise ClusterError("timed out while expanding the TPU VM host expression") from exc
    if completed.returncode != 0:
        raise ClusterError(_command_failure("could not expand TPU VM hosts", completed))
    hosts = tuple(
        dict.fromkeys(
            line.partition(":")[0].strip()
            for line in completed.stdout.splitlines()
            if ":" in line and line.partition(":")[0].strip()
        )
    )
    if not hosts:
        raise ClusterError("the TPU VM host expression expands to no hosts")
    return hosts


def probe_cluster(
    host_expression: str,
    host_count: int,
    *,
    environment: Mapping[str, str] | None = None,
) -> ClusterInventory:
    """Verify passwordless SSH and identify the controller in the target set."""

    if host_count <= 1:
        raise ClusterError("cluster probing requires at least two TPU VM hosts")
    hosts = expand_host_expression(host_expression, environment=environment)
    if len(hosts) != host_count:
        raise ClusterError(
            f"TPU VM host expression expands to {len(hosts)} host(s), expected {host_count}"
        )
    try:
        completed = subprocess.run(
            _pdsh_command(host_expression, host_count, "hostname", labels=True),
            env=pdsh_environment(environment),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=max(20.0, host_count * 2.0),
        )
    except subprocess.TimeoutExpired as exc:
        raise ClusterAccessError(SSH_SETUP_GUIDANCE) from exc
    if completed.returncode != 0:
        raise ClusterAccessError(SSH_SETUP_GUIDANCE)
    reported = _parse_labeled_output(completed.stdout)
    if set(reported) != set(hosts):
        raise ClusterAccessError(SSH_SETUP_GUIDANCE)

    local_name = _short_hostname(socket.gethostname())
    matches = [
        target
        for target, remote_name in reported.items()
        if _short_hostname(remote_name) == local_name
    ]
    if len(matches) != 1:
        raise ClusterError(
            "the configured TPU VM hosts must include this controller exactly once"
        )
    local_target = matches[0]
    return ClusterInventory(
        host_expression=host_expression,
        hosts=hosts,
        remote_hosts=tuple(host for host in hosts if host != local_target),
        local_host=local_target,
        reported_hostnames=reported,
    )


def sync_workspace(
    root: Path,
    inventory: ClusterInventory,
    *,
    artifacts_path: Path | None = None,
    data_path: Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> None:
    """Copy current source/config bytes to peers without caches or run artifacts."""

    if not inventory.remote_hosts:
        return
    root = root.resolve()
    excluded: set[PurePosixPath] = set()
    for ignored_path in (artifacts_path, data_path):
        if ignored_path is None:
            continue
        try:
            excluded.add(
                PurePosixPath(ignored_path.resolve().relative_to(root).as_posix())
            )
        except ValueError:
            pass

    with tempfile.TemporaryDirectory(prefix="speedrun-cluster-", dir="/tmp") as temporary:
        archive_path = Path(temporary) / "workspace.tar.gz"
        _create_workspace_archive(root, archive_path, excluded)
        remote_archive = f"/tmp/speedrun-workspace-{os.getpid()}.tar.gz"
        _copy_to_hosts(
            archive_path,
            remote_archive,
            inventory.remote_hosts,
            environment=environment,
        )
        quoted_root = shlex.quote(str(root))
        quoted_archive = shlex.quote(remote_archive)
        command = (
            f"install -d -m 755 {quoted_root} && "
            f"tar -xzf {quoted_archive} -C {quoted_root} && "
            f"rm -f {quoted_archive}"
        )
        run_pdsh(
            inventory.remote_hosts,
            command,
            environment=environment,
            labels=True,
            timeout=180.0,
        )


def bootstrap_uv(
    root: Path,
    hosts: Sequence[str],
    *,
    offline: bool = False,
    environment: Mapping[str, str] | None = None,
) -> None:
    """Install uv when missing and synchronize the frozen environment on peers."""

    if not hosts:
        return
    quoted_root = shlex.quote(str(root.resolve()))
    missing = (
        "echo 'uv is missing on a peer and --offline forbids installation' >&2; exit 2"
        if offline
        else (
            "curl -LsSf https://astral.sh/uv/install.sh | "
            "env UV_NO_MODIFY_PATH=1 sh"
        )
    )
    sync_flags = "--frozen --offline" if offline else "--frozen"
    command = (
        'UV_BIN="$HOME/.local/bin/uv"; '
        'if [ ! -x "$UV_BIN" ]; then '
        'if command -v uv >/dev/null 2>&1; then UV_BIN=uv; '
        f"else {missing}; fi; fi; "
        f"cd {quoted_root} && "
        f'"$UV_BIN" --cache-dir /tmp/uv-cache sync {sync_flags}'
    )
    run_pdsh(
        hosts,
        command,
        environment=environment,
        labels=True,
        timeout=900.0,
    )


def run_pdsh(
    hosts: Sequence[str],
    remote_command: str,
    *,
    environment: Mapping[str, str] | None = None,
    labels: bool = True,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run one command on an explicit host set and require every host to succeed."""

    if not hosts:
        raise ClusterError("pdsh target list may not be empty")
    expression = ",".join(hosts)
    try:
        completed = subprocess.run(
            _pdsh_command(expression, len(hosts), remote_command, labels=labels),
            env=pdsh_environment(environment),
            stdout=None if labels else subprocess.PIPE,
            stderr=None if labels else subprocess.PIPE,
            text=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise ClusterError("pdsh command timed out") from exc
    if completed.returncode != 0:
        raise ClusterError(_command_failure("pdsh command failed", completed))
    return completed


def build_distributed_launch_command(
    *,
    host_expression: str,
    host_count: int,
    cwd: Path,
    command: Sequence[str],
    environment: Mapping[str, str],
) -> list[str]:
    """Build a label-free pdsh command suitable for harness stdout capture."""

    if host_count <= 1:
        raise ClusterError("distributed launch requires at least two TPU VM hosts")
    _require_program("pdsh")
    assignments: list[str] = []
    for key, value in sorted(environment.items()):
        if not key or not key.replace("_", "A").isalnum() or key[0].isdigit():
            raise ClusterError(f"invalid remote environment variable name: {key!r}")
        assignments.append(f"{key}={shlex.quote(str(value))}")
    rendered = " ".join(
        [
            f"cd {shlex.quote(str(cwd.resolve()))}",
            "&&",
            "env",
            *assignments,
            shlex.join([str(item) for item in command]),
        ]
    )
    return _pdsh_command(host_expression, host_count, rendered, labels=False)


def _pdsh_command(
    host_expression: str,
    host_count: int,
    remote_command: str,
    *,
    labels: bool,
) -> list[str]:
    expression = _validate_host_expression(host_expression)
    command = [
        "pdsh",
        "-S",
        "-R",
        "ssh",
        "-f",
        str(max(1, host_count)),
        "-w",
        expression,
    ]
    if not labels:
        command.append("-N")
    command.append(remote_command)
    return command


def _copy_to_hosts(
    source: Path,
    destination: str,
    hosts: Sequence[str],
    *,
    environment: Mapping[str, str] | None,
) -> None:
    _require_program("scp")
    copy_environment = pdsh_environment(environment)

    def copy(host: str) -> tuple[str, subprocess.CompletedProcess[str]]:
        command = ["scp", "-q"]
        for option in _SSH_OPTIONS:
            command.extend(("-o", option))
        command.extend((str(source), f"{host}:{destination}"))
        completed = subprocess.run(
            command,
            env=copy_environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=180.0,
        )
        return host, completed

    try:
        with ThreadPoolExecutor(max_workers=len(hosts)) as executor:
            results = tuple(executor.map(copy, hosts))
    except subprocess.TimeoutExpired as exc:
        raise ClusterError("workspace copy timed out") from exc
    for host, completed in results:
        if completed.returncode != 0:
            raise ClusterError(
                _command_failure(f"workspace copy to {host} failed", completed)
            )


def _create_workspace_archive(
    root: Path,
    destination: Path,
    excluded_roots: set[PurePosixPath],
) -> None:
    def archive_filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        relative = PurePosixPath(info.name)
        if any(part in _COMMON_ARCHIVE_EXCLUDES for part in relative.parts):
            return None
        if any(relative == item or item in relative.parents for item in excluded_roots):
            return None
        return info

    with tarfile.open(destination, mode="w:gz", dereference=False) as archive:
        for child in sorted(root.iterdir(), key=lambda item: item.name):
            if child.name in _COMMON_ARCHIVE_EXCLUDES:
                continue
            archive.add(child, arcname=child.name, recursive=True, filter=archive_filter)


def _parse_labeled_output(output: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in output.splitlines():
        target, separator, value = line.partition(": ")
        if separator and target.strip() and value.strip():
            result[target.strip()] = value.strip()
    return result


def _short_hostname(value: str) -> str:
    return value.strip().split(".", 1)[0]


def _validate_host_expression(value: str) -> str:
    expression = value.strip()
    if not expression or any(character in expression for character in "\x00\r\n"):
        raise ClusterError("TPU VM host expression must be a non-empty single line")
    if any(character.isspace() for character in expression):
        raise ClusterError("TPU VM host expression may not contain whitespace")
    return expression


def _require_program(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        remedy = (
            "install the pdsh package on the controller"
            if name == "pdsh"
            else f"install {name} on the controller"
        )
        raise ClusterError(f"{name} is required for multi-host runs; {remedy}")
    return path


def _command_failure(
    label: str, completed: subprocess.CompletedProcess[str]
) -> str:
    detail = " ".join((completed.stderr or completed.stdout or "").strip().split())
    if len(detail) > 360:
        detail = detail[-360:]
    return f"{label} (status {completed.returncode})" + (f": {detail}" if detail else "")


__all__ = [
    "ClusterAccessError",
    "ClusterError",
    "ClusterInventory",
    "SSH_SETUP_GUIDANCE",
    "bootstrap_uv",
    "build_distributed_launch_command",
    "expand_host_expression",
    "infer_host_expression",
    "pdsh_environment",
    "probe_cluster",
    "run_pdsh",
    "sync_workspace",
]
