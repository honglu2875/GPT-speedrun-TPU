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
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import time
from typing import Mapping, Sequence


SSH_SETUP_GUIDANCE = (
    "Cannot reach every configured TPU VM with non-interactive SSH. Add a public "
    "key from this controller to the same user's ~/.ssh/authorized_keys on every "
    "TPU VM, verify that `pdsh -R ssh -w HOSTS hostname` succeeds, and rerun "
    "`make prepare`; rig never creates or distributes SSH keys."
)
RAM_CACHE_SETUP_GUIDANCE = (
    "/dev/shm must be a writable tmpfs or ramfs on every configured TPU VM. "
    "On each host reported above, mount it there (for example, "
    "`sudo install -d -m 1777 /dev/shm && sudo mount -t tmpfs -o "
    "rw,nosuid,nodev,mode=1777 tmpfs /dev/shm`) or remount it writable, then "
    "rerun `make prepare`."
)
RAM_CACHE_PROTECTION_GUIDANCE = (
    "The multi-host RAM cache needs non-interactive root privilege on every "
    "configured TPU VM. `make prepare` uses `sudo -n` only to create and protect "
    "its dedicated /dev/shm/.speedrun-cache directory, so systemd-logind "
    "RemoveIPC cannot erase the dataset when a pdsh SSH session ends. Configure "
    "passwordless sudo for those cache ownership commands, then rerun "
    "`make prepare`; rig does not change the host-wide RemoveIPC policy."
)
RSYNC_SETUP_GUIDANCE = (
    "rsync is required on every configured TPU VM. Automatic installation with "
    "non-interactive apt-get failed; install the `rsync` package on the hosts "
    "reported above, then rerun `make prepare`."
)

_SSH_OPTIONS = (
    "BatchMode=yes",
    "ConnectTimeout=8",
    # TPU VM sshd occasionally closes a key exchange while several peers are
    # contacted together. Give OpenSSH several chances without weakening
    # authentication; idempotent orchestration commands also retry status 255.
    "ConnectionAttempts=4",
    # Warmed by probe_cluster one host at a time. Subsequent pdsh/rsync
    # commands multiplex over these sockets instead of bursting simultaneous
    # key exchanges at the TPU VM sshd.
    "ControlMaster=auto",
    "ControlPersist=600",
    f"ControlPath=/tmp/rig-ssh-{os.getuid()}-%C",
    "ServerAliveInterval=15",
    "ServerAliveCountMax=3",
    "StrictHostKeyChecking=accept-new",
)
_DEFAULT_SSH_ARGS = " ".join(f"-o {option}" for option in _SSH_OPTIONS)
_PROBE_ATTEMPTS = 4
# Deliberately keeps its pre-rename name: this directory holds the prepared
# corpus already installed on every TPU VM. Renaming it here would orphan that
# cache and force a full re-download on each host.
RAM_CACHE_ROOT = Path("/dev/shm/.speedrun-cache")
_COMMON_RSYNC_EXCLUDES = (
    ".git/",
    ".venv/",
    ".mypy_cache/",
    ".pytest_cache/",
    ".ruff_cache/",
    "__pycache__/",
    "shm",
    "runs/",
    "profiles/",
    "report.html",
)


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
    reported: dict[str, str] = {}
    for host in hosts:
        last_error: Exception | None = None
        for attempt in range(_PROBE_ATTEMPTS):
            try:
                completed = subprocess.run(
                    _ssh_command(host, "hostname"),
                    env=pdsh_environment(environment),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                    timeout=20.0,
                )
            except subprocess.TimeoutExpired as exc:
                last_error = exc
            else:
                remote_name = completed.stdout.strip()
                if completed.returncode == 0 and remote_name:
                    reported[host] = remote_name
                    break
                last_error = ClusterAccessError(SSH_SETUP_GUIDANCE)
            if attempt + 1 < _PROBE_ATTEMPTS:
                # TPU VM sshd can close a connection during key exchange.
                # Retrying this read-only probe is always safe.
                time.sleep(1.0)
        else:
            raise ClusterAccessError(SSH_SETUP_GUIDANCE) from last_error

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
    """Incrementally copy source/config bytes without deleting peer-only files."""

    if not inventory.remote_hosts:
        return
    root = root.resolve()
    bootstrap_rsync(inventory, environment=environment)
    excluded = set(_COMMON_RSYNC_EXCLUDES)
    for ignored_path in (artifacts_path, data_path):
        if ignored_path is None:
            continue
        try:
            relative = ignored_path.resolve().relative_to(root).as_posix()
        except ValueError:
            continue
        excluded.add(f"/{relative}/")

    run_pdsh(
        inventory.remote_hosts,
        f"install -d -m 755 {shlex.quote(str(root))}",
        environment=environment,
        labels=True,
        timeout=60.0,
    )
    _rsync_to_hosts(
        root,
        inventory.remote_hosts,
        tuple(sorted(excluded)),
        environment=environment,
    )


def prepare_ram_cache(
    root: Path,
    inventory: ClusterInventory,
    *,
    create_link: bool = True,
    environment: Mapping[str, str] | None = None,
) -> None:
    """Configure a logout-safe cache within RAM-backed ``/dev/shm``.

    systemd-logind's default ``RemoveIPC=yes`` recursively removes ordinary
    user-owned files from ``/dev/shm`` when the user's final SSH session ends.
    The dedicated cache directory and completed entries are therefore owned by
    root while remaining writable through the caller's primary group.
    """

    mount_check = (
        'target="$(findmnt -n -T /dev/shm -o TARGET 2>/dev/null || true)"; '
        'fstype="$(findmnt -n -T /dev/shm -o FSTYPE 2>/dev/null || true)"; '
        'if [ "$target" = /dev/shm ] && [ -w /dev/shm ]; then '
        'case "$fstype" in tmpfs|ramfs) exit 0;; esac; fi; '
        'echo "ERROR: /dev/shm is not a writable tmpfs or ramfs "'
        '"(target=${target:-missing}, type=${fstype:-missing})" >&2; exit 3'
    )
    try:
        run_pdsh(
            inventory.hosts,
            mount_check,
            environment=environment,
            labels=True,
            timeout=60.0,
        )
    except ClusterAccessError:
        raise
    except ClusterError as exc:
        raise ClusterError(RAM_CACHE_SETUP_GUIDANCE) from exc

    root = root.resolve()
    link = root / "shm"
    cache = RAM_CACHE_ROOT
    quoted_root = shlex.quote(str(root))
    quoted_link = shlex.quote(str(link))
    quoted_cache = shlex.quote(str(cache))
    replace_error = shlex.quote(
        f"ERROR: refusing to replace existing {link}; move it aside"
    )
    missing_error = shlex.quote(
        f"ERROR: {link} is not a symlink to {cache}; "
        "rerun make prepare without --check-only"
    )
    valid_link = (
        f"[ -d {quoted_cache} ] && [ -L {quoted_link} ] && "
        f"[ \"$(readlink -f {quoted_link} 2>/dev/null || true)\" = {quoted_cache} ]"
    )
    if create_link:
        cache_setup = (
            'group="$(id -g)"; '
            'if [ "$(id -u)" -eq 0 ]; then '
            f"install -d -o root -g \"$group\" -m 0775 {quoted_cache} || exit 5; "
            "elif command -v sudo >/dev/null 2>&1 && "
            "sudo -n true >/dev/null 2>&1; then "
            f"sudo -n install -d -o root -g \"$group\" -m 0775 {quoted_cache} "
            "|| exit 5; else "
            "echo 'ERROR: passwordless sudo is required to protect the RAM cache' "
            ">&2; exit 5; fi; "
        )
        legacy_link = (
            f"[ -L {quoted_link} ] && "
            f"[ \"$(readlink -f {quoted_link} 2>/dev/null || true)\" = /dev/shm ]"
        )
        link_command = (
            f"{cache_setup}if {valid_link}; then exit 0; "
            f"elif {legacy_link}; then unlink {quoted_link} && "
            f"ln -s {quoted_cache} {quoted_link}; "
            f"elif [ -e {quoted_link} ] || [ -L {quoted_link} ]; then "
            f"echo {replace_error} >&2; "
            "exit 4; fi; "
            f"install -d -m 755 {quoted_root} && "
            f"ln -s {quoted_cache} {quoted_link}"
        )
    else:
        link_command = (
            f"if {valid_link}; then exit 0; fi; "
            f"echo {missing_error} >&2; exit 4"
        )
    try:
        run_pdsh(
            inventory.hosts,
            link_command,
            environment=environment,
            labels=True,
            timeout=60.0,
        )
    except ClusterAccessError:
        raise
    except ClusterError as exc:
        raise ClusterError(RAM_CACHE_PROTECTION_GUIDANCE) from exc


def seal_ram_cache_command() -> str:
    """Return a shell fragment that protects completed cache files at logout."""

    cache = shlex.quote(str(RAM_CACHE_ROOT))
    protect_direct = (
        f"chown -R root:\"$group\" -- {cache} && "
        f"find {cache} -type d -exec chmod 0775 {{}} +"
    )
    protect_sudo = (
        f"sudo -n chown -R root:\"$group\" -- {cache} && "
        f"sudo -n find {cache} -type d -exec chmod 0775 {{}} +"
    )
    return (
        'group="$(id -g)"; '
        'if [ "$(id -u)" -eq 0 ]; then '
        f"{protect_direct}; "
        "elif command -v sudo >/dev/null 2>&1 && "
        "sudo -n true >/dev/null 2>&1; then "
        f"{protect_sudo}; "
        "else echo 'ERROR: passwordless sudo is required to protect the RAM cache' "
        ">&2; exit 5; fi"
    )


def bootstrap_rsync(
    inventory: ClusterInventory,
    *,
    environment: Mapping[str, str] | None = None,
) -> None:
    """Install rsync with apt-get when absent, then require it on every host."""

    command = (
        "if command -v rsync >/dev/null 2>&1; then exit 0; fi; "
        "if ! command -v apt-get >/dev/null 2>&1; then "
        "echo 'ERROR: rsync is missing and apt-get is unavailable' >&2; exit 5; fi; "
        "if [ \"$(id -u)\" -eq 0 ]; then privilege=''; "
        "elif command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; "
        "then privilege='sudo -n'; else "
        "echo 'ERROR: rsync is missing and passwordless sudo is unavailable' >&2; "
        "exit 5; fi; "
        "$privilege env DEBIAN_FRONTEND=noninteractive apt-get update && "
        "$privilege env DEBIAN_FRONTEND=noninteractive apt-get install -y rsync && "
        "command -v rsync >/dev/null 2>&1"
    )
    try:
        run_pdsh(
            inventory.hosts,
            command,
            environment=environment,
            labels=True,
            timeout=900.0,
        )
    except ClusterAccessError:
        raise
    except ClusterError as exc:
        raise ClusterError(RSYNC_SETUP_GUIDANCE) from exc
    _require_program("rsync")


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
    retry_transport: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run one command on an explicit host set and require every host to succeed."""

    if not hosts:
        raise ClusterError("pdsh target list may not be empty")
    expression = ",".join(hosts)
    attempts = _PROBE_ATTEMPTS if retry_transport else 1
    for attempt in range(attempts):
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
        if completed.returncode == 0:
            return completed
        # pdsh propagates ssh's reserved transport status 255. Repeating an
        # idempotent setup/check command is safe; ordinary remote failures are
        # returned immediately and never disguised as a connection problem.
        if completed.returncode != 255:
            raise ClusterError(_command_failure("pdsh command failed", completed))
        if attempt + 1 < attempts:
            time.sleep(1.0)
    raise ClusterAccessError(SSH_SETUP_GUIDANCE)


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


def terminate_distributed_workers(
    *,
    host_expression: str,
    host_count: int,
    executable: Path,
    script: Path,
    output_dir: Path,
    environment: Mapping[str, str] | None = None,
) -> bool:
    """Best-effort teardown for one exact failed or interrupted distributed run.

    Killing the local ``pdsh`` process group is insufficient when an SSH channel
    disappears while TPU execution is inside XLA: the remote Python processes
    can survive as orphans. Match the immutable executable, trainer path, and
    run directory so cleanup cannot affect another submission or run.
    """

    if host_count <= 1:
        return True
    try:
        _require_program("pdsh")
        expression = _validate_host_expression(host_expression)
    except ClusterError:
        return False
    pattern = (
        "^"
        + re.escape(str(executable))
        + "[[:space:]]+"
        + re.escape(str(script))
        + ".*[[:space:]]--output-dir[[:space:]]+"
        + re.escape(str(output_dir))
        + "([[:space:]]|$)"
    )
    remote_command = (
        f"pkill -KILL -f -- {shlex.quote(pattern)} >/dev/null 2>&1 || true"
    )
    try:
        completed = subprocess.run(
            _pdsh_command(expression, host_count, remote_command, labels=False),
            env=pdsh_environment(environment),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


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


def _ssh_command(host: str, remote_command: str) -> list[str]:
    """Build one direct SSH command using the same options as pdsh and rsync."""

    command = ["ssh"]
    for option in _SSH_OPTIONS:
        command.extend(("-o", option))
    command.extend((host, remote_command))
    return command


def _rsync_to_hosts(
    root: Path,
    hosts: Sequence[str],
    exclusions: Sequence[str],
    *,
    environment: Mapping[str, str] | None,
) -> None:
    _require_program("rsync")
    sync_environment = pdsh_environment(environment)
    ssh_arguments = ["ssh"]
    for option in _SSH_OPTIONS:
        ssh_arguments.extend(("-o", option))
    ssh_command = shlex.join(ssh_arguments)

    def copy(host: str) -> tuple[str, subprocess.CompletedProcess[str]]:
        command = ["rsync", "-az", "--protect-args", "--quiet"]
        for exclusion in exclusions:
            command.extend(("--exclude", exclusion))
        command.extend(
            (
                "-e",
                ssh_command,
                "--",
                f"{root}/",
                f"{host}:{root}/",
            )
        )
        for attempt in range(_PROBE_ATTEMPTS):
            completed = subprocess.run(
                command,
                env=sync_environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=300.0,
            )
            if completed.returncode != 255 or attempt + 1 == _PROBE_ATTEMPTS:
                return host, completed
            time.sleep(1.0)
        raise AssertionError("unreachable")

    try:
        with ThreadPoolExecutor(max_workers=len(hosts)) as executor:
            results = tuple(executor.map(copy, hosts))
    except subprocess.TimeoutExpired as exc:
        raise ClusterError("workspace rsync timed out") from exc
    for host, completed in results:
        if completed.returncode != 0:
            if completed.returncode == 255:
                raise ClusterAccessError(SSH_SETUP_GUIDANCE)
            raise ClusterError(
                _command_failure(f"workspace rsync to {host} failed", completed)
            )


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
    "RAM_CACHE_PROTECTION_GUIDANCE",
    "RAM_CACHE_ROOT",
    "RAM_CACHE_SETUP_GUIDANCE",
    "RSYNC_SETUP_GUIDANCE",
    "SSH_SETUP_GUIDANCE",
    "bootstrap_rsync",
    "bootstrap_uv",
    "build_distributed_launch_command",
    "expand_host_expression",
    "infer_host_expression",
    "pdsh_environment",
    "prepare_ram_cache",
    "probe_cluster",
    "run_pdsh",
    "seal_ram_cache_command",
    "sync_workspace",
    "terminate_distributed_workers",
]
