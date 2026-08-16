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
# Peer-local state: not copied, and never removed by the mirror.
_PROTECTED_RSYNC_EXCLUDES = (
    ".git/",
    ".venv/",
    "shm",
    "runs/",
    "profiles/",
    "report.html",
)
# Derived caches: not worth copying, but a peer should not keep them once the
# sources that produced them are gone. rsync protects excluded paths from
# --delete by default, so without marking these "risk" a deleted package
# directory survives on the peer as a bytecode-only directory -- which Python
# still imports as an empty namespace package.
_DERIVED_CACHE_EXCLUDES = (
    ".mypy_cache/",
    ".pytest_cache/",
    ".ruff_cache/",
    "__pycache__/",
)
_COMMON_RSYNC_EXCLUDES = _PROTECTED_RSYNC_EXCLUDES + _DERIVED_CACHE_EXCLUDES


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
    # None when this machine orchestrates from outside the slice.
    local_host: str | None
    reported_hostnames: Mapping[str, str]
    # The TPU VM that writes run artifacts. Equals local_host in the ordinary
    # setup; in remote mode it is a peer, and its runs/ is pulled back after.
    # No default: an inventory that has not decided this is not usable.
    artifact_host: str


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
    remote_controller: bool = False,
    artifact_host: str = "",
) -> ClusterInventory:
    """Verify passwordless SSH and decide which host owns the artifacts.

    Ordinarily this machine is one of the TPU VMs and keeps the artifacts
    itself. Under ``remote_controller`` it is outside the slice entirely, so
    a peer is designated instead -- ``artifact_host`` when given, otherwise
    the first host in the expression.
    """

    if host_count <= 1 and not remote_controller:
        # In-slice mode needs peers; remote mode may legitimately drive one host.
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
    if remote_controller:
        if matches:
            raise ClusterError(
                "remote_controller is set, but this machine is one of the "
                f"configured TPU VM hosts ({matches[0]}). Either clear "
                "remote_controller or remove this host from tpu_vm_hosts."
            )
        chosen = _resolve_artifact_host(artifact_host, hosts, reported)
        return ClusterInventory(
            host_expression=host_expression,
            hosts=hosts,
            remote_hosts=hosts,
            local_host=None,
            reported_hostnames=reported,
            artifact_host=chosen,
        )
    if len(matches) != 1:
        raise ClusterError(
            "the configured TPU VM hosts must include this controller exactly "
            "once. Set remote_controller = true on this cluster to orchestrate "
            "from outside the slice."
        )
    local_target = matches[0]
    if artifact_host:
        chosen = _resolve_artifact_host(artifact_host, hosts, reported)
    else:
        chosen = local_target
    return ClusterInventory(
        host_expression=host_expression,
        hosts=hosts,
        remote_hosts=tuple(host for host in hosts if host != local_target),
        local_host=local_target,
        reported_hostnames=reported,
        artifact_host=chosen,
    )


def _resolve_artifact_host(
    requested: str, hosts: tuple[str, ...], reported: Mapping[str, str]
) -> str:
    """Pick the artifact-owning target, matching on target or reported name."""

    if not requested:
        return hosts[0]
    wanted = _short_hostname(requested)
    for target in hosts:
        if _short_hostname(target) == wanted:
            return target
        if _short_hostname(reported.get(target, "")) == wanted:
            return target
    raise ClusterError(
        f"artifact_host {requested!r} is not one of the configured TPU VM "
        f"hosts: {', '.join(hosts)}"
    )


def sync_workspace(
    root: Path,
    inventory: ClusterInventory,
    *,
    artifacts_path: Path | None = None,
    data_path: Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> None:
    """Mirror source/config bytes onto every peer VM.

    The copy is incremental but authoritative: files that no longer exist in the
    controller's checkout are removed from the peers, so peer state is a
    function of the current checkout rather than of every sync that came before
    it. Excluded paths — Git metadata, the virtual environment, ``shm``, the
    data cache, caches, profiles, and run artifacts — are never deleted.
    """

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
        risk=_DERIVED_CACHE_EXCLUDES,
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


def ship_uv_binary(
    hosts: Sequence[str],
    *,
    environment: Mapping[str, str] | None = None,
    timeout: float = 300.0,
) -> str | None:
    """Copy this machine's uv to every peer that lacks it.

    Peers without internet cannot fetch the installer, and the curl attempt
    fails slowly -- a four-minute connect timeout per host before anything says
    so. Shipping the binary we already run is faster, offline-safe, and pins
    peers to the controller's exact uv version instead of whatever the
    installer serves today.

    Returns the shipped version, or None when there was nothing to ship.
    """

    if not hosts:
        return None
    local = shutil.which("uv") or str(Path.home() / ".local/bin/uv")
    source = Path(local)
    if not source.is_file():
        return None
    try:
        version = subprocess.run(
            [str(source), "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
            timeout=30.0,
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        version = ""

    ssh_arguments = ["ssh"]
    for option in _SSH_OPTIONS:
        ssh_arguments.extend(("-o", option))
    failures: list[str] = []
    for host in hosts:
        try:
            prepare = subprocess.run(
                _ssh_command(host, "mkdir -p ~/.local/bin"),
                env=pdsh_environment(environment),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=60.0,
            )
            if prepare.returncode != 0:
                failures.append(host)
                continue
            copied = subprocess.run(
                [
                    "rsync",
                    "-a",
                    "--times",
                    "-e",
                    shlex.join(ssh_arguments),
                    str(source),
                    f"{host}:.local/bin/uv",
                ],
                env=pdsh_environment(environment),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=timeout,
            )
            if copied.returncode != 0:
                failures.append(host)
        except (OSError, subprocess.TimeoutExpired):
            failures.append(host)
    if failures:
        raise ClusterError(
            "could not copy uv to: " + ", ".join(failures)
        )
    return version or None


# uv keeps managed interpreters here, as a real versioned directory plus a
# "latest minor" symlink beside it.
_UV_PYTHON_ROOT = Path.home() / ".local/share/uv/python"


def ship_uv_python(
    hosts: Sequence[str],
    *,
    environment: Mapping[str, str] | None = None,
    timeout: float = 900.0,
) -> tuple[str, ...]:
    """Copy this machine's uv-managed interpreters to peers.

    An offline peer cannot fetch python-build-standalone, so ``uv sync`` fails
    on a host that has no matching interpreter. Shipping ours also guarantees
    peers run the same patch release the controller resolved its lockfile
    against.

    The real directory is transferred with its internal symlinks intact
    (``bin/python`` points at ``python3.12`` and must stay a link), and the
    sibling "latest minor" symlink is recreated remotely afterwards. Following
    symlinks wholesale instead would both triple the payload and break the
    install.
    """

    if not hosts or not _UV_PYTHON_ROOT.is_dir():
        return ()
    real = sorted(
        entry
        for entry in _UV_PYTHON_ROOT.iterdir()
        if entry.is_dir() and not entry.is_symlink()
    )
    if not real:
        return ()
    links = [
        entry
        for entry in _UV_PYTHON_ROOT.iterdir()
        if entry.is_symlink() and entry.resolve().parent == _UV_PYTHON_ROOT
    ]
    remote_root = ".local/share/uv/python"
    ssh_arguments = ["ssh"]
    for option in _SSH_OPTIONS:
        ssh_arguments.extend(("-o", option))

    failures: list[str] = []
    for host in hosts:
        try:
            if subprocess.run(
                _ssh_command(host, f"mkdir -p {shlex.quote(remote_root)}"),
                env=pdsh_environment(environment),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=60.0,
            ).returncode != 0:
                failures.append(host)
                continue
            copied = subprocess.run(
                [
                    "rsync",
                    "-a",
                    "--partial",
                    "-e",
                    shlex.join(ssh_arguments),
                    *[str(entry) for entry in real],
                    f"{host}:{remote_root}/",
                ],
                env=pdsh_environment(environment),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=timeout,
            )
            if copied.returncode != 0:
                failures.append(host)
                continue
            for link in links:
                target = link.resolve().name
                command = (
                    f"cd {shlex.quote(remote_root)} && "
                    f"ln -sfn {shlex.quote(target)} {shlex.quote(link.name)}"
                )
                subprocess.run(
                    _ssh_command(host, command),
                    env=pdsh_environment(environment),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                    timeout=60.0,
                )
        except (OSError, subprocess.TimeoutExpired):
            failures.append(host)
    if failures:
        raise ClusterError(
            "could not copy the uv-managed Python to: " + ", ".join(failures)
        )
    return tuple(entry.name for entry in real)


def ship_uv_cache(
    hosts: Sequence[str],
    cache_dir: Path,
    *,
    environment: Mapping[str, str] | None = None,
    timeout: float = 1800.0,
) -> int:
    """Copy the controller's uv cache to peers so a frozen sync needs no network.

    ``uv sync --frozen`` still downloads wheels; only a populated cache makes it
    offline. Transferring the controller's cache is what lets a peer with no
    route to PyPI resolve the same lockfile.

    Environment and interpreter directories are excluded: those are built
    against local absolute paths and are rebuilt cheaply on the peer, whereas
    copying them invites stale interpreter records pointing at paths that
    exist here and not there.

    Returns the number of hosts updated.
    """

    if not hosts or not cache_dir.is_dir():
        return 0
    ssh_arguments = ["ssh"]
    for option in _SSH_OPTIONS:
        ssh_arguments.extend(("-o", option))
    failures: list[str] = []
    updated = 0
    for host in hosts:
        try:
            copied = subprocess.run(
                [
                    "rsync",
                    "-a",
                    "--partial",
                    "--exclude",
                    "environments-v2/",
                    "--exclude",
                    "interpreter-v4/",
                    "-e",
                    shlex.join(ssh_arguments),
                    f"{cache_dir.resolve().as_posix()}/",
                    f"{host}:{cache_dir.resolve().as_posix()}/",
                ],
                env=pdsh_environment(environment),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=timeout,
            )
            if copied.returncode != 0:
                failures.append(host)
            else:
                updated += 1
        except (OSError, subprocess.TimeoutExpired):
            failures.append(host)
    if failures:
        raise ClusterError("could not copy the uv cache to: " + ", ".join(failures))
    return updated


def ship_dataset(
    hosts: Sequence[str],
    source: Path,
    *,
    environment: Mapping[str, str] | None = None,
    timeout: float = 7200.0,
) -> int:
    """Copy a prepared corpus from this machine's RAM cache to peers.

    An offline peer cannot download shards, so the controller's already
    verified copy is the source. The transfer and the seal happen inside one
    SSH session on purpose: systemd-logind defaults to RemoveIPC=yes and
    deletes user-owned /dev/shm entries once that user's last session on the
    host ends, so a copy that finishes and then loses its session evaporates.
    That failure is silent -- rsync reports success and the files are gone --
    and it is exactly what happened during the v5e-64 bring-up.

    Returns the number of hosts updated.
    """

    if not hosts or not source.is_dir():
        return 0
    relative = source.resolve().relative_to(RAM_CACHE_ROOT.resolve())
    remote_parent = (RAM_CACHE_ROOT / relative).parent
    ssh_arguments = ["ssh"]
    for option in _SSH_OPTIONS:
        ssh_arguments.extend(("-o", option))
    failures: list[str] = []
    updated = 0
    for host in hosts:
        try:
            made = subprocess.run(
                _ssh_command(
                    host,
                    'install -d -o "$(id -u)" -g "$(id -gn)" -m 0775 '
                    f"{shlex.quote(str(remote_parent))} 2>/dev/null || "
                    f"mkdir -p {shlex.quote(str(remote_parent))}",
                ),
                env=pdsh_environment(environment),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=120.0,
            )
            if made.returncode != 0:
                failures.append(host)
                continue
            copied = subprocess.run(
                [
                    "rsync",
                    "-a",
                    "--partial",
                    "-e",
                    shlex.join(ssh_arguments),
                    f"{source.resolve().as_posix()}",
                    f"{host}:{remote_parent.as_posix()}/",
                ],
                env=pdsh_environment(environment),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=timeout,
            )
            if copied.returncode != 0:
                failures.append(host)
                continue
            sealed = subprocess.run(
                _ssh_command(host, seal_ram_cache_command()),
                env=pdsh_environment(environment),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=600.0,
            )
            if sealed.returncode != 0:
                failures.append(host)
                continue
            updated += 1
        except (OSError, subprocess.TimeoutExpired, ValueError):
            failures.append(host)
    if failures:
        raise ClusterError("could not copy the dataset to: " + ", ".join(failures))
    return updated


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
    run directory so cleanup cannot affect another recipe or run.
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


def fetch_run_artifacts(
    host: str,
    remote_dir: Path,
    local_dir: Path,
    *,
    environment: Mapping[str, str] | None = None,
    timeout: float = 900.0,
) -> None:
    """Pull one run directory back from the artifact-owning TPU VM.

    Only used under a remote controller, where the run's outputs were written
    on a peer's disk. The transfer is additive: it never deletes locally, so a
    partial pull leaves whatever already arrived rather than truncating the
    run. The trailing slash copies the directory's contents, not the
    directory itself, so the local run id is preserved.
    """

    _require_program("rsync")
    local_dir.mkdir(parents=True, exist_ok=True)
    ssh_arguments = ["ssh"]
    for option in _SSH_OPTIONS:
        ssh_arguments.extend(("-o", option))
    command = [
        "rsync",
        "-a",
        "--partial",
        "-e",
        shlex.join(ssh_arguments),
        f"{host}:{remote_dir.as_posix()}/",
        f"{local_dir.as_posix()}/",
    ]
    try:
        completed = subprocess.run(
            command,
            env=pdsh_environment(environment),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise ClusterError(
            f"timed out pulling run artifacts from {host}"
        ) from exc
    if completed.returncode != 0:
        raise ClusterError(
            _command_failure(f"could not pull run artifacts from {host}", completed)
        )


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
    risk: Sequence[str] = (),
    environment: Mapping[str, str] | None,
) -> None:
    _require_program("rsync")
    sync_environment = pdsh_environment(environment)
    ssh_arguments = ["ssh"]
    for option in _SSH_OPTIONS:
        ssh_arguments.extend(("-o", option))
    ssh_command = shlex.join(ssh_arguments)

    def copy(host: str) -> tuple[str, subprocess.CompletedProcess[str]]:
        # --delete makes a peer a mirror of the controller instead of an
        # accumulation of every checkout it has ever received: a renamed or
        # deleted module would otherwise survive on the peers indefinitely and
        # stay importable there. --delete-after defers removal until the
        # transfer succeeds, so an interrupted sync never strips a peer.
        # --delete-excluded is deliberately NOT passed, which is what keeps
        # runs/, .venv/, .git/, profiles/, shm, and the data cache — all listed
        # in `exclusions` — from being touched.
        command = [
            "rsync",
            "-az",
            "--delete",
            "--delete-after",
            "--protect-args",
            "--quiet",
        ]
        # Risk rules must precede the exclusions: rsync takes the first
        # matching rule, and an --exclude would otherwise protect the pattern.
        for pattern in risk:
            command.extend(("--filter", f"R {pattern}"))
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
    "ship_dataset",
    "ship_uv_binary",
    "ship_uv_cache",
    "ship_uv_python",
    "fetch_run_artifacts",
    "infer_host_expression",
    "pdsh_environment",
    "prepare_ram_cache",
    "probe_cluster",
    "run_pdsh",
    "seal_ram_cache_command",
    "sync_workspace",
    "terminate_distributed_workers",
]
