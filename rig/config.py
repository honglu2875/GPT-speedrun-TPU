"""Local, gitignored preferences for the rig command-line interface."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import math
import tomllib
from typing import Any, Mapping


CONFIG_FILENAME = ".rig.toml"


class ConfigError(ValueError):
    """A local configuration file or requested setting is invalid."""


_TOPOLOGY_FIELDS = (
    "tpu_vm_count",
    "tpu_vm_hosts",
    "accelerator",
    "chips_per_host",
    "remote_controller",
    "artifact_host",
)
_DATA_FIELDS = ("dataset", "train_shards")
_RIG_FIELDS = (
    "data_path",
    "artifacts_path",
    "active_cluster",
    "default_profile",
    "checkpoint_policy",
    "color",
    "target_loss",
)
_LEGACY_RIG_FIELDS = (
    *_TOPOLOGY_FIELDS,
    *_DATA_FIELDS,
    "data_profile",
    "checkpoint_retention",
)


@dataclass(frozen=True, slots=True)
class ClusterProfile:
    """One named accelerator cluster: its hosts and what it is supposed to be."""

    name: str
    tpu_vm_count: int
    tpu_vm_hosts: str
    accelerator: str
    chips_per_host: int
    # True when this machine orchestrates the slice without being part of it:
    # it holds no accelerator, takes no JAX rank, and pulls artifacts back
    # from the designated host afterwards.
    remote_controller: bool = False
    # Which TPU VM writes the run artifacts. Empty means the first host in
    # the expression. Naming one explicitly matters on preemptible pods,
    # where you may want artifacts off a host you expect to lose.
    artifact_host: str = ""

    def overlay(self) -> dict[str, Any]:
        return {
            "tpu_vm_count": self.tpu_vm_count,
            "tpu_vm_hosts": self.tpu_vm_hosts,
            "accelerator": self.accelerator,
            "chips_per_host": self.chips_per_host,
            "remote_controller": self.remote_controller,
            "artifact_host": self.artifact_host,
        }


@dataclass(frozen=True, slots=True)
class LocalConfig:
    data_path: str = "shm"
    artifacts_path: str = "runs"
    tpu_vm_count: int = 1
    tpu_vm_hosts: str = ""
    # The accelerator this cluster is supposed to be. The doctor asserts it
    # rather than assuming one generation, so a v4 slice and a v5e slice are
    # each checked against their own contract instead of a hard-coded default.
    accelerator: str = "TPU v4"
    chips_per_host: int = 4
    # See ClusterProfile: orchestrate-without-participating, and which host
    # keeps the artifacts.
    remote_controller: bool = False
    artifact_host: str = ""
    # Explicit immutable corpus and optional train-shard prefix (0 = manifest default).
    dataset: str = "classic"
    train_shards: int = 0
    active_cluster: str = ""
    default_profile: str = "official"
    checkpoint_policy: str = "qualifying"
    color: str = "auto"
    target_loss: float = 3.28

    def validate(self) -> "LocalConfig":
        if isinstance(self.tpu_vm_count, bool) or not isinstance(
            self.tpu_vm_count, int
        ):
            raise ConfigError("tpu_vm_count must be a positive integer")
        if self.tpu_vm_count <= 0:
            raise ConfigError("tpu_vm_count must be a positive integer")
        if not isinstance(self.tpu_vm_hosts, str):
            raise ConfigError("tpu_vm_hosts must be a string")
        if any(character in self.tpu_vm_hosts for character in "\x00\r\n"):
            raise ConfigError("tpu_vm_hosts must be a single-line pdsh expression")
        if self.tpu_vm_count > 1 and not self.tpu_vm_hosts.strip():
            raise ConfigError(
                "tpu_vm_hosts is required when tpu_vm_count is greater than 1"
            )
        if any(character.isspace() for character in self.tpu_vm_hosts):
            raise ConfigError("tpu_vm_hosts may not contain whitespace")
        if not isinstance(self.remote_controller, bool):
            raise ConfigError("remote_controller must be true or false")
        if not isinstance(self.artifact_host, str):
            raise ConfigError("artifact_host must be a string")
        if any(character.isspace() for character in self.artifact_host):
            raise ConfigError("artifact_host may not contain whitespace")
        if self.remote_controller and not self.tpu_vm_hosts.strip():
            # A single remote host is legitimate and is the simplest remote
            # case: this machine holds no accelerator and the work runs on the
            # one host named here. What remote mode actually requires is a
            # host to reach, not a multi-host slice.
            raise ConfigError("remote_controller requires tpu_vm_hosts")
        if self.artifact_host and not self.tpu_vm_hosts.strip():
            raise ConfigError("artifact_host requires tpu_vm_hosts")
        if self.dataset not in {"classic", "2B", "4B", "8B", "hero"}:
            raise ConfigError("dataset must be classic, 2B, 4B, 8B, or hero")
        if isinstance(self.train_shards, bool) or not isinstance(
            self.train_shards, int
        ):
            raise ConfigError("train_shards must be a nonnegative integer")
        if self.train_shards < 0:
            raise ConfigError("train_shards must be a nonnegative integer")
        if not isinstance(self.accelerator, str) or not self.accelerator.strip():
            raise ConfigError("accelerator must be a non-empty string")
        if isinstance(self.chips_per_host, bool) or not isinstance(
            self.chips_per_host, int
        ):
            raise ConfigError("chips_per_host must be a positive integer")
        if self.chips_per_host <= 0:
            raise ConfigError("chips_per_host must be a positive integer")
        if self.default_profile not in {"smoke", "dev", "official"}:
            raise ConfigError("default_profile must be smoke, dev, or official")
        if self.checkpoint_policy not in {"always", "qualifying", "none"}:
            raise ConfigError("checkpoint_policy must be always, qualifying, or none")
        if self.color not in {"auto", "always", "never"}:
            raise ConfigError("color must be auto, always, or never")
        if not math.isfinite(self.target_loss) or self.target_loss < 0:
            raise ConfigError("target_loss must be finite and non-negative")
        if not self.data_path.strip() or not self.artifacts_path.strip():
            raise ConfigError("data and artifact paths may not be empty")
        return self


def repo_root() -> Path:
    """Return the checkout root, independent of the caller's current directory."""

    return Path(__file__).resolve().parent.parent


def config_path(root: Path | None = None) -> Path:
    return (root or repo_root()) / CONFIG_FILENAME


def _read_payload(path: Path) -> dict[str, Any]:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc


def load_clusters(root: Path | None = None) -> dict[str, ClusterProfile]:
    """Read every ``[cluster.<name>]`` table, if any are defined."""

    path = config_path(root)
    if not path.exists():
        return {}
    tables = _read_payload(path).get("cluster")
    if tables is None:
        return {}
    if not isinstance(tables, dict):
        raise ConfigError(f"{path}: [cluster] must contain named tables")
    profiles: dict[str, ClusterProfile] = {}
    base = asdict(LocalConfig())
    for name, table in tables.items():
        if not isinstance(table, dict):
            raise ConfigError(f"{path}: [cluster.{name}] must be a table")
        # Older files attached corpus selection to each cluster. Accept those
        # two keys during migration, but dataset choice is now authoritative in
        # the single top-level [data] table and never changes with --cluster.
        unknown = sorted(set(table) - set((*_TOPOLOGY_FIELDS, *_DATA_FIELDS)))
        if unknown:
            raise ConfigError(
                f"{path}: unknown setting(s) in [cluster.{name}]: {', '.join(unknown)}"
            )
        values = {field: table.get(field, base[field]) for field in _TOPOLOGY_FIELDS}
        profile = ClusterProfile(name=name, **values)
        # Validate through LocalConfig so a cluster cannot encode a state the
        # rest of the tool would reject later.
        LocalConfig(**{**base, **profile.overlay()}).validate()
        profiles[name] = profile
    return profiles


def load_config(root: Path | None = None, cluster: str | None = None) -> LocalConfig:
    """Load personal defaults, overlaid with the selected cluster profile.

    A file with no ``[cluster.*]`` tables keeps working unchanged: its flat
    ``[rig]`` host settings are the cluster.
    """

    path = config_path(root)
    if not path.exists():
        if cluster:
            raise ConfigError(
                f"no {CONFIG_FILENAME}; cannot select cluster {cluster!r}"
            )
        return LocalConfig().validate()
    payload = _read_payload(path)
    section = payload.get("rig")
    if not isinstance(section, dict):
        raise ConfigError(f"{path} must contain a [rig] table")
    defaults = asdict(LocalConfig())
    allowed_rig = set((*_RIG_FIELDS, *_LEGACY_RIG_FIELDS))
    unknown = sorted(set(section) - allowed_rig)
    if unknown:
        raise ConfigError(f"unknown setting(s) in {path}: {', '.join(unknown)}")
    for field in (*_RIG_FIELDS, *_TOPOLOGY_FIELDS, *_DATA_FIELDS):
        if field in section:
            defaults[field] = section[field]
    legacy_policy = section.get("checkpoint_retention")
    current_policy = section.get("checkpoint_policy")
    if (
        legacy_policy is not None
        and current_policy is not None
        and legacy_policy != current_policy
    ):
        raise ConfigError(
            f"{path}: checkpoint_policy conflicts with legacy checkpoint_retention"
        )
    if current_policy is None and legacy_policy is not None:
        defaults["checkpoint_policy"] = legacy_policy

    data_section = payload.get("data", {})
    if not isinstance(data_section, dict):
        raise ConfigError(f"{path}: [data] must be a table")
    unknown_data = sorted(set(data_section) - set(_DATA_FIELDS))
    if unknown_data:
        raise ConfigError(
            f"{path}: unknown setting(s) in [data]: {', '.join(unknown_data)}"
        )
    for field in _DATA_FIELDS:
        if field in data_section:
            defaults[field] = data_section[field]
    profiles = load_clusters(root)
    selected = cluster or str(defaults.get("active_cluster") or "")
    if selected:
        if selected not in profiles:
            known = ", ".join(sorted(profiles)) or "none defined"
            raise ConfigError(f"unknown cluster {selected!r}; known: {known}")
        defaults.update(profiles[selected].overlay())
        defaults["active_cluster"] = selected
    elif profiles:
        raise ConfigError(
            "clusters are defined but none is active; set active_cluster or pass "
            f"--cluster (known: {', '.join(sorted(profiles))})"
        )
    try:
        return LocalConfig(**defaults).validate()
    except TypeError as exc:
        raise ConfigError(f"invalid settings in {path}: {exc}") from exc


def save_config(
    config: LocalConfig,
    root: Path | None = None,
    clusters: Mapping[str, ClusterProfile] | None = None,
) -> Path:
    """Write personal defaults, preserving any named cluster profiles.

    Clusters default to whatever is already on disk, so saving settings from
    one cluster never silently drops another.
    """

    config.validate()
    path = config_path(root)
    if clusters is None:
        clusters = load_clusters(root)
    values = asdict(config)
    resolved_clusters = dict(clusters)
    if resolved_clusters and not str(values.get("active_cluster") or "").strip():
        matching = [
            name
            for name, profile in resolved_clusters.items()
            if profile.tpu_vm_hosts == config.tpu_vm_hosts
            and profile.tpu_vm_count == config.tpu_vm_count
        ]
        if len(matching) != 1:
            raise ConfigError(
                "refusing to save a config with clusters but no active_cluster; "
                f"pass --cluster (known: {', '.join(sorted(resolved_clusters))})"
            )
        values["active_cluster"] = matching[0]
    if resolved_clusters:
        active = str(values["active_cluster"])
        if active not in resolved_clusters:
            raise ConfigError(
                f"cannot save unknown active_cluster {active!r}; known: "
                + ", ".join(sorted(resolved_clusters))
            )
        resolved_clusters[active] = ClusterProfile(
            name=active,
            **{field: values[field] for field in _TOPOLOGY_FIELDS},
        )
    lines = [
        "# Personal defaults written by `rig prepare`.",
        "# Official constants live in data/manifests and docs/RULES.md.",
        "[rig]",
    ]
    rig_fields = list(_RIG_FIELDS)
    if not resolved_clusters:
        # A simple one-cluster file remains flat. Named-cluster files keep all
        # topology in [cluster.*] so the active overlay has one source of truth.
        rig_fields.extend(_TOPOLOGY_FIELDS)
    for key in rig_fields:
        value = values[key]
        lines.append(f"{key} = {_toml_scalar(key, value)}")
    lines.extend(
        (
            "",
            "# Immutable corpus selection used by every non-smoke execution type.",
            "[data]",
        )
    )
    for key in _DATA_FIELDS:
        lines.append(f"{key} = {_toml_scalar(key, values[key])}")
    for name in sorted(resolved_clusters):
        profile = resolved_clusters[name]
        lines.append("")
        lines.append(f"[cluster.{name}]")
        for key, value in profile.overlay().items():
            lines.append(f"{key} = {_toml_scalar(key, value)}")
    lines.append("")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)
    return path


def _toml_scalar(key: str, value: Any) -> str:
    """Encode one setting. bool is checked first: it subclasses int, and TOML
    spells it lowercase, so falling through would emit Python's ``True``."""

    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return _toml_string(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, int):
        return str(value)
    raise ConfigError(f"cannot encode setting {key}")


def resolve_path(value: str | Path, root: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (root or repo_root()) / path
    return path.resolve(strict=False)


def with_overrides(config: LocalConfig, values: Mapping[str, Any]) -> LocalConfig:
    payload = asdict(config)
    payload.update({key: value for key, value in values.items() if value is not None})
    return LocalConfig(**payload).validate()


def _toml_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


__all__ = [
    "CONFIG_FILENAME",
    "ClusterProfile",
    "ConfigError",
    "LocalConfig",
    "load_clusters",
    "config_path",
    "load_config",
    "repo_root",
    "resolve_path",
    "save_config",
    "with_overrides",
]
