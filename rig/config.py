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


_CLUSTER_FIELDS = ("tpu_vm_count", "tpu_vm_hosts", "accelerator", "chips_per_host")


@dataclass(frozen=True)
class ClusterProfile:
    """One named accelerator cluster: its hosts and what it is supposed to be."""

    name: str
    tpu_vm_count: int
    tpu_vm_hosts: str
    accelerator: str
    chips_per_host: int

    def overlay(self) -> dict[str, Any]:
        return {
            "tpu_vm_count": self.tpu_vm_count,
            "tpu_vm_hosts": self.tpu_vm_hosts,
            "accelerator": self.accelerator,
            "chips_per_host": self.chips_per_host,
        }


@dataclass(frozen=True)
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
    active_cluster: str = ""
    data_profile: str = "official"
    default_profile: str = "official"
    default_track: str = "open"
    checkpoint_retention: str = "qualifying"
    color: str = "auto"
    target_loss: float = 3.28
    # Personal immutable corpus capacity used by non-smoke runs.
    training_tokens: int = 624_984_064

    def validate(self) -> "LocalConfig":
        if isinstance(self.tpu_vm_count, bool) or not isinstance(self.tpu_vm_count, int):
            raise ConfigError("tpu_vm_count must be a positive integer")
        if self.tpu_vm_count <= 0:
            raise ConfigError("tpu_vm_count must be a positive integer")
        if not isinstance(self.tpu_vm_hosts, str):
            raise ConfigError("tpu_vm_hosts must be a string")
        if any(character in self.tpu_vm_hosts for character in "\x00\r\n"):
            raise ConfigError("tpu_vm_hosts must be a single-line pdsh expression")
        if self.tpu_vm_count > 1 and not self.tpu_vm_hosts.strip():
            raise ConfigError("tpu_vm_hosts is required when tpu_vm_count is greater than 1")
        if any(character.isspace() for character in self.tpu_vm_hosts):
            raise ConfigError("tpu_vm_hosts may not contain whitespace")
        if not isinstance(self.accelerator, str) or not self.accelerator.strip():
            raise ConfigError("accelerator must be a non-empty string")
        if isinstance(self.chips_per_host, bool) or not isinstance(
            self.chips_per_host, int
        ):
            raise ConfigError("chips_per_host must be a positive integer")
        if self.chips_per_host <= 0:
            raise ConfigError("chips_per_host must be a positive integer")
        if self.data_profile not in {"smoke", "dev", "official"}:
            raise ConfigError("data_profile must be smoke, dev, or official")
        if self.default_profile not in {"smoke", "dev", "official"}:
            raise ConfigError("default_profile must be smoke, dev, or official")
        if self.default_track not in {"open", "sample_efficiency"}:
            raise ConfigError("default_track must be open or sample_efficiency")
        if self.checkpoint_retention not in {
            "all",
            "qualifying",
            "none-after-validation",
        }:
            raise ConfigError("invalid checkpoint_retention")
        if self.color not in {"auto", "always", "never"}:
            raise ConfigError("color must be auto, always, or never")
        if not math.isfinite(self.target_loss) or self.target_loss < 0:
            raise ConfigError("target_loss must be finite and non-negative")
        if isinstance(self.training_tokens, bool) or not isinstance(
            self.training_tokens, int
        ):
            raise ConfigError("training_tokens must be a positive integer")
        if self.training_tokens <= 0:
            raise ConfigError("training_tokens must be a positive integer")
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
        unknown = sorted(set(table) - set(_CLUSTER_FIELDS))
        if unknown:
            raise ConfigError(
                f"{path}: unknown setting(s) in [cluster.{name}]: {', '.join(unknown)}"
            )
        values = {field: table.get(field, base[field]) for field in _CLUSTER_FIELDS}
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
            raise ConfigError(f"no {CONFIG_FILENAME}; cannot select cluster {cluster!r}")
        return LocalConfig().validate()
    payload = _read_payload(path)
    section = payload.get("rig")
    if not isinstance(section, dict):
        raise ConfigError(f"{path} must contain a [rig] table")
    defaults = asdict(LocalConfig())
    unknown = sorted(set(section) - set(defaults))
    if unknown:
        raise ConfigError(f"unknown setting(s) in {path}: {', '.join(unknown)}")
    defaults.update(section)
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
    if clusters and not str(values.get("active_cluster") or "").strip():
        matching = [
            name
            for name, profile in clusters.items()
            if profile.tpu_vm_hosts == config.tpu_vm_hosts
            and profile.tpu_vm_count == config.tpu_vm_count
        ]
        if len(matching) != 1:
            raise ConfigError(
                "refusing to save a config with clusters but no active_cluster; "
                f"pass --cluster (known: {', '.join(sorted(clusters))})"
            )
        values["active_cluster"] = matching[0]
    lines = [
        "# Personal defaults written by `rig prepare`.",
        "# Official constants live in data/manifests and docs/RULES.md.",
        "# training_tokens selects the immutable corpus used by non-smoke runs.",
        "[rig]",
    ]
    for key, value in values.items():
        if isinstance(value, str):
            encoded = _toml_string(value)
        elif isinstance(value, float):
            encoded = repr(value)
        elif isinstance(value, int) and not isinstance(value, bool):
            encoded = str(value)
        else:  # defensive for future scalar settings
            raise ConfigError(f"cannot encode setting {key}")
        lines.append(f"{key} = {encoded}")
    for name in sorted(clusters):
        profile = clusters[name]
        lines.append("")
        lines.append(f"[cluster.{name}]")
        for key, value in profile.overlay().items():
            encoded = _toml_string(value) if isinstance(value, str) else str(value)
            lines.append(f"{key} = {encoded}")
    lines.append("")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)
    return path


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
