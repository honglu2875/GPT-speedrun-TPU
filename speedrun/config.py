"""Local, gitignored preferences for the speedrun command-line interface."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import math
import tomllib
from typing import Any, Mapping


CONFIG_FILENAME = ".speedrun.toml"


class ConfigError(ValueError):
    """A local configuration file or requested setting is invalid."""


@dataclass(frozen=True)
class LocalConfig:
    data_path: str = "shm"
    artifacts_path: str = "runs"
    data_profile: str = "official"
    default_profile: str = "official"
    default_track: str = "open"
    checkpoint_retention: str = "qualifying"
    color: str = "auto"
    target_loss: float = 3.28

    def validate(self) -> "LocalConfig":
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
        if not self.data_path.strip() or not self.artifacts_path.strip():
            raise ConfigError("data and artifact paths may not be empty")
        return self


def repo_root() -> Path:
    """Return the checkout root, independent of the caller's current directory."""

    return Path(__file__).resolve().parent.parent


def config_path(root: Path | None = None) -> Path:
    return (root or repo_root()) / CONFIG_FILENAME


def load_config(root: Path | None = None) -> LocalConfig:
    path = config_path(root)
    if not path.exists():
        return LocalConfig().validate()
    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    section = payload.get("speedrun")
    if not isinstance(section, dict):
        raise ConfigError(f"{path} must contain a [speedrun] table")
    defaults = asdict(LocalConfig())
    unknown = sorted(set(section) - set(defaults))
    if unknown:
        raise ConfigError(f"unknown setting(s) in {path}: {', '.join(unknown)}")
    defaults.update(section)
    try:
        return LocalConfig(**defaults).validate()
    except TypeError as exc:
        raise ConfigError(f"invalid settings in {path}: {exc}") from exc


def save_config(config: LocalConfig, root: Path | None = None) -> Path:
    config.validate()
    path = config_path(root)
    values = asdict(config)
    lines = [
        "# Personal defaults written by `speedrun prepare`.",
        "# Official constants live in data/manifests and docs/RULES.md.",
        "[speedrun]",
    ]
    for key, value in values.items():
        if isinstance(value, str):
            encoded = _toml_string(value)
        elif isinstance(value, float):
            encoded = repr(value)
        else:  # defensive for future scalar settings
            raise ConfigError(f"cannot encode setting {key}")
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
    "ConfigError",
    "LocalConfig",
    "config_path",
    "load_config",
    "repo_root",
    "resolve_path",
    "save_config",
    "with_overrides",
]

