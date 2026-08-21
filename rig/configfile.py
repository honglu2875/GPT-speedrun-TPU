"""Strict reading and selection of a recipe's explicit YAML configurations.

This is the *reader*, not the schema. It resolves the path, bounds the file,
rejects the YAML features that would make a config non-obvious, and loads
exactly one string-keyed document. What the keys mean, which are required, and
how they become a training configuration is each recipe's own business and
stays in its entry program.

The strictness is deliberate. A config that decides what a run measures should
read exactly as it is written, so anchors, aliases, tags, directives, duplicate
keys, and multiple documents are all refused rather than resolved. The file's
sha256 travels with the parsed mapping because the run record identifies a
configuration by content, not by path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import hashlib

import yaml


MAX_CONFIG_BYTES = 256 * 1024
PROFILE_CONFIG_FILENAMES = {
    "smoke": "smoke.yaml",
    "dev": "dev.yaml",
    "official": "config.yaml",
}


def profile_config_filename(profile: str) -> str:
    """Return the standalone recipe config selected by an execution profile.

    Unknown names retain the historical ``config.yaml`` convention for trusted
    third-party recipes using :mod:`rig.harness` directly. The public CLI only
    admits the three standard profiles above.
    """

    return PROFILE_CONFIG_FILENAMES.get(profile, "config.yaml")


class StrictSafeLoader(yaml.SafeLoader):
    """Safe YAML loader which rejects duplicate mapping keys."""

    config_label = "experiment config"


def _construct_unique_mapping(
    loader: StrictSafeLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    label = loader.config_label
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise ValueError(f"{label} mapping keys must be scalar values") from exc
        if duplicate:
            raise ValueError(f"{label} contains duplicate key {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


StrictSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def read_config_document(path: Path) -> tuple[Mapping[str, Any], str]:
    """Return the single strict YAML mapping in ``path`` and the file's sha256."""

    label = path.name

    class FileLoader(StrictSafeLoader):
        config_label = label

    if path.is_symlink() or not path.is_file():
        raise ValueError(
            f"required experiment config must be a regular, non-symlink file: {path}"
        )
    raw = path.read_bytes()
    if len(raw) > MAX_CONFIG_BYTES:
        raise ValueError(f"{label} exceeds the {MAX_CONFIG_BYTES:,}-byte safety limit")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} must be UTF-8") from exc
    try:
        forbidden_tokens = (
            yaml.tokens.AliasToken,
            yaml.tokens.AnchorToken,
            yaml.tokens.DirectiveToken,
            yaml.tokens.TagToken,
        )
        for token in yaml.scan(text, Loader=FileLoader):
            if isinstance(token, forbidden_tokens):
                kind = type(token).__name__.removesuffix("Token").lower()
                raise ValueError(f"{label} may not contain YAML {kind}s")
        documents = list(yaml.load_all(text, Loader=FileLoader))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid {label} YAML: {exc}") from exc
    if len(documents) != 1:
        raise ValueError(f"{label} must contain exactly one YAML document")
    document = documents[0]
    if not isinstance(document, Mapping):
        raise ValueError(f"{label} document must be a mapping")
    if any(not isinstance(key, str) for key in document):
        raise ValueError(f"{label} document keys must be strings")
    return document, hashlib.sha256(raw).hexdigest()


__all__ = (
    "MAX_CONFIG_BYTES",
    "PROFILE_CONFIG_FILENAMES",
    "profile_config_filename",
    "read_config_document",
)
