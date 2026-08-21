"""Strict reading of a recipe's sibling ``config.yaml``.

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


class StrictSafeLoader(yaml.SafeLoader):
    """Safe YAML loader which rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: StrictSafeLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise ValueError("config.yaml mapping keys must be scalar values") from exc
        if duplicate:
            raise ValueError(f"config.yaml contains duplicate key {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


StrictSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def resolve_sibling_config_path(requested: Path | None, expected: Path) -> Path:
    """Resolve the one accepted config path: ``expected``, and nothing else.

    ``--config`` exists so a run records which file it read, not so a run can
    read a different one. Accepting an arbitrary path would let a recipe be
    measured against a configuration that is not the one beside it.
    """

    if expected.is_symlink() or not expected.is_file():
        raise ValueError(
            "required sibling experiment config must be a regular, "
            f"non-symlink file: {expected}"
        )
    try:
        resolved_expected = expected.expanduser().resolve(strict=True)
    except OSError as exc:
        raise ValueError(
            f"required sibling experiment config is unavailable: {expected}"
        ) from exc
    candidate = expected if requested is None else requested
    if candidate.is_symlink():
        raise ValueError("--config may not be a symlink")
    try:
        resolved = candidate.expanduser().resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"experiment config is unavailable: {candidate}") from exc
    if resolved != resolved_expected:
        raise ValueError(
            f"--config must name the config.yaml beside train.py: {expected}"
        )
    return resolved_expected


def read_config_document(path: Path) -> tuple[Mapping[str, Any], str]:
    """Return the single strict YAML mapping in ``path`` and the file's sha256."""

    raw = path.read_bytes()
    if len(raw) > MAX_CONFIG_BYTES:
        raise ValueError(
            f"config.yaml exceeds the {MAX_CONFIG_BYTES:,}-byte safety limit"
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("config.yaml must be UTF-8") from exc
    try:
        forbidden_tokens = (
            yaml.tokens.AliasToken,
            yaml.tokens.AnchorToken,
            yaml.tokens.DirectiveToken,
            yaml.tokens.TagToken,
        )
        for token in yaml.scan(text, Loader=StrictSafeLoader):
            if isinstance(token, forbidden_tokens):
                kind = type(token).__name__.removesuffix("Token").lower()
                raise ValueError(f"config.yaml may not contain YAML {kind}s")
        documents = list(yaml.load_all(text, Loader=StrictSafeLoader))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid config.yaml YAML: {exc}") from exc
    if len(documents) != 1:
        raise ValueError("config.yaml must contain exactly one YAML document")
    document = documents[0]
    if not isinstance(document, Mapping):
        raise ValueError("config.yaml document must be a mapping")
    if any(not isinstance(key, str) for key in document):
        raise ValueError("config.yaml document keys must be strings")
    return document, hashlib.sha256(raw).hexdigest()
