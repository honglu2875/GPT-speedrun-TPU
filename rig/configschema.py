"""Strict, annotation-driven decoding for trusted recipe configuration schemas.

The YAML reader in :mod:`rig.configfile` establishes a safe mapping boundary.
This module turns that mapping into recipe-local dataclasses while enforcing
only structural and single-field rules: required and unknown keys, primitive
types, literals, optional values, typed dictionaries, numeric bounds, string
patterns, and collection lengths.

It deliberately does not implement aliases, coercion from strings, arbitrary
unions, field-name remapping, or cross-field callbacks. Scientific contracts
such as parameter-count identities, backend/dtype compatibility, and ladder
scaling remain explicit recipe code.
"""

from __future__ import annotations

from collections.abc import Mapping, Sized
from dataclasses import MISSING, dataclass, fields, is_dataclass
import math
import re
import types
from typing import (
    Annotated,
    Any,
    Literal,
    Self,
    TypeAlias,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)


@dataclass(frozen=True, slots=True)
class Bounds:
    """Inclusive or exclusive numeric bounds attached through ``Annotated``."""

    gt: int | float | None = None
    ge: int | float | None = None
    lt: int | float | None = None
    le: int | float | None = None

    def __post_init__(self) -> None:
        if self.gt is not None and self.ge is not None:
            raise ValueError("Bounds may specify only one of gt and ge")
        if self.lt is not None and self.le is not None:
            raise ValueError("Bounds may specify only one of lt and le")


@dataclass(frozen=True, slots=True)
class Matches:
    """Require a string to fully match one regular expression."""

    pattern: str

    def __post_init__(self) -> None:
        try:
            re.compile(self.pattern)
        except re.error as exc:
            raise ValueError(f"invalid Matches pattern: {exc}") from exc


@dataclass(frozen=True, slots=True)
class Length:
    """Inclusive lower and upper bounds for a sized value."""

    ge: int | None = None
    le: int | None = None

    def __post_init__(self) -> None:
        if self.ge is not None and self.ge < 0:
            raise ValueError("Length.ge must be nonnegative")
        if self.le is not None and self.le < 0:
            raise ValueError("Length.le must be nonnegative")
        if self.ge is not None and self.le is not None and self.ge > self.le:
            raise ValueError("Length.ge must not exceed Length.le")


PositiveInt: TypeAlias = Annotated[int, Bounds(gt=0)]
NonnegativeInt: TypeAlias = Annotated[int, Bounds(ge=0)]
PositiveFloat: TypeAlias = Annotated[float, Bounds(gt=0.0)]
NonnegativeFloat: TypeAlias = Annotated[float, Bounds(ge=0.0)]


class SchemaError(ValueError):
    """A configuration value does not satisfy its declared schema."""


class SchemaDefinitionError(TypeError):
    """A trusted dataclass uses a schema annotation this decoder does not support."""


class ConfigSchema:
    """Opt-in classmethod for recipe-local dataclass schemas."""

    @classmethod
    def from_mapping(cls, value: Any, *, label: str = "config.yaml") -> Self:
        return decode_dataclass(value, cls, label=label)


SchemaT = TypeVar("SchemaT")


def decode_dataclass(
    value: Any,
    schema: type[SchemaT],
    *,
    label: str = "config.yaml",
) -> SchemaT:
    """Decode one mapping into ``schema`` with no undeclared coercions."""

    if not isinstance(schema, type) or not is_dataclass(schema):
        raise SchemaDefinitionError("schema must be a dataclass type")
    return _decode(value, schema, label)


def _decode(value: Any, annotation: Any, label: str) -> Any:
    origin = get_origin(annotation)

    if origin is Annotated:
        base, *constraints = get_args(annotation)
        decoded = _decode(value, base, label)
        for constraint in constraints:
            _validate_constraint(decoded, constraint, label)
        return decoded

    if origin is Literal:
        choices = get_args(annotation)
        if not any(
            type(value) is type(choice) and value == choice for choice in choices
        ):
            expected = ", ".join(repr(choice) for choice in choices)
            raise SchemaError(f"{label} must be one of {expected}; got {value!r}")
        return value

    if origin in (types.UnionType, Union):
        members = get_args(annotation)
        non_none = tuple(member for member in members if member is not type(None))
        if len(non_none) != 1 or len(non_none) == len(members):
            raise SchemaDefinitionError(
                f"{label} uses an unsupported union; only T | None is accepted"
            )
        if value is None:
            return None
        return _decode(value, non_none[0], label)

    if origin is dict:
        key_type, value_type = get_args(annotation)
        if not isinstance(value, Mapping):
            raise SchemaError(f"{label} must be a mapping; got {type(value).__name__}")
        result = {}
        for key, item in value.items():
            decoded_key = _decode(key, key_type, f"{label} key {key!r}")
            item_label = (
                f"{label}.{key}" if isinstance(key, str) else f"{label}[{key!r}]"
            )
            result[decoded_key] = _decode(item, value_type, item_label)
        return result

    if isinstance(annotation, type) and is_dataclass(annotation):
        return _decode_dataclass_value(value, annotation, label)

    if annotation is bool:
        if not isinstance(value, bool):
            raise SchemaError(f"{label} must be a boolean; got {value!r}")
        return value
    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise SchemaError(f"{label} must be an integer; got {value!r}")
        return value
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SchemaError(f"{label} must be a finite number; got {value!r}")
        try:
            result = float(value)
        except OverflowError as exc:
            raise SchemaError(
                f"{label} must be a finite number; got {value!r}"
            ) from exc
        if not math.isfinite(result):
            raise SchemaError(f"{label} must be a finite number; got {value!r}")
        return result
    if annotation is str:
        if not isinstance(value, str):
            raise SchemaError(f"{label} must be a string; got {value!r}")
        return value

    raise SchemaDefinitionError(f"{label} uses unsupported annotation {annotation!r}")


def _decode_dataclass_value(value: Any, schema: type[Any], label: str) -> Any:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{label} must be a mapping; got {type(value).__name__}")
    schema_fields = tuple(field for field in fields(schema) if field.init)
    accepted = {field.name for field in schema_fields}
    unknown = sorted((key for key in value if key not in accepted), key=repr)
    if unknown:
        rendered = ", ".join(repr(key) for key in unknown)
        raise SchemaError(f"{label} contains unknown key(s): {rendered}")

    hints = get_type_hints(schema, include_extras=True)
    decoded = {}
    for field in schema_fields:
        if field.name not in value:
            if field.default is not MISSING or field.default_factory is not MISSING:
                continue
            raise SchemaError(f"{label} is missing required key {field.name!r}")
        if field.name not in hints:
            raise SchemaDefinitionError(
                f"{schema.__name__}.{field.name} has no resolved type annotation"
            )
        decoded[field.name] = _decode(
            value[field.name], hints[field.name], f"{label}.{field.name}"
        )
    return schema(**decoded)


def _validate_constraint(value: Any, constraint: Any, label: str) -> None:
    if isinstance(constraint, Bounds):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SchemaDefinitionError(f"{label} applies Bounds to a non-number")
        checks = (
            (
                constraint.gt,
                value > constraint.gt if constraint.gt is not None else True,
                ">",
            ),
            (
                constraint.ge,
                value >= constraint.ge if constraint.ge is not None else True,
                ">=",
            ),
            (
                constraint.lt,
                value < constraint.lt if constraint.lt is not None else True,
                "<",
            ),
            (
                constraint.le,
                value <= constraint.le if constraint.le is not None else True,
                "<=",
            ),
        )
        for bound, accepted, operator in checks:
            if bound is not None and not accepted:
                raise SchemaError(f"{label} must be {operator} {bound}; got {value!r}")
        return
    if isinstance(constraint, Matches):
        if not isinstance(value, str):
            raise SchemaDefinitionError(f"{label} applies Matches to a non-string")
        if re.fullmatch(constraint.pattern, value) is None:
            raise SchemaError(
                f"{label} must match /{constraint.pattern}/; got {value!r}"
            )
        return
    if isinstance(constraint, Length):
        if not isinstance(value, Sized):
            raise SchemaDefinitionError(f"{label} applies Length to an unsized value")
        size = len(value)
        if constraint.ge is not None and size < constraint.ge:
            raise SchemaError(
                f"{label} must contain at least {constraint.ge} item(s); got {size}"
            )
        if constraint.le is not None and size > constraint.le:
            raise SchemaError(
                f"{label} must contain at most {constraint.le} item(s); got {size}"
            )
        return
    raise SchemaDefinitionError(
        f"{label} uses unsupported Annotated metadata {constraint!r}"
    )


__all__ = (
    "Bounds",
    "ConfigSchema",
    "Length",
    "Matches",
    "NonnegativeFloat",
    "NonnegativeInt",
    "PositiveFloat",
    "PositiveInt",
    "SchemaDefinitionError",
    "SchemaError",
    "decode_dataclass",
)
