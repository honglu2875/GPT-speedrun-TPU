"""Tests for the deliberately small recipe configuration decoder."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Annotated, Literal
import unittest

from rig.configschema import (
    Bounds,
    ConfigSchema,
    Length,
    Matches,
    NonnegativeInt,
    PositiveFloat,
    PositiveInt,
    SchemaDefinitionError,
    SchemaError,
    decode_dataclass,
)


Name = Annotated[str, Matches(r"[a-z][a-z0-9_-]*")]
Probability = Annotated[float, Bounds(ge=0.0, le=1.0)]


@dataclass(frozen=True, slots=True)
class Child(ConfigSchema):
    count: PositiveInt
    ratio: Probability
    enabled: bool
    mode: Literal["dense", "tiled"]


@dataclass(frozen=True, slots=True)
class Document(ConfigSchema):
    schema_version: Literal[4]
    children: Annotated[dict[Name, Child], Length(ge=1)]
    optional_count: NonnegativeInt | None = None
    labels: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RequiredOptional(ConfigSchema):
    value: int | None


@dataclass(frozen=True)
class UnsupportedUnion(ConfigSchema):
    value: int | str


@dataclass(frozen=True)
class UnsupportedCollection(ConfigSchema):
    values: list[int]


@dataclass(frozen=True)
class UnsupportedMetadata(ConfigSchema):
    value: Annotated[int, "custom callback"]


def valid_document() -> dict[str, object]:
    return {
        "schema_version": 4,
        "children": {
            "main": {
                "count": 2,
                "ratio": 1,
                "enabled": True,
                "mode": "dense",
            }
        },
    }


class ConfigSchemaDecodingTests(unittest.TestCase):
    def test_nested_dataclasses_decode_to_ordinary_python_values(self) -> None:
        result = Document.from_mapping(valid_document())
        self.assertEqual(result.schema_version, 4)
        self.assertEqual(result.children["main"].count, 2)
        self.assertEqual(result.children["main"].ratio, 1.0)
        self.assertIs(type(result.children["main"].ratio), float)
        self.assertEqual(result.labels, {})
        self.assertIsNone(result.optional_count)
        self.assertFalse(hasattr(result, "__dict__"))
        self.assertFalse(hasattr(result.children["main"], "__dict__"))

    def test_unknown_and_missing_keys_are_rejected_at_the_exact_path(self) -> None:
        unknown = valid_document()
        unknown["children"]["main"]["surprise"] = 1  # type: ignore[index]
        with self.assertRaisesRegex(
            SchemaError, r"config\.yaml\.children\.main contains unknown key.*surprise"
        ):
            Document.from_mapping(unknown)

        missing = valid_document()
        del missing["children"]["main"]["count"]  # type: ignore[index]
        with self.assertRaisesRegex(
            SchemaError, r"config\.yaml\.children\.main is missing required key 'count'"
        ):
            Document.from_mapping(missing)

    def test_primitives_are_strict_and_bool_is_never_numeric(self) -> None:
        cases = (
            ("count", True, "must be an integer"),
            ("count", 2.0, "must be an integer"),
            ("ratio", True, "must be a finite number"),
            ("ratio", "0.5", "must be a finite number"),
            ("enabled", 1, "must be a boolean"),
        )
        for field_name, value, message in cases:
            payload = valid_document()
            payload["children"]["main"][field_name] = value  # type: ignore[index]
            with (
                self.subTest(field=field_name, value=value),
                self.assertRaisesRegex(SchemaError, message),
            ):
                Document.from_mapping(payload)

    def test_every_float_is_finite_before_bounds_are_checked(self) -> None:
        for value in (math.nan, math.inf, -math.inf):
            payload = valid_document()
            payload["children"]["main"]["ratio"] = value  # type: ignore[index]
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(SchemaError, "must be a finite number"),
            ):
                Document.from_mapping(payload)

    def test_bounds_literal_pattern_and_length_constraints_are_enforced(self) -> None:
        cases = []
        for field_name, value, message in (
            ("count", 0, "must be > 0"),
            ("ratio", -0.1, "must be >= 0.0"),
            ("ratio", 1.1, "must be <= 1.0"),
            ("mode", "flash", "must be one of"),
        ):
            payload = valid_document()
            payload["children"]["main"][field_name] = value  # type: ignore[index]
            cases.append((payload, message))
        cases.append(
            (
                {
                    **valid_document(),
                    "children": {
                        "Bad Name": valid_document()["children"]["main"]  # type: ignore[index]
                    },
                },
                "must match",
            )
        )
        cases.append(({**valid_document(), "children": {}}, "at least 1 item"))

        for payload, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(SchemaError, message),
            ):
                Document.from_mapping(payload)

    def test_optional_does_not_mean_optional_key_without_a_default(self) -> None:
        with self.assertRaisesRegex(SchemaError, "missing required key 'value'"):
            RequiredOptional.from_mapping({})
        self.assertIsNone(RequiredOptional.from_mapping({"value": None}).value)
        self.assertEqual(RequiredOptional.from_mapping({"value": 3}).value, 3)

    def test_root_requires_a_mapping_and_a_dataclass_schema(self) -> None:
        with self.assertRaisesRegex(SchemaError, "must be a mapping"):
            Document.from_mapping([])
        with self.assertRaisesRegex(SchemaDefinitionError, "dataclass type"):
            decode_dataclass({}, dict)


class ConfigSchemaDefinitionTests(unittest.TestCase):
    def test_only_optional_unions_and_declared_collection_types_are_supported(
        self,
    ) -> None:
        with self.assertRaisesRegex(SchemaDefinitionError, "unsupported union"):
            UnsupportedUnion.from_mapping({"value": 1})
        with self.assertRaisesRegex(SchemaDefinitionError, "unsupported annotation"):
            UnsupportedCollection.from_mapping({"values": [1]})
        with self.assertRaisesRegex(
            SchemaDefinitionError, "unsupported Annotated metadata"
        ):
            UnsupportedMetadata.from_mapping({"value": 1})

    def test_constraint_definitions_reject_ambiguous_or_impossible_ranges(self) -> None:
        with self.assertRaisesRegex(ValueError, "only one of gt and ge"):
            Bounds(gt=0, ge=0)
        with self.assertRaisesRegex(ValueError, "only one of lt and le"):
            Bounds(lt=1, le=1)
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            Length(ge=2, le=1)
        with self.assertRaisesRegex(ValueError, "unterminated"):
            Matches("[")

    def test_positive_float_alias_retains_plain_float_semantics(self) -> None:
        @dataclass(frozen=True)
        class Value:
            amount: PositiveFloat

        result = decode_dataclass({"amount": 2}, Value)
        self.assertEqual(result.amount, 2.0)
        self.assertIs(type(result.amount), float)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
