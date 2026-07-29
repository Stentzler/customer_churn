"""Typed configuration for deterministic customer-churn data operations.

The YAML file is an external boundary: it is easy for a person to edit, but Python
cannot assume that its keys or values are valid. This module converts that untrusted
mapping into immutable objects before data logic is allowed to use it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import cast

import yaml

NUMERIC_RANGE_NAMES = frozenset(
    {
        "age",
        "tenure_months",
        "monthly_spend",
        "support_tickets_90d",
        "late_payments_12m",
        "usage_hours_monthly",
    }
)
CATEGORY_NAMES = frozenset({"plan_type", "region"})
DATA_CONTRACT_KEYS = frozenset(
    {
        "schema_version",
        "strict",
        "ordered",
        "minimum_batch_size",
        "maximum_failure_examples",
        "numeric_ranges",
        "allowed_categories",
        "target_values",
    }
)
DATA_GENERATION_KEYS = frozenset(
    {
        "reference_rows",
        "fixed_test_rows",
        "batch_rows",
        "seed_offsets",
    }
)
DATASET_SCENARIOS = frozenset(
    {
        "reference",
        "fixed_test",
        "normal",
        "drifted",
        "invalid",
    }
)

type NumericValue = int | float
type StringMapping = dict[str, object]


class DataContractConfigurationError(ValueError):
    """Raised when versioned data-contract configuration is missing or unsafe."""


@dataclass(frozen=True)
class NumericRange:
    """Inclusive minimum and maximum accepted for one numerical feature."""

    minimum: NumericValue
    maximum: NumericValue


@dataclass(frozen=True)
class DataContractConfig:
    """Validated, immutable policy used to build the dataframe schema."""

    schema_version: str
    strict: bool
    ordered: bool
    minimum_batch_size: int
    maximum_failure_examples: int
    numeric_ranges: Mapping[str, NumericRange]
    allowed_categories: Mapping[str, tuple[str, ...]]
    target_values: tuple[int, ...]


@dataclass(frozen=True)
class DataGenerationConfig:
    """Validated policy controlling synthetic dataset sizes and random streams."""

    random_seed: int
    reference_rows: int
    fixed_test_rows: int
    batch_rows: int
    seed_offsets: Mapping[str, int]

    def seed_for(self, scenario: str) -> int:
        """Return the deterministic random seed assigned to one dataset scenario."""

        try:
            return self.random_seed + self.seed_offsets[scenario]
        except KeyError as error:
            supported = ", ".join(sorted(self.seed_offsets))
            message = (
                f"Unknown dataset scenario '{scenario}'; expected one of: {supported}"
            )
            raise ValueError(message) from error


def load_data_contract(params_path: Path) -> DataContractConfig:
    """Load and validate the data-contract section of a YAML parameters file.

    Args:
        params_path: Path to the repository's versioned YAML configuration.

    Returns:
        An immutable configuration safe for deterministic validation code.

    Raises:
        DataContractConfigurationError: If the file cannot be read or its contract
            section is missing, malformed, or internally inconsistent.
    """

    root = _load_yaml_mapping(params_path)
    contract = _require_mapping(root.get("data_contract"), "data_contract")
    _require_exact_keys(contract, DATA_CONTRACT_KEYS, "data_contract")

    numeric_ranges = _parse_numeric_ranges(contract.get("numeric_ranges"))
    allowed_categories = _parse_allowed_categories(contract.get("allowed_categories"))

    return DataContractConfig(
        schema_version=_require_non_empty_string(
            contract.get("schema_version"), "data_contract.schema_version"
        ),
        strict=_require_boolean(contract.get("strict"), "data_contract.strict"),
        ordered=_require_boolean(contract.get("ordered"), "data_contract.ordered"),
        minimum_batch_size=_require_positive_integer(
            contract.get("minimum_batch_size"),
            "data_contract.minimum_batch_size",
        ),
        maximum_failure_examples=_require_positive_integer(
            contract.get("maximum_failure_examples"),
            "data_contract.maximum_failure_examples",
        ),
        # A frozen dataclass does not make nested dictionaries immutable. Read-only
        # proxies prevent policy from being changed accidentally after loading.
        numeric_ranges=MappingProxyType(numeric_ranges),
        allowed_categories=MappingProxyType(allowed_categories),
        target_values=_parse_target_values(contract.get("target_values")),
    )


def load_data_generation(params_path: Path) -> DataGenerationConfig:
    """Load deterministic synthetic-data settings from versioned parameters.

    Each scenario receives a distinct offset from the project seed. Independent
    random streams make generation reproducible while preventing two scenarios
    from accidentally producing identical rows.
    """

    root = _load_yaml_mapping(params_path)
    project = _require_mapping(root.get("project"), "project")
    generation = _require_mapping(root.get("data_generation"), "data_generation")
    _require_exact_keys(generation, DATA_GENERATION_KEYS, "data_generation")

    seed_offsets = _parse_seed_offsets(generation.get("seed_offsets"))
    return DataGenerationConfig(
        random_seed=_require_non_negative_integer(
            project.get("random_seed"),
            "project.random_seed",
        ),
        reference_rows=_require_positive_integer(
            generation.get("reference_rows"),
            "data_generation.reference_rows",
        ),
        fixed_test_rows=_require_positive_integer(
            generation.get("fixed_test_rows"),
            "data_generation.fixed_test_rows",
        ),
        batch_rows=_require_positive_integer(
            generation.get("batch_rows"),
            "data_generation.batch_rows",
        ),
        seed_offsets=MappingProxyType(seed_offsets),
    )


def _parse_seed_offsets(value: object) -> dict[str, int]:
    offsets = _require_mapping(value, "data_generation.seed_offsets")
    _require_exact_keys(offsets, DATASET_SCENARIOS, "data_generation.seed_offsets")

    parsed_offsets = {
        scenario: _require_non_negative_integer(
            offsets.get(scenario),
            f"data_generation.seed_offsets.{scenario}",
        )
        for scenario in sorted(DATASET_SCENARIOS)
    }
    if len(set(parsed_offsets.values())) != len(parsed_offsets):
        message = "data_generation.seed_offsets values must be unique"
        raise DataContractConfigurationError(message)
    return parsed_offsets


def _load_yaml_mapping(params_path: Path) -> StringMapping:
    try:
        yaml_content = params_path.read_text(encoding="utf-8")
        parsed_yaml = yaml.safe_load(yaml_content)
    except OSError as error:
        message = f"Cannot read parameters file '{params_path}': {error}"
        raise DataContractConfigurationError(message) from error
    except yaml.YAMLError as error:
        message = f"Parameters file '{params_path}' contains invalid YAML: {error}"
        raise DataContractConfigurationError(message) from error

    return _require_mapping(parsed_yaml, "configuration root")


def _parse_numeric_ranges(value: object) -> dict[str, NumericRange]:
    ranges = _require_mapping(value, "data_contract.numeric_ranges")
    _require_exact_keys(
        ranges,
        NUMERIC_RANGE_NAMES,
        "data_contract.numeric_ranges",
    )

    parsed_ranges: dict[str, NumericRange] = {}
    for feature_name in sorted(NUMERIC_RANGE_NAMES):
        location = f"data_contract.numeric_ranges.{feature_name}"
        range_mapping = _require_mapping(ranges.get(feature_name), location)
        _require_exact_keys(range_mapping, frozenset({"minimum", "maximum"}), location)

        minimum = _require_finite_number(
            range_mapping.get("minimum"), f"{location}.minimum"
        )
        maximum = _require_finite_number(
            range_mapping.get("maximum"), f"{location}.maximum"
        )
        if minimum >= maximum:
            message = f"{location}.minimum must be less than {location}.maximum"
            raise DataContractConfigurationError(message)

        parsed_ranges[feature_name] = NumericRange(
            minimum=minimum,
            maximum=maximum,
        )

    return parsed_ranges


def _parse_allowed_categories(value: object) -> dict[str, tuple[str, ...]]:
    categories = _require_mapping(value, "data_contract.allowed_categories")
    _require_exact_keys(
        categories,
        CATEGORY_NAMES,
        "data_contract.allowed_categories",
    )

    return {
        category_name: _require_unique_string_list(
            categories.get(category_name),
            f"data_contract.allowed_categories.{category_name}",
        )
        for category_name in sorted(CATEGORY_NAMES)
    }


def _parse_target_values(value: object) -> tuple[int, ...]:
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        message = "data_contract.target_values must be a list of integers"
        raise DataContractConfigurationError(message)

    target_values = tuple(cast(list[int], value))
    if target_values != (0, 1):
        message = "data_contract.target_values must be exactly [0, 1]"
        raise DataContractConfigurationError(message)
    return target_values


def _require_mapping(value: object, location: str) -> StringMapping:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        message = f"{location} must be a mapping with string keys"
        raise DataContractConfigurationError(message)
    return cast(StringMapping, value)


def _require_exact_keys(
    mapping: StringMapping,
    expected_keys: frozenset[str],
    location: str,
) -> None:
    actual_keys = set(mapping)
    missing_keys = sorted(expected_keys - actual_keys)
    unexpected_keys = sorted(actual_keys - expected_keys)
    if not missing_keys and not unexpected_keys:
        return

    details: list[str] = []
    if missing_keys:
        details.append(f"missing keys: {', '.join(missing_keys)}")
    if unexpected_keys:
        details.append(f"unexpected keys: {', '.join(unexpected_keys)}")

    message = f"{location} has invalid fields ({'; '.join(details)})"
    raise DataContractConfigurationError(message)


def _require_non_empty_string(value: object, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        message = f"{location} must be a non-empty string"
        raise DataContractConfigurationError(message)
    return value


def _require_boolean(value: object, location: str) -> bool:
    if type(value) is not bool:
        message = f"{location} must be a boolean"
        raise DataContractConfigurationError(message)
    return cast(bool, value)


def _require_positive_integer(value: object, location: str) -> int:
    if type(value) is not int or cast(int, value) < 1:
        message = f"{location} must be a positive integer"
        raise DataContractConfigurationError(message)
    return cast(int, value)


def _require_non_negative_integer(value: object, location: str) -> int:
    if type(value) is not int or cast(int, value) < 0:
        message = f"{location} must be a non-negative integer"
        raise DataContractConfigurationError(message)
    return cast(int, value)


def _require_finite_number(value: object, location: str) -> NumericValue:
    if type(value) not in {int, float} or not math.isfinite(cast(float, value)):
        message = f"{location} must be a finite number"
        raise DataContractConfigurationError(message)
    return cast(NumericValue, value)


def _require_unique_string_list(value: object, location: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        message = f"{location} must be a non-empty list"
        raise DataContractConfigurationError(message)
    if any(not isinstance(item, str) or not item.strip() for item in value):
        message = f"{location} must contain only non-empty strings"
        raise DataContractConfigurationError(message)

    categories = tuple(cast(list[str], value))
    if len(categories) != len(set(categories)):
        message = f"{location} must not contain duplicate values"
        raise DataContractConfigurationError(message)
    return categories
