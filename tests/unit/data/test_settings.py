from pathlib import Path

import pytest
import yaml
from src.data.settings import (
    DataContractConfigurationError,
    DataGenerationConfig,
    NumericRange,
    load_data_contract,
    load_data_generation,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


def test_load_data_contract_from_versioned_params() -> None:
    contract = load_data_contract(PARAMS_PATH)

    assert contract.schema_version == "1.0"
    assert contract.strict is True
    assert contract.ordered is True
    assert contract.minimum_batch_size == 50
    assert contract.maximum_failure_examples == 10
    assert contract.numeric_ranges["age"] == NumericRange(18, 100)
    assert contract.allowed_categories["plan_type"] == (
        "basic",
        "standard",
        "premium",
    )
    assert contract.allowed_categories["region"] == (
        "north",
        "south",
        "east",
        "west",
    )
    assert contract.target_values == (0, 1)


def test_load_data_generation_from_versioned_params() -> None:
    generation = load_data_generation(PARAMS_PATH)

    assert generation == DataGenerationConfig(
        random_seed=42,
        reference_rows=1000,
        fixed_test_rows=300,
        batch_rows=200,
        seed_offsets={
            "reference": 0,
            "fixed_test": 1,
            "normal": 2,
            "drifted": 3,
            "invalid": 4,
        },
    )
    assert generation.seed_for("reference") == 42
    assert generation.seed_for("drifted") == 45


def test_duplicate_generation_seed_offsets_are_rejected(tmp_path: Path) -> None:
    params = _load_versioned_params()
    generation = _get_mapping(params, "data_generation")
    offsets = _get_mapping(generation, "seed_offsets")
    offsets["normal"] = offsets["reference"]
    params_path = _write_params(tmp_path, params)

    with pytest.raises(
        DataContractConfigurationError,
        match="seed_offsets values must be unique",
    ):
        load_data_generation(params_path)


def test_negative_project_seed_is_rejected(tmp_path: Path) -> None:
    params = _load_versioned_params()
    project = _get_mapping(params, "project")
    project["random_seed"] = -1
    params_path = _write_params(tmp_path, params)

    with pytest.raises(
        DataContractConfigurationError,
        match=r"project\.random_seed must be a non-negative integer",
    ):
        load_data_generation(params_path)


def test_unknown_generation_scenario_is_rejected() -> None:
    generation = load_data_generation(PARAMS_PATH)

    with pytest.raises(ValueError, match="Unknown dataset scenario 'future'"):
        generation.seed_for("future")


def test_missing_data_contract_is_rejected(tmp_path: Path) -> None:
    params_path = _write_params(tmp_path, {"project": {"random_seed": 42}})

    with pytest.raises(
        DataContractConfigurationError,
        match="data_contract must be a mapping",
    ):
        load_data_contract(params_path)


def test_malformed_yaml_is_rejected(tmp_path: Path) -> None:
    params_path = tmp_path / "params.yaml"
    params_path.write_text("data_contract: [invalid", encoding="utf-8")

    with pytest.raises(
        DataContractConfigurationError,
        match="contains invalid YAML",
    ):
        load_data_contract(params_path)


def test_inverted_numeric_range_is_rejected(tmp_path: Path) -> None:
    params = _load_versioned_params()
    contract = _get_mapping(params, "data_contract")
    ranges = _get_mapping(contract, "numeric_ranges")
    age_range = _get_mapping(ranges, "age")
    age_range["minimum"] = 101

    params_path = _write_params(tmp_path, params)

    with pytest.raises(
        DataContractConfigurationError,
        match=r"numeric_ranges\.age\.minimum must be less",
    ):
        load_data_contract(params_path)


def test_empty_category_list_is_rejected(tmp_path: Path) -> None:
    params = _load_versioned_params()
    contract = _get_mapping(params, "data_contract")
    categories = _get_mapping(contract, "allowed_categories")
    categories["region"] = []

    params_path = _write_params(tmp_path, params)

    with pytest.raises(
        DataContractConfigurationError,
        match=r"allowed_categories\.region must be a non-empty list",
    ):
        load_data_contract(params_path)


def test_invalid_minimum_batch_size_is_rejected(tmp_path: Path) -> None:
    params = _load_versioned_params()
    contract = _get_mapping(params, "data_contract")
    contract["minimum_batch_size"] = 0

    params_path = _write_params(tmp_path, params)

    with pytest.raises(
        DataContractConfigurationError,
        match="minimum_batch_size must be a positive integer",
    ):
        load_data_contract(params_path)


def test_unexpected_contract_setting_is_rejected(tmp_path: Path) -> None:
    params = _load_versioned_params()
    contract = _get_mapping(params, "data_contract")
    contract["unknown_policy"] = True

    params_path = _write_params(tmp_path, params)

    with pytest.raises(
        DataContractConfigurationError,
        match="unexpected keys: unknown_policy",
    ):
        load_data_contract(params_path)


def _load_versioned_params() -> dict[str, object]:
    parsed_yaml = yaml.safe_load(PARAMS_PATH.read_text(encoding="utf-8"))
    assert isinstance(parsed_yaml, dict)
    return parsed_yaml


def _get_mapping(mapping: dict[str, object], key: str) -> dict[str, object]:
    value = mapping[key]
    assert isinstance(value, dict)
    return value


def _write_params(tmp_path: Path, params: object) -> Path:
    params_path = tmp_path / "params.yaml"
    params_path.write_text(
        yaml.safe_dump(params, sort_keys=False),
        encoding="utf-8",
    )
    return params_path
