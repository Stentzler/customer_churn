from pathlib import Path

import pandas as pd
import pytest
from src.data.generate import (
    DatasetScenario,
    generate_synthetic_dataset,
    generate_valid_customer_dataframe,
)
from src.data.schema import CUSTOMER_CHURN_COLUMNS, build_customer_churn_schema
from src.data.settings import (
    DataContractConfig,
    DataGenerationConfig,
    load_data_generation,
)
from src.data.validate import validate_dataframe

PARAMS_PATH = Path(__file__).resolve().parents[3] / "params.yaml"


def test_generated_dataframe_satisfies_the_data_contract(
    data_contract_config: DataContractConfig,
) -> None:
    dataframe = generate_valid_customer_dataframe(
        row_count=data_contract_config.minimum_batch_size,
        seed=42,
        contract=data_contract_config,
    )

    validated_dataframe = build_customer_churn_schema(data_contract_config).validate(
        dataframe
    )

    assert len(validated_dataframe) == data_contract_config.minimum_batch_size
    assert tuple(validated_dataframe.columns) == CUSTOMER_CHURN_COLUMNS
    assert validated_dataframe["customer_id"].is_unique
    assert set(validated_dataframe["churned"]) <= {0, 1}


def test_generation_is_reproducible_for_the_same_inputs(
    data_contract_config: DataContractConfig,
) -> None:
    first_dataframe = generate_valid_customer_dataframe(
        row_count=60,
        seed=42,
        contract=data_contract_config,
    )
    second_dataframe = generate_valid_customer_dataframe(
        row_count=60,
        seed=42,
        contract=data_contract_config,
    )

    pd.testing.assert_frame_equal(first_dataframe, second_dataframe)


def test_different_seeds_produce_different_data(
    data_contract_config: DataContractConfig,
) -> None:
    first_dataframe = generate_valid_customer_dataframe(
        row_count=60,
        seed=42,
        contract=data_contract_config,
    )
    second_dataframe = generate_valid_customer_dataframe(
        row_count=60,
        seed=43,
        contract=data_contract_config,
    )

    assert not first_dataframe.equals(second_dataframe)


@pytest.mark.parametrize(
    "scenario",
    [
        DatasetScenario.REFERENCE,
        DatasetScenario.FIXED_TEST,
        DatasetScenario.NORMAL,
        DatasetScenario.DRIFTED,
    ],
)
def test_valid_scenarios_satisfy_the_contract(
    scenario: DatasetScenario,
    data_contract_config: DataContractConfig,
) -> None:
    generation = load_data_generation(PARAMS_PATH)

    dataframe = generate_synthetic_dataset(
        scenario,
        generation,
        data_contract_config,
    )

    assert validate_dataframe(dataframe, data_contract_config).is_valid is True


def test_scenarios_use_their_configured_row_counts(
    data_contract_config: DataContractConfig,
) -> None:
    generation = load_data_generation(PARAMS_PATH)

    assert (
        len(
            generate_synthetic_dataset(
                DatasetScenario.REFERENCE,
                generation,
                data_contract_config,
            )
        )
        == generation.reference_rows
    )
    assert (
        len(
            generate_synthetic_dataset(
                DatasetScenario.FIXED_TEST,
                generation,
                data_contract_config,
            )
        )
        == generation.fixed_test_rows
    )
    assert (
        len(
            generate_synthetic_dataset(
                DatasetScenario.NORMAL,
                generation,
                data_contract_config,
            )
        )
        == generation.batch_rows
    )


def test_drifted_scenario_shifts_selected_features(
    data_contract_config: DataContractConfig,
) -> None:
    generation = load_data_generation(PARAMS_PATH)
    normal = generate_synthetic_dataset(
        DatasetScenario.NORMAL,
        generation,
        data_contract_config,
    )
    drifted = generate_synthetic_dataset(
        DatasetScenario.DRIFTED,
        generation,
        data_contract_config,
    )

    assert drifted["monthly_spend"].mean() > normal["monthly_spend"].mean()
    assert drifted["support_tickets_90d"].mean() > normal["support_tickets_90d"].mean()
    assert drifted["usage_hours_monthly"].mean() < normal["usage_hours_monthly"].mean()


def test_generated_target_has_an_understandable_learnable_signal(
    data_contract_config: DataContractConfig,
) -> None:
    dataframe = generate_valid_customer_dataframe(
        row_count=2000,
        seed=42,
        contract=data_contract_config,
    )
    low_risk = dataframe[
        (dataframe["support_tickets_90d"] <= 2) & (dataframe["late_payments_12m"] <= 1)
    ]
    high_risk = dataframe[
        (dataframe["support_tickets_90d"] >= 10) | (dataframe["late_payments_12m"] >= 7)
    ]

    assert len(low_risk) >= 50
    assert len(high_risk) >= 50
    assert high_risk["churned"].mean() > low_risk["churned"].mean() + 0.20


def test_invalid_scenario_has_expected_contract_failures(
    data_contract_config: DataContractConfig,
) -> None:
    generation = load_data_generation(PARAMS_PATH)
    dataframe = generate_synthetic_dataset(
        DatasetScenario.INVALID,
        generation,
        data_contract_config,
    )

    result = validate_dataframe(dataframe, data_contract_config)
    issue_codes = {issue.code for issue in result.issues}

    assert result.is_valid is False
    assert {"duplicate_customer_id", "invalid_category", "out_of_range"} <= issue_codes


@pytest.mark.parametrize("scenario", list(DatasetScenario))
def test_each_scenario_is_reproducible(
    scenario: DatasetScenario,
    data_contract_config: DataContractConfig,
) -> None:
    generation = load_data_generation(PARAMS_PATH)

    first_dataframe = generate_synthetic_dataset(
        scenario,
        generation,
        data_contract_config,
    )
    second_dataframe = generate_synthetic_dataset(
        scenario,
        generation,
        data_contract_config,
    )

    pd.testing.assert_frame_equal(first_dataframe, second_dataframe)


def test_scenario_rejects_a_batch_smaller_than_the_contract(
    data_contract_config: DataContractConfig,
) -> None:
    generation = DataGenerationConfig(
        random_seed=42,
        reference_rows=1000,
        fixed_test_rows=300,
        batch_rows=data_contract_config.minimum_batch_size - 1,
        seed_offsets={
            "reference": 0,
            "fixed_test": 1,
            "normal": 2,
            "drifted": 3,
            "invalid": 4,
        },
    )

    with pytest.raises(ValueError, match="below the contract minimum"):
        generate_synthetic_dataset(
            DatasetScenario.NORMAL,
            generation,
            data_contract_config,
        )


@pytest.mark.parametrize(
    ("row_count", "seed", "expected_message"),
    [
        (0, 42, "row_count must be a positive integer"),
        (50, -1, "seed must be a non-negative integer"),
    ],
)
def test_invalid_generation_inputs_are_rejected(
    data_contract_config: DataContractConfig,
    row_count: int,
    seed: int,
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        generate_valid_customer_dataframe(
            row_count=row_count,
            seed=seed,
            contract=data_contract_config,
        )
