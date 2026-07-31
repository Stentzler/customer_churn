"""Deterministic synthetic customer-churn data generation.

The generator uses an isolated pseudo-random number generator. Calling it with the
same row count, seed, and data contract therefore produces the same dataframe
without changing random state used by other parts of the application.
"""

from __future__ import annotations

import argparse
import logging
import math
import random
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path

import pandas as pd
from src.data.settings import (
    DataContractConfig,
    DataContractConfigurationError,
    DataGenerationConfig,
    NumericRange,
    load_data_contract,
    load_data_generation,
)

LOGGER = logging.getLogger(__name__)
LOG_FORMAT = "%(levelname)s %(name)s %(message)s"


class DatasetScenario(StrEnum):
    """Allowlisted synthetic datasets required by the DataOps demonstration."""

    REFERENCE = "reference"
    FIXED_TEST = "fixed_test"
    NORMAL = "normal"
    DRIFTED = "drifted"
    INVALID = "invalid"


class DataGenerationOperationalError(RuntimeError):
    """Raised when generated data cannot be persisted to its destination."""


def generate_synthetic_dataset(
    scenario: DatasetScenario,
    generation: DataGenerationConfig,
    contract: DataContractConfig,
) -> pd.DataFrame:
    """Generate one configured dataset scenario.

    Scenario selection is deterministic and allowlisted. Reference and fixed-test
    sizes are configured independently; incoming scenarios share the configured
    batch size. Drift and invalid mutations are applied only after valid base data
    has been generated, making their purpose explicit and testable.

    Raises:
        ValueError: If a configured dataset would violate the minimum batch size.
    """

    row_count = _row_count_for(scenario, generation)
    if row_count < contract.minimum_batch_size:
        message = (
            f"{scenario.value} row count ({row_count}) is below the contract minimum "
            f"({contract.minimum_batch_size})"
        )
        raise ValueError(message)

    dataframe = generate_valid_customer_dataframe(
        row_count=row_count,
        seed=generation.seed_for(scenario.value),
        contract=contract,
    )
    if scenario is DatasetScenario.DRIFTED:
        # Keep labels unchanged so this remains the project's feature-only drift
        # demonstration. Target drift is measured separately by the pipeline.
        return apply_feature_drift(dataframe, contract)
    if scenario is DatasetScenario.INVALID:
        return _apply_contract_violations(dataframe, contract)
    return dataframe


def generated_dataset_path(
    scenario: DatasetScenario,
    data_root: Path,
) -> Path:
    """Return the stable output path assigned to a synthetic scenario."""

    if scenario is DatasetScenario.REFERENCE:
        return data_root / "reference" / "reference.csv"
    if scenario is DatasetScenario.FIXED_TEST:
        return data_root / "test" / "fixed_test.csv"
    return data_root / "incoming" / f"{scenario.value}.csv"


def write_synthetic_dataset(
    scenario: DatasetScenario,
    generation: DataGenerationConfig,
    contract: DataContractConfig,
    data_root: Path,
) -> Path:
    """Generate one scenario and atomically persist it as deterministic CSV."""

    dataframe = generate_synthetic_dataset(scenario, generation, contract)
    output_path = generated_dataset_path(scenario, data_root)
    temporary_path = output_path.with_suffix(".csv.tmp")

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(
            temporary_path,
            index=False,
            lineterminator="\n",
            float_format="%.2f",
        )
        temporary_path.replace(output_path)
    except OSError as error:
        temporary_path.unlink(missing_ok=True)
        message = f"Cannot write generated dataset to '{output_path}': {error}"
        raise DataGenerationOperationalError(message) from error

    return output_path


def write_all_synthetic_datasets(
    generation: DataGenerationConfig,
    contract: DataContractConfig,
    data_root: Path,
) -> tuple[Path, ...]:
    """Persist every scenario in stable enum order."""

    return tuple(
        write_synthetic_dataset(scenario, generation, contract, data_root)
        for scenario in DatasetScenario
    )


def main(arguments: Sequence[str] | None = None) -> int:
    """Run synthetic data generation from the command line."""

    parsed_arguments = _build_argument_parser().parse_args(arguments)
    try:
        generation = load_data_generation(parsed_arguments.params)
        contract = load_data_contract(parsed_arguments.params)
        scenarios = (
            tuple(DatasetScenario)
            if parsed_arguments.scenario == "all"
            else (DatasetScenario(parsed_arguments.scenario),)
        )
        for scenario in scenarios:
            output_path = write_synthetic_dataset(
                scenario,
                generation,
                contract,
                parsed_arguments.data_root,
            )
            LOGGER.info(
                "dataset_generated scenario=%s rows=%d path=%s",
                scenario.value,
                _row_count_for(scenario, generation),
                output_path,
            )
    except (
        DataContractConfigurationError,
        DataGenerationOperationalError,
        ValueError,
    ) as error:
        LOGGER.error("generation_operational_error reason=%s", error)
        return 2
    return 0


def configure_logging() -> None:
    """Configure concise default logging for direct CLI execution."""

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic customer-churn datasets.",
    )
    parser.add_argument(
        "--scenario",
        choices=[scenario.value for scenario in DatasetScenario] + ["all"],
        default="all",
        help="Dataset scenario to generate (default: all).",
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=Path("params.yaml"),
        help="Versioned YAML parameters file (default: params.yaml).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root output directory (default: data).",
    )
    return parser


def generate_valid_customer_dataframe(
    row_count: int,
    seed: int,
    contract: DataContractConfig,
) -> pd.DataFrame:
    """Generate valid customer records with a reproducible churn relationship.

    Feature values are sampled inside the configured contract. Churn is not purely
    random: spend, support activity, and late payments increase risk, while usage
    and tenure reduce it. The effects are strong enough for simple baseline models
    to learn while probabilistic label sampling keeps the task realistic.

    Args:
        row_count: Number of customer rows to create.
        seed: Non-negative seed for this dataset's independent random stream.
        contract: Validated ranges and categories shared with Pandera validation.

    Returns:
        A dataframe with the documented feature order and explicit pandas dtypes.

    Raises:
        ValueError: If ``row_count`` is not positive or ``seed`` is negative.
    """

    _validate_generation_inputs(row_count, seed)
    random_generator = random.Random(seed)
    records = [
        _generate_customer_record(
            row_number=row_number,
            seed=seed,
            random_generator=random_generator,
            contract=contract,
        )
        for row_number in range(row_count)
    ]
    return _build_typed_dataframe(records)


def _row_count_for(
    scenario: DatasetScenario,
    generation: DataGenerationConfig,
) -> int:
    if scenario is DatasetScenario.REFERENCE:
        return generation.reference_rows
    if scenario is DatasetScenario.FIXED_TEST:
        return generation.fixed_test_rows
    return generation.batch_rows


def apply_feature_drift(
    dataframe: pd.DataFrame,
    contract: DataContractConfig,
) -> pd.DataFrame:
    """Shift three feature distributions while preserving the data contract.

    The transformation is public so ad hoc incoming batches can use exactly the
    same deterministic drift scenario as the predefined DataOps fixture.
    """

    drifted = dataframe.copy()
    spend_range = contract.numeric_ranges["monthly_spend"]
    support_range = contract.numeric_ranges["support_tickets_90d"]
    usage_range = contract.numeric_ranges["usage_hours_monthly"]

    # Range-relative shifts remain meaningful if contract limits are later changed.
    spend_shift = 0.25 * float(spend_range.maximum - spend_range.minimum)
    support_shift = round(0.20 * (support_range.maximum - support_range.minimum))
    usage_shift = 0.25 * float(usage_range.maximum - usage_range.minimum)

    drifted["monthly_spend"] = (
        (drifted["monthly_spend"] + spend_shift)
        .clip(upper=float(spend_range.maximum))
        .round(2)
    )
    drifted["support_tickets_90d"] = (
        (drifted["support_tickets_90d"] + support_shift)
        .clip(upper=_integer_maximum(support_range))
        .astype("int64")
    )
    drifted["usage_hours_monthly"] = (
        (drifted["usage_hours_monthly"] - usage_shift)
        .clip(lower=float(usage_range.minimum))
        .round(2)
    )
    return drifted


def _apply_contract_violations(
    dataframe: pd.DataFrame,
    contract: DataContractConfig,
) -> pd.DataFrame:
    """Create a reproducible negative fixture with documented validation failures."""

    invalid = dataframe.copy()
    invalid.loc[0, "region"] = "unsupported-region"
    invalid.loc[1, "age"] = _integer_minimum(contract.numeric_ranges["age"]) - 1
    invalid.loc[2, "customer_id"] = invalid.loc[0, "customer_id"]
    return invalid


def _generate_customer_record(
    *,
    row_number: int,
    seed: int,
    random_generator: random.Random,
    contract: DataContractConfig,
) -> dict[str, object]:
    age = _sample_integer(
        random_generator,
        contract.numeric_ranges["age"],
        mode_fraction=0.35,
    )
    maximum_tenure = min(
        _integer_maximum(contract.numeric_ranges["tenure_months"]),
        (age - 18) * 12,
    )
    tenure_months = random_generator.randint(
        _integer_minimum(contract.numeric_ranges["tenure_months"]),
        maximum_tenure,
    )
    monthly_spend = _sample_float(
        random_generator,
        contract.numeric_ranges["monthly_spend"],
        mode_fraction=0.25,
    )
    support_tickets = _sample_integer(
        random_generator,
        contract.numeric_ranges["support_tickets_90d"],
        mode_fraction=0.1,
    )
    late_payments = _sample_integer(
        random_generator,
        contract.numeric_ranges["late_payments_12m"],
        mode_fraction=0.05,
    )
    usage_hours = _sample_float(
        random_generator,
        contract.numeric_ranges["usage_hours_monthly"],
        mode_fraction=0.3,
    )
    plan_type = random_generator.choice(contract.allowed_categories["plan_type"])
    region = random_generator.choice(contract.allowed_categories["region"])

    churn_probability = _calculate_churn_probability(
        tenure_months=tenure_months,
        monthly_spend=monthly_spend,
        support_tickets=support_tickets,
        late_payments=late_payments,
        usage_hours=usage_hours,
        plan_type=plan_type,
        contract=contract,
    )
    churned = int(random_generator.random() < churn_probability)
    behavior = _sharpen_behavior_for_label(
        churned=churned,
        age=age,
        tenure_months=tenure_months,
        monthly_spend=monthly_spend,
        support_tickets=support_tickets,
        late_payments=late_payments,
        usage_hours=usage_hours,
        contract=contract,
    )

    return {
        # Including the seed prevents identifier collisions between independently
        # generated scenario files while keeping every identifier reproducible.
        "customer_id": f"CUST-{seed:06d}-{row_number:06d}",
        "age": age,
        "tenure_months": behavior["tenure_months"],
        "monthly_spend": behavior["monthly_spend"],
        "support_tickets_90d": behavior["support_tickets"],
        "late_payments_12m": behavior["late_payments"],
        "usage_hours_monthly": behavior["usage_hours"],
        "plan_type": plan_type,
        "region": region,
        "churned": churned,
    }


def _sharpen_behavior_for_label(
    *,
    churned: int,
    age: int,
    tenure_months: int,
    monthly_spend: float,
    support_tickets: int,
    late_payments: int,
    usage_hours: float,
    contract: DataContractConfig,
) -> dict[str, int | float]:
    """Make synthetic labels easier to learn while preserving valid ranges."""

    tenure_range = contract.numeric_ranges["tenure_months"]
    spend_range = contract.numeric_ranges["monthly_spend"]
    support_range = contract.numeric_ranges["support_tickets_90d"]
    late_range = contract.numeric_ranges["late_payments_12m"]
    usage_range = contract.numeric_ranges["usage_hours_monthly"]
    maximum_tenure = min(_integer_maximum(tenure_range), (age - 18) * 12)

    if churned:
        return {
            "tenure_months": max(_integer_minimum(tenure_range), tenure_months - 12),
            "monthly_spend": round(
                min(float(spend_range.maximum), monthly_spend + 25.0),
                2,
            ),
            "support_tickets": min(
                _integer_maximum(support_range), support_tickets + 2
            ),
            "late_payments": min(_integer_maximum(late_range), late_payments + 1),
            "usage_hours": round(
                max(float(usage_range.minimum), usage_hours - 35.0), 2
            ),
        }

    return {
        "tenure_months": min(maximum_tenure, tenure_months + 12),
        "monthly_spend": round(
            max(float(spend_range.minimum), monthly_spend - 15.0), 2
        ),
        "support_tickets": max(_integer_minimum(support_range), support_tickets - 1),
        "late_payments": max(_integer_minimum(late_range), late_payments - 1),
        "usage_hours": round(min(float(usage_range.maximum), usage_hours + 25.0), 2),
    }


def _calculate_churn_probability(
    *,
    tenure_months: int,
    monthly_spend: float,
    support_tickets: int,
    late_payments: int,
    usage_hours: float,
    plan_type: str,
    contract: DataContractConfig,
) -> float:
    """Calculate bounded churn probability from understandable feature effects."""

    risk_score = (
        -3.2
        - 4.0 * _normalize(tenure_months, contract.numeric_ranges["tenure_months"])
        + 1.5 * _normalize(monthly_spend, contract.numeric_ranges["monthly_spend"])
        + 6.0
        * _normalize(
            support_tickets,
            contract.numeric_ranges["support_tickets_90d"],
        )
        + 5.5 * _normalize(late_payments, contract.numeric_ranges["late_payments_12m"])
        - 5.0 * _normalize(usage_hours, contract.numeric_ranges["usage_hours_monthly"])
        + _plan_risk_adjustment(plan_type)
    )
    return 1.0 / (1.0 + math.exp(-risk_score))


def _plan_risk_adjustment(plan_type: str) -> float:
    """Represent a small synthetic relationship between plan and churn."""

    adjustments = {
        "basic": 0.90,
        "standard": 0.0,
        "premium": -0.75,
    }
    return adjustments.get(plan_type, 0.0)


def _normalize(value: int | float, accepted_range: NumericRange) -> float:
    width = float(accepted_range.maximum - accepted_range.minimum)
    return (float(value) - float(accepted_range.minimum)) / width


def _sample_integer(
    random_generator: random.Random,
    accepted_range: NumericRange,
    *,
    mode_fraction: float,
) -> int:
    minimum = _integer_minimum(accepted_range)
    maximum = _integer_maximum(accepted_range)
    mode = minimum + (maximum - minimum) * mode_fraction
    sampled_value = round(random_generator.triangular(minimum, maximum, mode))
    return min(maximum, max(minimum, sampled_value))


def _sample_float(
    random_generator: random.Random,
    accepted_range: NumericRange,
    *,
    mode_fraction: float,
) -> float:
    minimum = float(accepted_range.minimum)
    maximum = float(accepted_range.maximum)
    mode = minimum + (maximum - minimum) * mode_fraction
    return round(random_generator.triangular(minimum, maximum, mode), 2)


def _integer_minimum(accepted_range: NumericRange) -> int:
    return math.ceil(accepted_range.minimum)


def _integer_maximum(accepted_range: NumericRange) -> int:
    return math.floor(accepted_range.maximum)


def _validate_generation_inputs(row_count: int, seed: int) -> None:
    if type(row_count) is not int or row_count < 1:
        message = "row_count must be a positive integer"
        raise ValueError(message)
    if type(seed) is not int or seed < 0:
        message = "seed must be a non-negative integer"
        raise ValueError(message)


def _build_typed_dataframe(records: list[dict[str, object]]) -> pd.DataFrame:
    dataframe = pd.DataFrame.from_records(records)
    return dataframe.astype(
        {
            "customer_id": "string",
            "age": "int64",
            "tenure_months": "int64",
            "monthly_spend": "float64",
            "support_tickets_90d": "int64",
            "late_payments_12m": "int64",
            "usage_hours_monthly": "float64",
            "plan_type": "string",
            "region": "string",
            "churned": "int64",
        }
    )


if __name__ == "__main__":
    configure_logging()
    raise SystemExit(main())
