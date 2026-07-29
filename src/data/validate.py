"""Deterministic in-memory validation of customer-churn data.

This module adapts Pandera's library-specific errors into stable project models.
Filesystem loading, report writing, and CLI behavior are intentionally deferred to
later steps so the validation core remains pure and easy to test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Real

import pandas as pd
import pandera.pandas as pa
from src.data.schema import build_customer_churn_schema
from src.data.settings import DataContractConfig
from src.data.validation_models import ValidationIssue, ValidationResult

DATAFRAME_CHECKS = frozenset(
    {
        "minimum_batch_size",
        "duplicate_rows",
        "tenure_consistent_with_age",
    }
)

CHECK_CODES = {
    "column_in_dataframe": "missing_column",
    "column_in_schema": "unexpected_column",
    "column_ordered": "incorrect_column_order",
    "field_uniqueness": "duplicate_customer_id",
    "not_nullable": "null_value",
    "non_empty_customer_id": "empty_customer_id",
    "minimum_batch_size": "minimum_batch_size",
    "duplicate_rows": "duplicate_rows",
    "tenure_consistent_with_age": "tenure_inconsistent_with_age",
    "allowed_plan_type": "invalid_category",
    "allowed_region": "invalid_category",
    "allowed_target_values": "invalid_target",
}

ISSUE_MESSAGES = {
    "missing_column": "A required column is missing.",
    "unexpected_column": "An unexpected column is not allowed by the contract.",
    "incorrect_column_order": "Columns are not in the required contract order.",
    "duplicate_customer_id": "Customer identifiers must be unique within the batch.",
    "null_value": "The column does not allow null values.",
    "empty_customer_id": "Customer identifiers must contain non-whitespace text.",
    "minimum_batch_size": "The dataset does not meet the minimum batch size.",
    "duplicate_rows": "The dataset contains completely duplicated rows.",
    "tenure_inconsistent_with_age": (
        "Customer tenure cannot begin before the customer reaches age 18."
    ),
    "invalid_category": "The column contains a category outside the allowlist.",
    "invalid_target": "The churn target must contain only 0 or 1.",
    "out_of_range": "A numerical value is outside the configured inclusive range.",
    "invalid_dtype": "The column data type does not match the contract.",
    "contract_violation": "The dataset failed a data-contract check.",
}


@dataclass
class _IssueAccumulator:
    """Collect unique failures before creating an immutable bounded issue."""

    code: str
    column: str | None
    check: str
    message: str
    failure_markers: set[str] = field(default_factory=set)
    examples: set[str] = field(default_factory=set)


def validate_dataframe(
    dataframe: pd.DataFrame,
    config: DataContractConfig,
    *,
    dataset_name: str = "in_memory",
) -> ValidationResult:
    """Validate a dataframe and return a stable result instead of raising.

    Pandera runs with ``lazy=True`` to collect independent contract failures in one
    pass. Only expected schema failures are converted; unexpected programming or
    library errors continue to raise so they cannot be mistaken for bad input data.

    Args:
        dataframe: Customer-churn records to validate.
        config: Validated contract policy used to build the Pandera schema.
        dataset_name: Logical filename or label included in the result.

    Returns:
        A deterministic accepted or rejected validation result.
    """

    schema = build_customer_churn_schema(config)
    try:
        schema.validate(dataframe, lazy=True)
    except pa.errors.SchemaErrors as error:
        issues = _normalize_schema_errors(
            error,
            maximum_examples=config.maximum_failure_examples,
        )
        return ValidationResult.create(
            schema_version=config.schema_version,
            dataset_name=dataset_name,
            row_count=len(dataframe),
            column_count=len(dataframe.columns),
            issues=issues,
        )

    return ValidationResult.create(
        schema_version=config.schema_version,
        dataset_name=dataset_name,
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
    )


def _normalize_schema_errors(
    error: pa.errors.SchemaErrors,
    *,
    maximum_examples: int,
) -> tuple[ValidationIssue, ...]:
    accumulators: dict[tuple[str, str | None, str], _IssueAccumulator] = {}

    for failure in error.failure_cases.to_dict(orient="records"):
        check = _stringify(failure.get("check"))
        code = _resolve_issue_code(check)
        column = _resolve_column(
            check=check,
            reported_column=failure.get("column"),
            failure_case=failure.get("failure_case"),
        )
        key = (code, column, check)
        accumulator = accumulators.setdefault(
            key,
            _IssueAccumulator(
                code=code,
                column=column,
                check=check,
                message=ISSUE_MESSAGES[code],
            ),
        )

        marker, example = _failure_marker_and_example(
            check=check,
            index=failure.get("index"),
            failure_case=failure.get("failure_case"),
        )
        accumulator.failure_markers.add(marker)
        accumulator.examples.add(example)

    if not accumulators:
        # Pandera normally provides failure cases, but preserving a generic issue
        # ensures a rejected dataframe can never become a contradictory valid result.
        accumulators[("contract_violation", None, "unknown")] = _IssueAccumulator(
            code="contract_violation",
            column=None,
            check="unknown",
            message=ISSUE_MESSAGES["contract_violation"],
            failure_markers={"unknown"},
            examples={"No structured failure details were returned."},
        )

    return tuple(
        ValidationIssue.create(
            code=accumulator.code,
            column=accumulator.column,
            check=accumulator.check,
            message=accumulator.message,
            failure_count=len(accumulator.failure_markers),
            failure_examples=accumulator.examples,
            maximum_examples=maximum_examples,
        )
        for accumulator in accumulators.values()
    )


def _resolve_issue_code(check: str) -> str:
    if check in CHECK_CODES:
        return CHECK_CODES[check]
    if check.endswith("_range"):
        return "out_of_range"
    if check.startswith("dtype("):
        return "invalid_dtype"
    return "contract_violation"


def _resolve_column(
    *,
    check: str,
    reported_column: object,
    failure_case: object,
) -> str | None:
    if check in DATAFRAME_CHECKS:
        return None
    if check == "column_ordered":
        return None
    if check in {"column_in_dataframe", "column_in_schema"}:
        return _optional_string(failure_case)
    return _optional_string(reported_column)


def _failure_marker_and_example(
    *,
    check: str,
    index: object,
    failure_case: object,
) -> tuple[str, str]:
    normalized_index = _optional_string(index)
    normalized_value = _stringify(failure_case)

    if check in DATAFRAME_CHECKS and normalized_index is not None:
        row_marker = f"row={normalized_index}"
        return row_marker, row_marker

    if normalized_index is None:
        return normalized_value, normalized_value

    example = f"row={normalized_index}, value={normalized_value}"
    return example, example


def _optional_string(value: object) -> str | None:
    if _is_null(value):
        return None
    return str(value)


def _stringify(value: object) -> str:
    if _is_null(value):
        return "<null>"
    return str(value)


def _is_null(value: object) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    return isinstance(value, Real) and math.isnan(float(value))
