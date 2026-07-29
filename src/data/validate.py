"""Deterministic validation of customer-churn data.

This module adapts Pandera's library-specific errors into stable project models.
The in-memory validation core remains pure, while the CSV and CLI functions provide
explicit filesystem boundaries around it.
"""

from __future__ import annotations

import argparse
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from src.data.schema import build_customer_churn_schema
from src.data.settings import (
    DataContractConfig,
    DataContractConfigurationError,
    load_data_contract,
)
from src.data.validation_models import ValidationIssue, ValidationResult
from src.data.validation_report import (
    validation_report_paths,
    write_validation_reports,
)

LOGGER = logging.getLogger(__name__)
LOG_FORMAT = "%(levelname)s %(name)s %(message)s"

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


class DataValidationOperationalError(RuntimeError):
    """Raised when validation cannot run because an external operation failed."""


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


def validate_csv(
    input_path: Path,
    params_path: Path,
    report_directory: Path,
) -> ValidationResult:
    """Load, validate, and report one CSV batch.

    Args:
        input_path: CSV batch to read through pandas.
        params_path: Versioned YAML file containing the data contract.
        report_directory: Destination for JSON and Markdown reports.

    Returns:
        The same validation result persisted in the two report artifacts.

    Raises:
        DataContractConfigurationError: If versioned policy is invalid.
        DataValidationOperationalError: If CSV loading or report writing fails.
    """

    dataframe = _read_csv(input_path)
    config = load_data_contract(params_path)
    result = validate_dataframe(
        dataframe,
        config,
        dataset_name=input_path.name,
    )
    try:
        write_validation_reports(result, report_directory)
    except OSError as error:
        message = f"Cannot write validation reports to '{report_directory}': {error}"
        raise DataValidationOperationalError(message) from error
    return result


def main(arguments: Sequence[str] | None = None) -> int:
    """Run CSV validation from the command line and return a process exit code."""

    parser = _build_argument_parser()
    parsed_arguments = parser.parse_args(arguments)

    try:
        result = validate_csv(
            input_path=parsed_arguments.input,
            params_path=parsed_arguments.params,
            report_directory=parsed_arguments.report_dir,
        )
    except (DataContractConfigurationError, DataValidationOperationalError) as error:
        LOGGER.error("validation_operational_error reason=%s", error)
        return 2

    report_paths = validation_report_paths(
        result.dataset_name,
        parsed_arguments.report_dir,
    )
    if result.is_valid:
        LOGGER.info(
            "validation_status=accepted dataset=%s",
            result.dataset_name,
        )
    else:
        LOGGER.warning(
            "validation_status=rejected dataset=%s issue_count=%d",
            result.dataset_name,
            len(result.issues),
        )

    LOGGER.info(
        "dataset_shape dataset=%s rows=%d columns=%d",
        result.dataset_name,
        result.row_count,
        result.column_count,
    )
    LOGGER.info(
        "validation_report format=json path=%s",
        report_paths.json_path,
    )
    LOGGER.info(
        "validation_report format=markdown path=%s",
        report_paths.markdown_path,
    )
    return 0 if result.is_valid else 1


def configure_logging() -> None:
    """Configure concise default logging for direct CLI execution.

    ``basicConfig`` leaves logging unchanged when a host application or test runner
    has already installed handlers, allowing this module to remain reusable.
    """

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a customer-churn CSV batch against the data contract.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="CSV batch to validate.",
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=Path("params.yaml"),
        help="Versioned YAML parameters file (default: params.yaml).",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports/data-quality"),
        help="Validation report directory (default: reports/data-quality).",
    )
    return parser


def _read_csv(input_path: Path) -> pd.DataFrame:
    if not input_path.is_file():
        message = f"Input CSV does not exist or is not a file: '{input_path}'"
        raise DataValidationOperationalError(message)

    try:
        return pd.read_csv(input_path)
    except (
        OSError,
        UnicodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as error:
        message = f"Cannot read input CSV '{input_path}': {error}"
        raise DataValidationOperationalError(message) from error


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


if __name__ == "__main__":
    configure_logging()
    raise SystemExit(main())
