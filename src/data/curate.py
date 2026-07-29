"""Deterministic curation of reference and accepted customer data."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from src.data.schema import CUSTOMER_CHURN_COLUMNS, CUSTOMER_IDENTIFIER_COLUMN
from src.data.settings import (
    DataContractConfig,
    DataContractConfigurationError,
    load_data_contract,
)
from src.data.validate import validate_dataframe

LOGGER = logging.getLogger(__name__)
LOG_FORMAT = "%(levelname)s %(name)s %(message)s"


class DataCurationError(RuntimeError):
    """Raised when trusted curation inputs or outputs violate their contract."""


@dataclass(frozen=True, slots=True)
class CurationResult:
    """Traceable summary of one deterministic curation run."""

    output_path: Path
    source_paths: tuple[Path, ...]
    input_row_count: int
    output_row_count: int
    duplicate_customer_count: int


def curate_training_dataset(
    data_root: Path,
    contract: DataContractConfig,
) -> CurationResult:
    """Merge reference and accepted data into the curated training dataset.

    The source locations are derived rather than caller-provided. This keeps the
    immutable fixed-test directory outside the curation interface by construction.
    Accepted filenames are sorted, and later files take precedence when a customer
    identifier occurs more than once.
    """

    reference_path = data_root / "reference" / "reference.csv"
    accepted_directory = data_root / "accepted"
    output_path = data_root / "curated" / "training.csv"
    source_paths = (reference_path, *_accepted_csv_paths(accepted_directory))

    dataframes = tuple(
        _read_and_validate_source(path, contract) for path in source_paths
    )
    merged = pd.concat(dataframes, ignore_index=True)
    input_row_count = len(merged)
    curated = (
        merged.drop_duplicates(subset=CUSTOMER_IDENTIFIER_COLUMN, keep="last")
        .sort_values(CUSTOMER_IDENTIFIER_COLUMN, kind="stable")
        .reset_index(drop=True)
    )
    duplicate_customer_count = input_row_count - len(curated)

    final_validation = validate_dataframe(
        curated,
        contract,
        dataset_name=output_path.name,
    )
    if not final_validation.is_valid:
        issue_codes = ", ".join(issue.code for issue in final_validation.issues)
        message = f"Curated dataset violates the data contract: {issue_codes}"
        raise DataCurationError(message)

    _write_curated_csv(curated, output_path)
    return CurationResult(
        output_path=output_path,
        source_paths=source_paths,
        input_row_count=input_row_count,
        output_row_count=len(curated),
        duplicate_customer_count=duplicate_customer_count,
    )


def main(arguments: Sequence[str] | None = None) -> int:
    """Build the curated training dataset from controlled data locations."""

    parsed_arguments = _build_argument_parser().parse_args(arguments)
    try:
        contract = load_data_contract(parsed_arguments.params)
        result = curate_training_dataset(parsed_arguments.data_root, contract)
    except (DataContractConfigurationError, DataCurationError) as error:
        LOGGER.error("curation_error reason=%s", error)
        return 2

    LOGGER.info(
        "curation_completed sources=%d input_rows=%d output_rows=%d "
        "duplicates_removed=%d path=%s",
        len(result.source_paths),
        result.input_row_count,
        result.output_row_count,
        result.duplicate_customer_count,
        result.output_path,
    )
    return 0


def configure_logging() -> None:
    """Configure concise default logging for direct CLI execution."""

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Curate reference and accepted customer-churn data.",
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
        help="Root data directory (default: data).",
    )
    return parser


def _accepted_csv_paths(accepted_directory: Path) -> tuple[Path, ...]:
    if not accepted_directory.is_dir():
        message = f"Accepted data directory does not exist: '{accepted_directory}'"
        raise DataCurationError(message)

    paths = tuple(sorted(accepted_directory.glob("*.csv"), key=lambda path: path.name))
    for path in paths:
        if path.is_symlink() or path.resolve().parent != accepted_directory.resolve():
            message = f"Accepted data source must be a regular local CSV: '{path}'"
            raise DataCurationError(message)
    return paths


def _read_and_validate_source(
    source_path: Path,
    contract: DataContractConfig,
) -> pd.DataFrame:
    if not source_path.is_file() or source_path.is_symlink():
        message = f"Curation source does not exist as a regular file: '{source_path}'"
        raise DataCurationError(message)

    try:
        dataframe = pd.read_csv(source_path)
    except (
        OSError,
        UnicodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as error:
        message = f"Cannot read curation source '{source_path}': {error}"
        raise DataCurationError(message) from error

    validation = validate_dataframe(
        dataframe,
        contract,
        dataset_name=source_path.name,
    )
    if not validation.is_valid:
        issue_codes = ", ".join(issue.code for issue in validation.issues)
        message = f"Curation source '{source_path}' is invalid: {issue_codes}"
        raise DataCurationError(message)
    return dataframe.loc[:, CUSTOMER_CHURN_COLUMNS]


def _write_curated_csv(dataframe: pd.DataFrame, output_path: Path) -> None:
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
        message = f"Cannot write curated dataset to '{output_path}': {error}"
        raise DataCurationError(message) from error


if __name__ == "__main__":
    configure_logging()
    raise SystemExit(main())
