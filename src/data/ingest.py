"""Validate and route incoming customer-churn batches.

Incoming files are preserved as raw deliveries. After validation and report
creation, each file is copied to exactly one controlled destination: accepted data
may continue to curation, while rejected data remains isolated for investigation.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from src.data.settings import DataContractConfigurationError
from src.data.validate import (
    DataValidationOperationalError,
    validate_csv,
)
from src.data.validation_models import ValidationResult
from src.data.validation_report import (
    ValidationReportPaths,
    validation_report_paths,
)

LOGGER = logging.getLogger(__name__)
LOG_FORMAT = "%(levelname)s %(name)s %(message)s"


class BatchDisposition(StrEnum):
    """Allowlisted destinations for a validated incoming batch."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"


class IncomingBatchOperationalError(RuntimeError):
    """Raised when an incoming batch cannot be safely routed."""


@dataclass(frozen=True, slots=True)
class IncomingBatchResult:
    """Structured outcome consumed by later DataOps stages."""

    disposition: BatchDisposition
    source_path: Path
    routed_path: Path
    validation: ValidationResult
    reports: ValidationReportPaths


def process_incoming_batch(
    input_path: Path,
    params_path: Path,
    data_root: Path,
    report_directory: Path,
) -> IncomingBatchResult:
    """Validate one raw delivery and copy it to its controlled destination.

    Args:
        input_path: CSV file located directly under ``data_root/incoming``.
        params_path: Versioned configuration containing the data contract.
        data_root: Root containing incoming, accepted, and rejected directories.
        report_directory: Destination for JSON and Markdown validation reports.

    Returns:
        A structured accepted or rejected routing result.

    Raises:
        IncomingBatchOperationalError: If the source is outside the incoming
            boundary or cannot be copied atomically.
        DataValidationOperationalError: If validation or report creation fails.
    """

    _require_incoming_source(input_path, data_root)
    validation = validate_csv(input_path, params_path, report_directory)
    disposition = (
        BatchDisposition.ACCEPTED if validation.is_valid else BatchDisposition.REJECTED
    )
    routed_path = data_root / disposition.value / input_path.name
    _copy_atomically(input_path, routed_path)

    return IncomingBatchResult(
        disposition=disposition,
        source_path=input_path,
        routed_path=routed_path,
        validation=validation,
        reports=validation_report_paths(validation.dataset_name, report_directory),
    )


def main(arguments: Sequence[str] | None = None) -> int:
    """Process one incoming CSV and return a pipeline-friendly exit code."""

    parsed_arguments = _build_argument_parser().parse_args(arguments)
    try:
        result = process_incoming_batch(
            input_path=parsed_arguments.input,
            params_path=parsed_arguments.params,
            data_root=parsed_arguments.data_root,
            report_directory=parsed_arguments.report_dir,
        )
    except (
        DataContractConfigurationError,
        DataValidationOperationalError,
        IncomingBatchOperationalError,
    ) as error:
        LOGGER.error("incoming_batch_operational_error reason=%s", error)
        return 2

    log_method = (
        LOGGER.info
        if result.disposition is BatchDisposition.ACCEPTED
        else LOGGER.warning
    )
    log_method(
        "incoming_batch_routed disposition=%s source=%s destination=%s issues=%d",
        result.disposition.value,
        result.source_path,
        result.routed_path,
        len(result.validation.issues),
    )
    return 0 if result.disposition is BatchDisposition.ACCEPTED else 1


def configure_logging() -> None:
    """Configure concise default logging for direct CLI execution."""

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and route one incoming customer-churn CSV batch.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="CSV file located directly under the incoming data directory.",
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
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports/data-quality"),
        help="Validation report directory (default: reports/data-quality).",
    )
    return parser


def _require_incoming_source(input_path: Path, data_root: Path) -> None:
    expected_parent = (data_root / "incoming").resolve()
    actual_parent = input_path.resolve().parent
    if actual_parent != expected_parent:
        message = (
            f"Incoming batch '{input_path}' must be located directly under "
            f"'{data_root / 'incoming'}'"
        )
        raise IncomingBatchOperationalError(message)


def _copy_atomically(source_path: Path, destination_path: Path) -> None:
    temporary_path = destination_path.with_suffix(f"{destination_path.suffix}.tmp")
    try:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, temporary_path)
        temporary_path.replace(destination_path)
    except OSError as error:
        temporary_path.unlink(missing_ok=True)
        message = f"Cannot route incoming batch to '{destination_path}': {error}"
        raise IncomingBatchOperationalError(message) from error


if __name__ == "__main__":
    configure_logging()
    raise SystemExit(main())
