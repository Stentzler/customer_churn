"""Local orchestration from one incoming CSV to DagsHub model registration.

The individual data and training modules remain responsible for their own
business rules. This module only coordinates those trusted steps, stops on a
failed gate, and avoids registering the same curated data version twice.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

from src.data.drift import DataDriftError, analyze_drift
from src.data.ingest import (
    BatchDisposition,
    IncomingBatchOperationalError,
    process_incoming_batch,
)
from src.data.settings import DataContractConfigurationError
from src.data.validate import DataValidationOperationalError
from src.training.registry import (
    DEFAULT_DRIFT_DIRECTORY,
    DEFAULT_ENV_PATH,
    DEFAULT_METRICS_DIRECTORY,
    DEFAULT_MODEL_DIRECTORY,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_PARAMS_PATH,
    DEFAULT_PLAN_PATH,
    DEFAULT_PROFILE_PATH,
    ExperimentTrackingError,
    TrackingConfigurationError,
    load_tracking_settings,
    track_and_register_candidates,
)

LOGGER = logging.getLogger(__name__)
LOG_FORMAT = "%(levelname)s %(name)s %(message)s"
DEFAULT_DATA_ROOT = Path("data")
DEFAULT_QUALITY_REPORT_DIRECTORY = Path("reports/data-quality")
DvcCommandRunner = Callable[[Sequence[str]], None]


class LocalPipelineError(RuntimeError):
    """Raised when local orchestration cannot complete safely."""


class LocalPipelineStatus(StrEnum):
    """Terminal outcomes exposed to local automation and future CI."""

    REGISTERED = "registered"
    SKIPPED_UNCHANGED = "skipped_unchanged"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class LocalPipelineResult:
    """Structured summary of one local incoming-batch execution."""

    status: LocalPipelineStatus
    input_path: Path
    data_version: str | None
    registered_model_name: str | None = None
    registered_model_version: str | None = None


def run_local_pipeline(
    input_path: Path,
    *,
    params_path: Path = DEFAULT_PARAMS_PATH,
    env_path: Path = DEFAULT_ENV_PATH,
    data_root: Path = DEFAULT_DATA_ROOT,
    quality_report_directory: Path = DEFAULT_QUALITY_REPORT_DIRECTORY,
    profile_path: Path = DEFAULT_PROFILE_PATH,
    drift_directory: Path = DEFAULT_DRIFT_DIRECTORY,
    model_directory: Path = DEFAULT_MODEL_DIRECTORY,
    metrics_directory: Path = DEFAULT_METRICS_DIRECTORY,
    plan_path: Path = DEFAULT_PLAN_PATH,
    tracking_output_path: Path = DEFAULT_OUTPUT_PATH,
    dvc_command_runner: DvcCommandRunner | None = None,
) -> LocalPipelineResult:
    """Process one batch and register a model only for a new curated data version.

    Validation is the first gate. Rejected data receives its reports and routed
    copy, but no drift, curation, training, or remote tracking is attempted.
    """

    command_runner = dvc_command_runner or _run_dvc_command
    ingestion = process_incoming_batch(
        input_path=input_path,
        params_path=params_path,
        data_root=data_root,
        report_directory=quality_report_directory,
    )
    if ingestion.disposition is BatchDisposition.REJECTED:
        return LocalPipelineResult(
            status=LocalPipelineStatus.REJECTED,
            input_path=input_path,
            data_version=None,
        )

    drift_artifacts = analyze_drift(
        current_path=ingestion.routed_path,
        params_path=params_path,
        report_directory=drift_directory,
    )
    _version_batch_artifacts(
        command_runner,
        input_path=input_path,
        accepted_path=ingestion.routed_path,
        validation_json_path=ingestion.reports.json_path,
        validation_markdown_path=ingestion.reports.markdown_path,
        drift_json_path=drift_artifacts.json_path,
        drift_html_path=drift_artifacts.html_path,
    )

    previous_data_version = _read_optional_data_version(tracking_output_path)
    # Profile and train are sibling DVC stages, so both targets are explicit.
    command_runner(("repro", "profile", "train"))
    current_data_version = _read_required_data_version(profile_path)

    if current_data_version == previous_data_version:
        return LocalPipelineResult(
            status=LocalPipelineStatus.SKIPPED_UNCHANGED,
            input_path=input_path,
            data_version=current_data_version,
        )

    settings = load_tracking_settings(params_path, env_path)
    tracking = track_and_register_candidates(
        settings,
        params_path=params_path,
        model_directory=model_directory,
        metrics_directory=metrics_directory,
        plan_path=plan_path,
        profile_path=profile_path,
        drift_directory=drift_directory,
        output_path=tracking_output_path,
    )
    return LocalPipelineResult(
        status=LocalPipelineStatus.REGISTERED,
        input_path=input_path,
        data_version=current_data_version,
        registered_model_name=tracking.registered_model_name,
        registered_model_version=tracking.registered_model_version,
    )


def _version_batch_artifacts(
    command_runner: DvcCommandRunner,
    *,
    input_path: Path,
    accepted_path: Path,
    validation_json_path: Path,
    validation_markdown_path: Path,
    drift_json_path: Path,
    drift_html_path: Path,
) -> None:
    """Place raw, accepted, validation, and drift artifacts in the DVC cache."""

    command_runner(
        (
            "add",
            str(input_path),
            str(accepted_path),
            str(validation_json_path),
            str(validation_markdown_path),
            str(drift_json_path),
            str(drift_html_path),
        )
    )


def _run_dvc_command(arguments: Sequence[str]) -> None:
    """Execute DVC with inherited output and actionable failure context."""

    command = ("uv", "run", "dvc", *arguments)
    LOGGER.info("dvc_command_started command=%s", " ".join(command))
    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise LocalPipelineError(f"DVC command failed: {' '.join(command)}") from error


def _read_optional_data_version(path: Path) -> str | None:
    if not path.is_file():
        return None
    return _read_data_version(path)


def _read_required_data_version(path: Path) -> str:
    if not path.is_file():
        raise LocalPipelineError(f"Required data profile does not exist: '{path}'")
    return _read_data_version(path)


def _read_data_version(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LocalPipelineError(
            f"Cannot read data version from '{path}': {error}"
        ) from error
    if not isinstance(payload, dict):
        raise LocalPipelineError(f"Data-version artifact must be an object: '{path}'")
    data_version = cast(dict[str, object], payload).get("data_version")
    if not isinstance(data_version, str) or not data_version:
        raise LocalPipelineError(
            f"Data-version artifact has no valid data_version: '{path}'"
        )
    return data_version


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one local CSV through DataOps, training, and registration.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="CSV located directly under data/incoming.",
    )
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH)
    parser.add_argument("--env", type=Path, default=DEFAULT_ENV_PATH)
    return parser


def configure_logging() -> None:
    """Configure concise default logging for direct CLI execution."""

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the local lifecycle and return a shell-friendly status code."""

    args = _build_argument_parser().parse_args(arguments)
    configure_logging()
    try:
        result = run_local_pipeline(
            args.input,
            params_path=args.params,
            env_path=args.env,
        )
    except (
        DataContractConfigurationError,
        DataDriftError,
        DataValidationOperationalError,
        ExperimentTrackingError,
        IncomingBatchOperationalError,
        LocalPipelineError,
        TrackingConfigurationError,
    ) as error:
        LOGGER.error("local_pipeline_failed reason=%s", error)
        return 2

    log_method = (
        LOGGER.warning if result.status is LocalPipelineStatus.REJECTED else LOGGER.info
    )
    log_method(
        "local_pipeline_completed status=%s input=%s data_version=%s "
        "model=%s model_version=%s",
        result.status.value,
        result.input_path,
        result.data_version,
        result.registered_model_name,
        result.registered_model_version,
    )
    return 1 if result.status is LocalPipelineStatus.REJECTED else 0


if __name__ == "__main__":
    raise SystemExit(main())
