"""Reference-based feature and target drift evaluation with Evidently."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.metrics import ValueDrift
from evidently.presets import DataDriftPreset
from src.data.schema import CUSTOMER_TARGET_COLUMN
from src.data.settings import (
    CATEGORY_NAMES,
    NUMERIC_RANGE_NAMES,
    DataContractConfig,
    DataContractConfigurationError,
    DriftConfig,
    load_data_contract,
    load_drift_config,
)
from src.data.validate import validate_dataframe

LOGGER = logging.getLogger(__name__)
LOG_FORMAT = "%(levelname)s %(name)s %(message)s"

FEATURE_NAMES = (
    "age",
    "tenure_months",
    "monthly_spend",
    "support_tickets_90d",
    "late_payments_12m",
    "usage_hours_monthly",
    "plan_type",
    "region",
)


class DataDriftError(RuntimeError):
    """Raised when drift inputs or Evidently output cannot be trusted."""


class DriftSnapshot(Protocol):
    """Minimal Evidently snapshot behavior used by the persistence boundary."""

    def dict(self) -> dict[str, object]:
        """Return metric results."""

    def save_html(self, filename: str) -> None:
        """Save the visual report."""


@dataclass(frozen=True, slots=True)
class ColumnDrift:
    """Normalized drift result for one feature or target column."""

    column: str
    method: str
    score: float
    threshold: float
    is_drifted: bool


@dataclass(frozen=True, slots=True)
class FeatureDrift:
    """Dataset-level decision calculated only from model input features."""

    columns: tuple[ColumnDrift, ...]
    drifted_features: tuple[str, ...]
    drifted_feature_count: int
    feature_count: int
    drift_share: float
    drift_share_threshold: float
    is_significant: bool


@dataclass(frozen=True, slots=True)
class TargetDrift:
    """Target drift result kept separate from the feature drift gate."""

    column: ColumnDrift
    reference_distribution: Mapping[str, float]
    current_distribution: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class DriftResult:
    """Stable project-owned output for one reference/current comparison."""

    reference_dataset: str
    current_dataset: str
    reference_data_version: str
    current_data_version: str
    feature_drift: FeatureDrift
    target_drift: TargetDrift


@dataclass(frozen=True, slots=True)
class DriftArtifacts:
    """Structured result and report paths created by one drift run."""

    result: DriftResult
    json_path: Path
    html_path: Path


def evaluate_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    config: DriftConfig,
    *,
    reference_name: str,
    current_name: str,
    reference_data_version: str,
    current_data_version: str,
) -> tuple[DriftResult, DriftSnapshot]:
    """Evaluate feature and target drift and return the Evidently snapshot."""

    definition = DataDefinition(
        numerical_columns=sorted(NUMERIC_RANGE_NAMES),
        categorical_columns=[*sorted(CATEGORY_NAMES), CUSTOMER_TARGET_COLUMN],
    )
    selected_columns = [*FEATURE_NAMES, CUSTOMER_TARGET_COLUMN]
    reference_dataset = Dataset.from_pandas(
        reference.loc[:, selected_columns],
        data_definition=definition,
    )
    current_dataset = Dataset.from_pandas(
        current.loc[:, selected_columns],
        data_definition=definition,
    )
    report = Report(
        [
            DataDriftPreset(
                columns=list(FEATURE_NAMES),
                drift_share=config.feature_drift_share_threshold,
            ),
            ValueDrift(column=CUSTOMER_TARGET_COLUMN),
        ]
    )
    snapshot = report.run(
        current_data=current_dataset,
        reference_data=reference_dataset,
    )
    result = _normalize_evidently_result(
        cast(dict[str, object], snapshot.dict()),
        reference,
        current,
        config,
        reference_name=reference_name,
        current_name=current_name,
        reference_data_version=reference_data_version,
        current_data_version=current_data_version,
    )
    return result, cast(DriftSnapshot, snapshot)


def analyze_drift(
    current_path: Path,
    params_path: Path,
    report_directory: Path,
) -> DriftArtifacts:
    """Compare one accepted batch with the configured fixed reference dataset."""

    drift_config = load_drift_config(params_path)
    contract = load_data_contract(params_path)
    _require_accepted_current_path(current_path, drift_config.reference_path)
    reference, reference_version = _read_csv_with_hash(drift_config.reference_path)
    current, current_version = _read_csv_with_hash(current_path)
    _require_valid_dataset(reference, drift_config.reference_path, contract)
    _require_valid_dataset(current, current_path, contract)

    result, snapshot = evaluate_drift(
        reference,
        current,
        drift_config,
        reference_name=drift_config.reference_path.name,
        current_name=current_path.name,
        reference_data_version=reference_version,
        current_data_version=current_version,
    )
    json_path = report_directory / f"{current_path.stem}.drift.json"
    html_path = report_directory / f"{current_path.stem}.drift.html"
    _write_drift_reports(result, snapshot, json_path, html_path)
    return DriftArtifacts(result=result, json_path=json_path, html_path=html_path)


def render_drift_json(result: DriftResult) -> str:
    """Render deterministic drift details without raw customer records."""

    payload = {
        "current_data_version": result.current_data_version,
        "current_dataset": result.current_dataset,
        "feature_drift": {
            "columns": [
                _column_payload(column) for column in result.feature_drift.columns
            ],
            "drift_share": result.feature_drift.drift_share,
            "drift_share_threshold": result.feature_drift.drift_share_threshold,
            "drifted_feature_count": result.feature_drift.drifted_feature_count,
            "drifted_features": list(result.feature_drift.drifted_features),
            "feature_count": result.feature_drift.feature_count,
            "is_significant": result.feature_drift.is_significant,
        },
        "reference_data_version": result.reference_data_version,
        "reference_dataset": result.reference_dataset,
        "target_drift": {
            **_column_payload(result.target_drift.column),
            "current_distribution": result.target_drift.current_distribution,
            "reference_distribution": result.target_drift.reference_distribution,
        },
    }
    return f"{json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True)}\n"


def main(arguments: Sequence[str] | None = None) -> int:
    """Run drift evaluation for one accepted batch."""

    parsed_arguments = _build_argument_parser().parse_args(arguments)
    try:
        artifacts = analyze_drift(
            parsed_arguments.current,
            parsed_arguments.params,
            parsed_arguments.report_dir,
        )
    except (DataContractConfigurationError, DataDriftError) as error:
        LOGGER.error("drift_error reason=%s", error)
        return 2

    LOGGER.info(
        "drift_completed current=%s feature_drift=%s drift_share=%.3f "
        "target_drift=%s json=%s html=%s",
        artifacts.result.current_dataset,
        artifacts.result.feature_drift.is_significant,
        artifacts.result.feature_drift.drift_share,
        artifacts.result.target_drift.column.is_drifted,
        artifacts.json_path,
        artifacts.html_path,
    )
    return 0


def configure_logging() -> None:
    """Configure concise default logging for direct CLI execution."""

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare one accepted batch against the fixed reference data.",
    )
    parser.add_argument(
        "--current",
        type=Path,
        required=True,
        help="Accepted CSV batch to compare with the configured reference.",
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
        default=Path("reports/drift"),
        help="Drift report directory (default: reports/drift).",
    )
    return parser


def _normalize_evidently_result(
    payload: dict[str, object],
    reference: pd.DataFrame,
    current: pd.DataFrame,
    config: DriftConfig,
    *,
    reference_name: str,
    current_name: str,
    reference_data_version: str,
    current_data_version: str,
) -> DriftResult:
    try:
        raw_metrics = payload["metrics"]
        if not isinstance(raw_metrics, list):
            raise TypeError("metrics must be a list")
        columns = tuple(
            sorted(
                (
                    _parse_column_drift(metric)
                    for metric in raw_metrics
                    if _is_value_drift_metric(metric)
                    and _metric_column(metric) in FEATURE_NAMES
                ),
                key=lambda result: FEATURE_NAMES.index(result.column),
            )
        )
        target_column = next(
            _parse_column_drift(metric)
            for metric in raw_metrics
            if _is_value_drift_metric(metric)
            and _metric_column(metric) == CUSTOMER_TARGET_COLUMN
        )
    except (KeyError, StopIteration, TypeError, ValueError) as error:
        message = f"Evidently returned an unsupported result structure: {error}"
        raise DataDriftError(message) from error

    if len(columns) != len(FEATURE_NAMES):
        message = "Evidently did not return drift results for every feature"
        raise DataDriftError(message)

    drifted_features = tuple(column.column for column in columns if column.is_drifted)
    drift_share = len(drifted_features) / len(columns)
    feature_drift = FeatureDrift(
        columns=columns,
        drifted_features=drifted_features,
        drifted_feature_count=len(drifted_features),
        feature_count=len(columns),
        drift_share=drift_share,
        drift_share_threshold=config.feature_drift_share_threshold,
        is_significant=drift_share >= config.feature_drift_share_threshold,
    )
    return DriftResult(
        reference_dataset=Path(reference_name).name,
        current_dataset=Path(current_name).name,
        reference_data_version=reference_data_version,
        current_data_version=current_data_version,
        feature_drift=feature_drift,
        target_drift=TargetDrift(
            column=target_column,
            reference_distribution=_target_distribution(reference),
            current_distribution=_target_distribution(current),
        ),
    )


def _is_value_drift_metric(metric: object) -> bool:
    if not isinstance(metric, dict):
        return False
    config = metric.get("config")
    return isinstance(config, dict) and config.get("type") == (
        "evidently:metric_v2:ValueDrift"
    )


def _metric_column(metric: object) -> str:
    metric_mapping = _require_mapping(metric, "metric")
    config = _require_mapping(metric_mapping.get("config"), "metric.config")
    column = config.get("column")
    if not isinstance(column, str):
        raise TypeError("metric.config.column must be a string")
    return column


def _parse_column_drift(metric: object) -> ColumnDrift:
    metric_mapping = _require_mapping(metric, "metric")
    config = _require_mapping(metric_mapping.get("config"), "metric.config")
    method = config.get("method")
    score = metric_mapping.get("value")
    threshold = config.get("threshold")
    if not isinstance(method, str):
        raise TypeError("metric.config.method must be a string")
    if not isinstance(score, (int, float)):
        raise TypeError("metric.value must be numerical")
    if not isinstance(threshold, (int, float)):
        raise TypeError("metric.config.threshold must be numerical")

    numeric_score = float(score)
    numeric_threshold = float(threshold)
    return ColumnDrift(
        column=_metric_column(metric),
        method=method,
        score=numeric_score,
        threshold=numeric_threshold,
        is_drifted=_is_score_drifted(
            numeric_score,
            numeric_threshold,
            method,
        ),
    )


def _is_score_drifted(score: float, threshold: float, method: str) -> bool:
    return score < threshold if "p_value" in method.lower() else score >= threshold


def _target_distribution(dataframe: pd.DataFrame) -> dict[str, float]:
    proportions = dataframe[CUSTOMER_TARGET_COLUMN].value_counts(
        normalize=True,
        sort=False,
    )
    return {
        str(value): float(proportion)
        for value, proportion in sorted(
            proportions.items(),
            key=lambda item: str(item[0]),
        )
    }


def _column_payload(column: ColumnDrift) -> dict[str, object]:
    return {
        "column": column.column,
        "is_drifted": column.is_drifted,
        "method": column.method,
        "score": column.score,
        "threshold": column.threshold,
    }


def _read_csv_with_hash(path: Path) -> tuple[pd.DataFrame, str]:
    try:
        content = path.read_bytes()
        dataframe = pd.read_csv(path)
    except (
        OSError,
        UnicodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as error:
        message = f"Cannot read drift dataset '{path}': {error}"
        raise DataDriftError(message) from error
    return dataframe, hashlib.sha256(content).hexdigest()


def _require_valid_dataset(
    dataframe: pd.DataFrame,
    path: Path,
    contract: DataContractConfig,
) -> None:
    validation = validate_dataframe(dataframe, contract)
    if not validation.is_valid:
        issue_codes = ", ".join(issue.code for issue in validation.issues)
        message = f"Drift dataset '{path}' is invalid: {issue_codes}"
        raise DataDriftError(message)


def _write_drift_reports(
    result: DriftResult,
    snapshot: DriftSnapshot,
    json_path: Path,
    html_path: Path,
) -> None:
    json_temporary = json_path.with_suffix(".json.tmp")
    html_temporary = html_path.with_suffix(".html.tmp")
    try:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_temporary.write_text(render_drift_json(result), encoding="utf-8")
        snapshot.save_html(str(html_temporary))
        json_temporary.replace(json_path)
        html_temporary.replace(html_path)
    except OSError as error:
        json_temporary.unlink(missing_ok=True)
        html_temporary.unlink(missing_ok=True)
        message = f"Cannot write drift reports to '{json_path.parent}': {error}"
        raise DataDriftError(message) from error


def _require_accepted_current_path(
    current_path: Path,
    reference_path: Path,
) -> None:
    data_root = reference_path.resolve().parent.parent
    accepted_directory = data_root / "accepted"
    if current_path.resolve().parent != accepted_directory:
        message = (
            f"Current drift dataset '{current_path}' must be located directly under "
            f"'{accepted_directory}'"
        )
        raise DataDriftError(message)


def _require_mapping(value: object, location: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{location} must be a mapping")
    return cast(dict[str, object], value)


if __name__ == "__main__":
    configure_logging()
    raise SystemExit(main())
