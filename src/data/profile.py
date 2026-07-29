"""Deterministic, aggregate-only dataset profiling."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from src.data.schema import (
    CUSTOMER_CHURN_COLUMNS,
    CUSTOMER_IDENTIFIER_COLUMN,
    CUSTOMER_TARGET_COLUMN,
)
from src.data.settings import (
    CATEGORY_NAMES,
    NUMERIC_RANGE_NAMES,
    DataContractConfigurationError,
    load_data_contract,
)
from src.data.validate import validate_dataframe

LOGGER = logging.getLogger(__name__)
LOG_FORMAT = "%(levelname)s %(name)s %(message)s"


class DataProfileError(RuntimeError):
    """Raised when a trusted dataset cannot be profiled safely."""


@dataclass(frozen=True, slots=True)
class NumericSummary:
    """Aggregate statistics for one numerical feature."""

    minimum: float
    maximum: float
    mean: float
    median: float
    standard_deviation: float


@dataclass(frozen=True, slots=True)
class DatasetProfile:
    """Stable aggregate contract for one dataset version."""

    schema_version: str
    dataset_name: str
    data_version: str
    row_count: int
    feature_count: int
    feature_names: tuple[str, ...]
    feature_types: Mapping[str, str]
    missing_value_counts: Mapping[str, int]
    numerical_summaries: Mapping[str, NumericSummary]
    categorical_frequencies: Mapping[str, Mapping[str, int]]
    target_distribution: Mapping[str, int]
    duplicate_row_count: int
    drift_evaluated: bool
    drifted_features: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DatasetProfileArtifact:
    """Profile and path persisted for downstream pipeline stages."""

    profile: DatasetProfile
    output_path: Path


def build_dataset_profile(
    dataframe: pd.DataFrame,
    *,
    schema_version: str,
    dataset_name: str,
    data_version: str,
    drifted_features: Sequence[str] | None = None,
) -> DatasetProfile:
    """Build an aggregate profile without retaining customer-level values."""

    feature_names = tuple(
        column
        for column in CUSTOMER_CHURN_COLUMNS
        if column not in {CUSTOMER_IDENTIFIER_COLUMN, CUSTOMER_TARGET_COLUMN}
    )
    numerical_features = tuple(
        column for column in feature_names if column in NUMERIC_RANGE_NAMES
    )
    categorical_features = tuple(
        column for column in feature_names if column in CATEGORY_NAMES
    )
    ordered_drifted_features = tuple(sorted(set(drifted_features or ())))

    unknown_drifted_features = set(ordered_drifted_features) - set(feature_names)
    if unknown_drifted_features:
        unknown = ", ".join(sorted(unknown_drifted_features))
        raise ValueError(f"Unknown drifted features: {unknown}")

    return DatasetProfile(
        schema_version=schema_version,
        dataset_name=Path(dataset_name).name,
        data_version=data_version,
        row_count=len(dataframe),
        feature_count=len(feature_names),
        feature_names=feature_names,
        feature_types={
            feature_name: str(dataframe[feature_name].dtype)
            for feature_name in feature_names
        },
        missing_value_counts={
            column: int(dataframe[column].isna().sum())
            for column in CUSTOMER_CHURN_COLUMNS
        },
        numerical_summaries={
            feature_name: _summarize_numeric(dataframe[feature_name])
            for feature_name in numerical_features
        },
        categorical_frequencies={
            feature_name: _value_counts(dataframe[feature_name])
            for feature_name in categorical_features
        },
        target_distribution=_value_counts(dataframe[CUSTOMER_TARGET_COLUMN]),
        duplicate_row_count=int(dataframe.duplicated().sum()),
        drift_evaluated=drifted_features is not None,
        drifted_features=ordered_drifted_features,
    )


def profile_csv(
    input_path: Path,
    params_path: Path,
    report_directory: Path,
) -> DatasetProfileArtifact:
    """Validate, profile, and persist one CSV as deterministic JSON."""

    dataframe, data_version = _read_versioned_csv(input_path)
    contract = load_data_contract(params_path)
    validation = validate_dataframe(
        dataframe,
        contract,
        dataset_name=input_path.name,
    )
    if not validation.is_valid:
        issue_codes = ", ".join(issue.code for issue in validation.issues)
        message = f"Cannot profile invalid dataset '{input_path}': {issue_codes}"
        raise DataProfileError(message)

    profile = build_dataset_profile(
        dataframe,
        schema_version=contract.schema_version,
        dataset_name=input_path.name,
        data_version=data_version,
    )
    output_path = report_directory / f"{input_path.stem}.profile.json"
    _write_profile(profile, output_path)
    return DatasetProfileArtifact(profile=profile, output_path=output_path)


def render_profile_json(profile: DatasetProfile) -> str:
    """Render the stable machine-readable profile contract."""

    payload = {
        "categorical_frequencies": profile.categorical_frequencies,
        "data_version": profile.data_version,
        "dataset_name": profile.dataset_name,
        "drift_evaluated": profile.drift_evaluated,
        "drifted_features": list(profile.drifted_features),
        "duplicate_row_count": profile.duplicate_row_count,
        "feature_count": profile.feature_count,
        "feature_names": list(profile.feature_names),
        "feature_types": profile.feature_types,
        "missing_value_counts": profile.missing_value_counts,
        "numerical_summaries": {
            feature_name: {
                "maximum": summary.maximum,
                "mean": summary.mean,
                "median": summary.median,
                "minimum": summary.minimum,
                "standard_deviation": summary.standard_deviation,
            }
            for feature_name, summary in profile.numerical_summaries.items()
        },
        "row_count": profile.row_count,
        "schema_version": profile.schema_version,
        "target_distribution": profile.target_distribution,
    }
    return f"{json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True)}\n"


def main(arguments: Sequence[str] | None = None) -> int:
    """Profile a trusted CSV and return a pipeline-friendly exit code."""

    parsed_arguments = _build_argument_parser().parse_args(arguments)
    try:
        artifact = profile_csv(
            parsed_arguments.input,
            parsed_arguments.params,
            parsed_arguments.report_dir,
        )
    except (DataContractConfigurationError, DataProfileError) as error:
        LOGGER.error("profile_error reason=%s", error)
        return 2

    LOGGER.info(
        "profile_created dataset=%s rows=%d features=%d data_version=%s path=%s",
        artifact.profile.dataset_name,
        artifact.profile.row_count,
        artifact.profile.feature_count,
        artifact.profile.data_version,
        artifact.output_path,
    )
    return 0


def configure_logging() -> None:
    """Configure concise default logging for direct CLI execution."""

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create an aggregate JSON profile for trusted customer data.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/curated/training.csv"),
        help="Trusted CSV to profile (default: data/curated/training.csv).",
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
        default=Path("reports/data-profile"),
        help="Profile report directory (default: reports/data-profile).",
    )
    return parser


def _summarize_numeric(series: pd.Series) -> NumericSummary:
    return NumericSummary(
        minimum=float(series.min()),
        maximum=float(series.max()),
        mean=float(series.mean()),
        median=float(series.median()),
        standard_deviation=float(series.std(ddof=0)),
    )


def _value_counts(series: pd.Series) -> dict[str, int]:
    counts = series.value_counts(dropna=False)
    return {
        str(value): int(count)
        for value, count in sorted(
            counts.items(),
            key=lambda item: str(item[0]),
        )
    }


def _read_versioned_csv(input_path: Path) -> tuple[pd.DataFrame, str]:
    try:
        content = input_path.read_bytes()
        dataframe = pd.read_csv(input_path)
    except (
        OSError,
        UnicodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as error:
        message = f"Cannot read dataset for profiling '{input_path}': {error}"
        raise DataProfileError(message) from error
    return dataframe, hashlib.sha256(content).hexdigest()


def _write_profile(profile: DatasetProfile, output_path: Path) -> None:
    temporary_path = output_path.with_suffix(".json.tmp")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path.write_text(render_profile_json(profile), encoding="utf-8")
        temporary_path.replace(output_path)
    except OSError as error:
        temporary_path.unlink(missing_ok=True)
        message = f"Cannot write dataset profile to '{output_path}': {error}"
        raise DataProfileError(message) from error


if __name__ == "__main__":
    configure_logging()
    raise SystemExit(main())
