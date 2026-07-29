"""Trusted dataset loading and leakage-safe train/validation splitting.

The CLI runs the deterministic fallback candidates locally. MLflow tracking and
registry operations are intentionally separate concerns added in the next phase.
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from src.agent.plan_validator import (
    ExperimentPlanValidationError,
    load_and_validate_experiment_plan,
)
from src.agent.schemas import ExperimentPlan
from src.data.schema import (
    CUSTOMER_CHURN_COLUMNS,
    CUSTOMER_IDENTIFIER_COLUMN,
    CUSTOMER_TARGET_COLUMN,
)
from src.training.artifacts import TrainingArtifactError, persist_training_run
from src.training.catalog import (
    CandidateEstimator,
    create_candidate,
    create_fallback_candidates,
)
from src.training.evaluate import (
    CandidateEvaluation,
    CandidateMetrics,
    calculate_candidate_metrics,
    select_best_candidate,
)
from src.training.preprocessing import MODEL_FEATURE_COLUMNS, create_model_pipeline
from src.training.settings import (
    ModelName,
    TrainingConfigurationError,
    TrainingSettings,
    load_training_settings,
)

LOGGER = logging.getLogger(__name__)
LOG_FORMAT = "%(levelname)s %(name)s %(message)s"
DEFAULT_CURATED_PATH = Path("data/curated/training.csv")
DEFAULT_PARAMS_PATH = Path("params.yaml")
DEFAULT_MODEL_DIRECTORY = Path("artifacts/models")
DEFAULT_METRICS_DIRECTORY = Path("artifacts/metrics")
DEFAULT_PLAN_PATH = Path("artifacts/experiment-plans/fallback.json")


class TrainingDataError(ValueError):
    """Raised when curated data cannot safely be used for candidate training."""


class CandidateTrainingError(RuntimeError):
    """Raised when a deterministic candidate cannot be fitted or evaluated."""


@dataclass(frozen=True, slots=True)
class TrainingDatasetSplit:
    """Feature and target partitions used for candidate selection."""

    training_features: pd.DataFrame
    validation_features: pd.DataFrame
    training_target: pd.Series
    validation_target: pd.Series


@dataclass(frozen=True, slots=True)
class TrainedCandidate:
    """A fitted serving pipeline and its verified validation result."""

    model_name: ModelName
    pipeline: Pipeline
    metrics: CandidateMetrics
    training_seconds: float

    @property
    def evaluation(self) -> CandidateEvaluation:
        """Expose only the immutable information needed for model selection."""

        return CandidateEvaluation(model_name=self.model_name, metrics=self.metrics)


@dataclass(frozen=True, slots=True)
class FailedCandidate:
    """Safe failure summary for one candidate without traceback or data."""

    model_name: ModelName
    reason: str


@dataclass(frozen=True, slots=True)
class CandidateTrainingBatch:
    """Independent successful and failed candidate outcomes."""

    successful: tuple[TrainedCandidate, ...]
    failed: tuple[FailedCandidate, ...]


@dataclass(frozen=True, slots=True)
class TrainingRun:
    """All successful candidates and the deterministically selected winner."""

    candidates: tuple[TrainedCandidate, ...]
    failed_candidates: tuple[FailedCandidate, ...]
    selected: TrainedCandidate


def load_and_split_training_data(
    curated_path: Path,
    settings: TrainingSettings,
) -> TrainingDatasetSplit:
    """Load trusted curated CSV data and create deterministic partitions."""

    if curated_path.parent.name != "curated":
        message = (
            "Training data must come from a directory named 'curated'; "
            "fixed-test and incoming datasets are not training sources"
        )
        raise TrainingDataError(message)
    dataframe = _read_curated_csv(curated_path)
    return split_training_data(dataframe, settings)


def run_training(
    curated_path: Path,
    settings: TrainingSettings,
) -> TrainingRun:
    """Load curated data, train fallback candidates, and select the best one."""

    split = load_and_split_training_data(curated_path, settings)
    batch = train_candidates(
        split,
        create_fallback_candidates(settings),
    )
    return _select_training_run(
        batch,
        settings.primary_metric,
        settings.minimum_successful_candidates,
    )


def run_training_from_plan(
    curated_path: Path,
    settings: TrainingSettings,
    approved_plan: ExperimentPlan,
) -> TrainingRun:
    """Train candidates constructed only from an approved experiment plan."""

    candidates = tuple(
        create_candidate(
            experiment.algorithm,
            settings,
            experiment.parameters,
        )
        for experiment in approved_plan.experiments
    )
    split = load_and_split_training_data(curated_path, settings)
    batch = train_candidates(split, candidates)
    return _select_training_run(
        batch,
        approved_plan.primary_metric,
        settings.minimum_successful_candidates,
    )


def _select_training_run(
    batch: CandidateTrainingBatch,
    primary_metric: str,
    minimum_successful_candidates: int,
) -> TrainingRun:
    if len(batch.successful) < minimum_successful_candidates:
        message = (
            "Successful candidate count "
            f"{len(batch.successful)} is below required minimum "
            f"{minimum_successful_candidates}"
        )
        raise CandidateTrainingError(message)
    selected_evaluation = select_best_candidate(
        tuple(candidate.evaluation for candidate in batch.successful),
        primary_metric,
    )
    selected = next(
        candidate
        for candidate in batch.successful
        if candidate.model_name is selected_evaluation.model_name
    )
    return TrainingRun(
        candidates=batch.successful,
        failed_candidates=batch.failed,
        selected=selected,
    )


def train_candidates(
    split: TrainingDatasetSplit,
    candidates: tuple[CandidateEstimator, ...],
) -> CandidateTrainingBatch:
    """Fit and evaluate every approved candidate independently."""

    if not candidates:
        raise CandidateTrainingError("At least one candidate is required for training")

    successful: list[TrainedCandidate] = []
    failed: list[FailedCandidate] = []
    for candidate in candidates:
        try:
            successful.append(_train_candidate(split, candidate))
        except CandidateTrainingError as error:
            failed.append(FailedCandidate(model_name=candidate.name, reason=str(error)))
    return CandidateTrainingBatch(
        successful=tuple(successful),
        failed=tuple(failed),
    )


def _train_candidate(
    split: TrainingDatasetSplit,
    candidate: CandidateEstimator,
) -> TrainedCandidate:
    pipeline = create_model_pipeline(candidate)
    started_at = time.perf_counter()
    try:
        # Fitting the complete pipeline on this partition is the central leakage
        # safeguard: scalers and encoders never observe validation values.
        pipeline.fit(split.training_features, split.training_target)
        training_seconds = time.perf_counter() - started_at
        predicted = pd.Series(
            pipeline.predict(split.validation_features),
            index=split.validation_target.index,
        )
        positive_probability = pd.Series(
            pipeline.predict_proba(split.validation_features)[:, 1],
            index=split.validation_target.index,
        )
        metrics = calculate_candidate_metrics(
            split.validation_target,
            predicted,
            positive_probability,
        )
    except Exception as error:
        message = f"Candidate '{candidate.name.value}' failed during training"
        raise CandidateTrainingError(message) from error

    return TrainedCandidate(
        model_name=candidate.name,
        pipeline=pipeline,
        metrics=metrics,
        training_seconds=training_seconds,
    )


def split_training_data(
    dataframe: pd.DataFrame,
    settings: TrainingSettings,
) -> TrainingDatasetSplit:
    """Split validated labeled data before any preprocessing is fitted.

    Stratification preserves the target-class proportion in both partitions. The
    customer identifier is deliberately excluded because it is an identity, not a
    behavior that should influence churn predictions.
    """

    _validate_training_dataframe(dataframe)
    features = dataframe.loc[:, MODEL_FEATURE_COLUMNS]
    target = dataframe.loc[:, CUSTOMER_TARGET_COLUMN]

    try:
        (
            training_features,
            validation_features,
            training_target,
            validation_target,
        ) = train_test_split(
            features,
            target,
            test_size=settings.validation_fraction,
            random_state=settings.random_seed,
            shuffle=True,
            stratify=target,
        )
    except ValueError as error:
        message = f"Cannot create a stratified training split: {error}"
        raise TrainingDataError(message) from error

    return TrainingDatasetSplit(
        training_features=training_features.reset_index(drop=True),
        validation_features=validation_features.reset_index(drop=True),
        training_target=training_target.reset_index(drop=True),
        validation_target=validation_target.reset_index(drop=True),
    )


def _read_curated_csv(curated_path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(curated_path)
    except FileNotFoundError as error:
        message = f"Curated training dataset does not exist: {curated_path}"
        raise TrainingDataError(message) from error
    except (OSError, pd.errors.ParserError) as error:
        message = f"Cannot read curated training dataset '{curated_path}': {error}"
        raise TrainingDataError(message) from error


def _validate_training_dataframe(dataframe: pd.DataFrame) -> None:
    actual_columns = tuple(dataframe.columns)
    if actual_columns != CUSTOMER_CHURN_COLUMNS:
        message = (
            "Curated training columns must exactly match the data contract; "
            f"expected {CUSTOMER_CHURN_COLUMNS}, received {actual_columns}"
        )
        raise TrainingDataError(message)
    if dataframe.empty:
        raise TrainingDataError("Curated training dataset must not be empty")
    if dataframe.isna().to_numpy().any():
        raise TrainingDataError("Curated training dataset must not contain null values")

    target_values = set(dataframe[CUSTOMER_TARGET_COLUMN].unique())
    if target_values != {0, 1}:
        message = (
            f"{CUSTOMER_TARGET_COLUMN} must contain both binary classes 0 and 1; "
            f"received {sorted(target_values)}"
        )
        raise TrainingDataError(message)

    # This assertion documents the safety rule even if the data schema changes.
    if CUSTOMER_IDENTIFIER_COLUMN in MODEL_FEATURE_COLUMNS:
        raise RuntimeError("Customer identifiers cannot be model features")


def configure_logging() -> None:
    """Configure concise default logging for direct CLI execution."""

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse local deterministic-training command arguments."""

    parser = argparse.ArgumentParser(
        description="Train and compare deterministic customer-churn baselines."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_CURATED_PATH,
        help="Curated labeled CSV (default: data/curated/training.csv).",
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=DEFAULT_PARAMS_PATH,
        help="Versioned training parameters (default: params.yaml).",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=DEFAULT_PLAN_PATH,
        help="Approved experiment plan (default: fallback plan artifact).",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIRECTORY,
        help="Candidate pipeline directory (default: artifacts/models).",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=DEFAULT_METRICS_DIRECTORY,
        help="Candidate metrics directory (default: artifacts/metrics).",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """Run local baseline training and report verified candidate metrics."""

    args = parse_args(arguments)
    configure_logging()
    try:
        settings = load_training_settings(args.params)
        approved_plan = load_and_validate_experiment_plan(args.plan, settings)
        training_run = run_training_from_plan(args.input, settings, approved_plan)
        artifacts = persist_training_run(
            training_run,
            args.model_dir,
            args.metrics_dir,
            settings.primary_metric,
        )
    except (
        TrainingConfigurationError,
        TrainingDataError,
        CandidateTrainingError,
        TrainingArtifactError,
        ExperimentPlanValidationError,
    ) as error:
        LOGGER.error("training_failed reason=%s", error)
        return 1

    for candidate in training_run.candidates:
        metrics = candidate.metrics
        LOGGER.info(
            (
                "candidate_trained model=%s roc_auc=%.6f pr_auc=%.6f f1=%.6f "
                "precision=%.6f recall=%.6f confusion_matrix=%s "
                "class_distribution=%s training_seconds=%.6f"
            ),
            candidate.model_name.value,
            metrics.roc_auc,
            metrics.pr_auc,
            metrics.f1,
            metrics.precision,
            metrics.recall,
            metrics.confusion_matrix,
            metrics.class_distribution,
            candidate.training_seconds,
        )
    for failure in training_run.failed_candidates:
        LOGGER.warning(
            "candidate_failed model=%s reason=%s",
            failure.model_name.value,
            failure.reason,
        )

    LOGGER.info(
        "candidate_selected model=%s primary_metric=%s value=%.6f manifest=%s",
        training_run.selected.model_name.value,
        settings.primary_metric,
        training_run.selected.metrics.value_for(settings.primary_metric),
        artifacts.selection_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
