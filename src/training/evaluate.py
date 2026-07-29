"""Pure candidate metric calculation and deterministic selection."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from src.training.settings import ModelName


@dataclass(frozen=True, slots=True)
class ClassDistribution:
    """Counts for the negative and positive classes in an evaluation partition."""

    retained: int
    churned: int


@dataclass(frozen=True, slots=True)
class CandidateMetrics:
    """Required binary-classification metrics for one candidate."""

    roc_auc: float
    pr_auc: float
    f1: float
    precision: float
    recall: float
    confusion_matrix: tuple[tuple[int, int], tuple[int, int]]
    class_distribution: ClassDistribution

    def value_for(self, metric_name: str) -> float:
        """Return one supported scalar metric by its configured identifier."""

        metric_values = {
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
        }
        try:
            return metric_values[metric_name]
        except KeyError as error:
            message = f"Unsupported candidate-selection metric: {metric_name}"
            raise ValueError(message) from error


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    """Model identity and verified validation metrics used for selection."""

    model_name: ModelName
    metrics: CandidateMetrics


def calculate_candidate_metrics(
    expected: pd.Series,
    predicted: pd.Series,
    positive_probability: pd.Series,
) -> CandidateMetrics:
    """Calculate the complete deterministic validation metric set."""

    matrix = confusion_matrix(expected, predicted, labels=[0, 1])
    class_counts = expected.value_counts()
    return CandidateMetrics(
        roc_auc=float(roc_auc_score(expected, positive_probability)),
        pr_auc=float(average_precision_score(expected, positive_probability)),
        f1=float(f1_score(expected, predicted, zero_division=0)),
        precision=float(precision_score(expected, predicted, zero_division=0)),
        recall=float(recall_score(expected, predicted, zero_division=0)),
        confusion_matrix=(
            (int(matrix[0, 0]), int(matrix[0, 1])),
            (int(matrix[1, 0]), int(matrix[1, 1])),
        ),
        class_distribution=ClassDistribution(
            retained=int(class_counts.get(0, 0)),
            churned=int(class_counts.get(1, 0)),
        ),
    )


def select_best_candidate(
    evaluations: tuple[CandidateEvaluation, ...],
    primary_metric: str,
) -> CandidateEvaluation:
    """Select the winner with stable, documented descending tie-breakers."""

    if not evaluations:
        raise ValueError("At least one successful candidate is required")

    return sorted(
        evaluations,
        key=lambda evaluation: (
            -evaluation.metrics.value_for(primary_metric),
            -evaluation.metrics.roc_auc,
            -evaluation.metrics.f1,
            -evaluation.metrics.recall,
            evaluation.model_name.value,
        ),
    )[0]
