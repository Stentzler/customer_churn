import pandas as pd
import pytest
from src.training.evaluate import (
    CandidateEvaluation,
    CandidateMetrics,
    ClassDistribution,
    calculate_candidate_metrics,
    select_best_candidate,
)
from src.training.settings import ModelName


def test_calculate_candidate_metrics_for_perfect_predictions() -> None:
    expected = pd.Series([0, 0, 1, 1])
    predicted = pd.Series([0, 0, 1, 1])
    probabilities = pd.Series([0.1, 0.2, 0.8, 0.9])

    metrics = calculate_candidate_metrics(expected, predicted, probabilities)

    assert metrics.roc_auc == pytest.approx(1.0)
    assert metrics.pr_auc == pytest.approx(1.0)
    assert metrics.f1 == pytest.approx(1.0)
    assert metrics.precision == pytest.approx(1.0)
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.confusion_matrix == ((2, 0), (0, 2))
    assert metrics.class_distribution == ClassDistribution(retained=2, churned=2)


def test_primary_metric_selects_the_best_candidate() -> None:
    logistic = _evaluation(ModelName.LOGISTIC_REGRESSION, roc_auc=0.80, f1=0.75)
    random_forest = _evaluation(ModelName.RANDOM_FOREST, roc_auc=0.85, f1=0.70)

    selected = select_best_candidate((logistic, random_forest), "f1")

    assert selected.model_name is ModelName.LOGISTIC_REGRESSION


def test_selection_uses_model_name_as_final_stable_tie_breaker() -> None:
    logistic = _evaluation(ModelName.LOGISTIC_REGRESSION, roc_auc=0.80, f1=0.70)
    random_forest = _evaluation(ModelName.RANDOM_FOREST, roc_auc=0.80, f1=0.70)

    selected = select_best_candidate((random_forest, logistic), "roc_auc")

    assert selected.model_name is ModelName.LOGISTIC_REGRESSION


def test_selection_requires_a_successful_candidate() -> None:
    with pytest.raises(ValueError, match="At least one"):
        select_best_candidate((), "roc_auc")


def _evaluation(
    model_name: ModelName,
    *,
    roc_auc: float,
    f1: float,
) -> CandidateEvaluation:
    return CandidateEvaluation(
        model_name=model_name,
        metrics=CandidateMetrics(
            roc_auc=roc_auc,
            pr_auc=0.75,
            f1=f1,
            precision=0.70,
            recall=0.70,
            confusion_matrix=((8, 2), (3, 7)),
            class_distribution=ClassDistribution(retained=10, churned=10),
        ),
    )
