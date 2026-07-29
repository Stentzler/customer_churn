from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from src.training.catalog import create_fallback_candidates
from src.training.settings import ModelName, load_training_settings

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


def test_fallback_catalog_creates_two_allowlisted_estimators() -> None:
    candidates = create_fallback_candidates(load_training_settings(PARAMS_PATH))

    assert tuple(candidate.name for candidate in candidates) == (
        ModelName.LOGISTIC_REGRESSION,
        ModelName.RANDOM_FOREST,
    )
    assert isinstance(candidates[0].estimator, LogisticRegression)
    assert isinstance(candidates[1].estimator, RandomForestClassifier)


def test_catalog_applies_deterministic_and_cpu_bounded_policy() -> None:
    logistic, random_forest = create_fallback_candidates(
        load_training_settings(PARAMS_PATH)
    )

    assert logistic.estimator.get_params()["random_state"] == 42
    assert logistic.estimator.get_params()["class_weight"] == "balanced"
    assert logistic.estimator.get_params()["C"] == 1.0
    assert random_forest.estimator.get_params()["random_state"] == 42
    assert random_forest.estimator.get_params()["n_jobs"] == 1
    assert random_forest.estimator.get_params()["n_estimators"] == 200


def test_catalog_returns_fresh_unfitted_estimators() -> None:
    settings = load_training_settings(PARAMS_PATH)

    first_candidates = create_fallback_candidates(settings)
    second_candidates = create_fallback_candidates(settings)

    assert first_candidates[0].estimator is not second_candidates[0].estimator
    assert first_candidates[1].estimator is not second_candidates[1].estimator
    assert not hasattr(first_candidates[0].estimator, "coef_")
    assert not hasattr(first_candidates[1].estimator, "estimators_")
