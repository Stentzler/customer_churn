from pathlib import Path

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from src.training.catalog import (
    CatalogValidationError,
    ParameterType,
    build_model_catalog,
    create_fallback_candidates,
    resolve_model_parameters,
)
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


def test_catalog_exposes_constructor_types_bounds_and_versioned_defaults() -> None:
    catalog = build_model_catalog(load_training_settings(PARAMS_PATH))

    logistic = catalog[ModelName.LOGISTIC_REGRESSION]
    forest = catalog[ModelName.RANDOM_FOREST]

    assert logistic.constructor is LogisticRegression
    assert logistic.parameters["C"].parameter_type is ParameterType.FLOAT
    assert logistic.parameters["C"].minimum == 0.01
    assert logistic.parameters["C"].maximum == 10.0
    assert logistic.parameters["C"].default == 1.0
    assert forest.constructor is RandomForestClassifier
    assert forest.parameters["n_estimators"].default == 200


def test_parameter_validation_fills_defaults_and_accepts_bounded_values() -> None:
    definition = build_model_catalog(load_training_settings(PARAMS_PATH))[
        ModelName.RANDOM_FOREST
    ]

    resolved = resolve_model_parameters(definition, {"n_estimators": 300})

    assert resolved == {
        "n_estimators": 300,
        "max_depth": 8,
        "min_samples_leaf": 2,
    }


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"unknown": 1}, "unsupported parameters"),
        ({"n_estimators": "200"}, "must be an integer"),
        ({"n_estimators": True}, "must be an integer"),
        ({"n_estimators": 501}, "must be between"),
    ],
)
def test_parameter_validation_rejects_unsafe_proposals(
    parameters: dict[str, object],
    message: str,
) -> None:
    definition = build_model_catalog(load_training_settings(PARAMS_PATH))[
        ModelName.RANDOM_FOREST
    ]

    with pytest.raises(CatalogValidationError, match=message):
        resolve_model_parameters(definition, parameters)
