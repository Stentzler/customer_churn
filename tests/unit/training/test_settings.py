from pathlib import Path

import pytest
import yaml
from src.training.settings import (
    LogisticRegressionSettings,
    RandomForestSettings,
    TrainingConfigurationError,
    TrainingSettings,
    load_training_settings,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


def test_load_training_settings_from_versioned_params() -> None:
    assert load_training_settings(PARAMS_PATH) == TrainingSettings(
        random_seed=42,
        validation_fraction=0.20,
        maximum_candidates=3,
        minimum_successful_candidates=1,
        primary_metric="roc_auc",
        logistic_regression=LogisticRegressionSettings(
            regularization_strength=1.0,
            maximum_iterations=1000,
        ),
        random_forest=RandomForestSettings(
            estimator_count=200,
            maximum_depth=8,
            minimum_samples_per_leaf=2,
        ),
    )


@pytest.mark.parametrize("validation_fraction", [0, 1, -0.1, 1.1])
def test_invalid_validation_fraction_is_rejected(
    tmp_path: Path,
    validation_fraction: float,
) -> None:
    params = _load_versioned_params()
    training = _get_mapping(params, "training")
    training["validation_fraction"] = validation_fraction

    with pytest.raises(TrainingConfigurationError, match="must be greater than 0"):
        load_training_settings(_write_params(tmp_path, params))


def test_unknown_model_is_rejected(tmp_path: Path) -> None:
    params = _load_versioned_params()
    training = _get_mapping(params, "training")
    models = _get_mapping(training, "models")
    models["xgboost"] = {}

    with pytest.raises(
        TrainingConfigurationError,
        match="unexpected keys: xgboost",
    ):
        load_training_settings(_write_params(tmp_path, params))


def test_candidate_limit_must_include_the_fallback_models(tmp_path: Path) -> None:
    params = _load_versioned_params()
    experiments = _get_mapping(params, "experiments")
    experiments["maximum_candidates"] = 1

    with pytest.raises(
        TrainingConfigurationError,
        match="must allow every fallback candidate",
    ):
        load_training_settings(_write_params(tmp_path, params))


def test_unsupported_primary_metric_is_rejected(tmp_path: Path) -> None:
    params = _load_versioned_params()
    experiments = _get_mapping(params, "experiments")
    experiments["primary_metric"] = "accuracy"

    with pytest.raises(
        TrainingConfigurationError,
        match="primary_metric must be one of",
    ):
        load_training_settings(_write_params(tmp_path, params))


def test_minimum_successful_candidates_cannot_exceed_limit(tmp_path: Path) -> None:
    params = _load_versioned_params()
    experiments = _get_mapping(params, "experiments")
    experiments["minimum_successful_candidates"] = 4

    with pytest.raises(
        TrainingConfigurationError,
        match="cannot exceed",
    ):
        load_training_settings(_write_params(tmp_path, params))


def _load_versioned_params() -> dict[str, object]:
    parsed = yaml.safe_load(PARAMS_PATH.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict)
    return parsed


def _get_mapping(mapping: dict[str, object], key: str) -> dict[str, object]:
    value = mapping[key]
    assert isinstance(value, dict)
    return value


def _write_params(tmp_path: Path, params: object) -> Path:
    params_path = tmp_path / "params.yaml"
    params_path.write_text(
        yaml.safe_dump(params, sort_keys=False),
        encoding="utf-8",
    )
    return params_path
