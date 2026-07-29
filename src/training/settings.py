"""Typed configuration for deterministic baseline model training.

Training configuration is loaded separately from DataOps configuration because the
two subsystems have different responsibilities and failure boundaries. YAML remains
an untrusted input until every required field and policy constraint is validated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

import yaml

TRAINING_KEYS = frozenset({"validation_fraction", "models"})
EXPERIMENT_KEYS = frozenset({"maximum_candidates", "primary_metric"})
MODEL_NAMES = frozenset({"logistic_regression", "random_forest"})
LOGISTIC_REGRESSION_KEYS = frozenset({"C", "max_iter"})
RANDOM_FOREST_KEYS = frozenset(
    {
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
    }
)
SUPPORTED_PRIMARY_METRICS = frozenset(
    {
        "roc_auc",
        "pr_auc",
        "f1",
        "precision",
        "recall",
    }
)

type StringMapping = dict[str, object]


class ModelName(StrEnum):
    """Allowlisted algorithms available to deterministic training."""

    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"


class TrainingConfigurationError(ValueError):
    """Raised when versioned training policy is missing or unsafe."""


@dataclass(frozen=True, slots=True)
class LogisticRegressionSettings:
    """Fixed baseline parameters for logistic regression."""

    regularization_strength: float
    maximum_iterations: int


@dataclass(frozen=True, slots=True)
class RandomForestSettings:
    """Fixed baseline parameters for the random-forest candidate."""

    estimator_count: int
    maximum_depth: int
    minimum_samples_per_leaf: int


@dataclass(frozen=True, slots=True)
class TrainingSettings:
    """Validated policy shared by splitting, training, and selection."""

    random_seed: int
    validation_fraction: float
    maximum_candidates: int
    primary_metric: str
    logistic_regression: LogisticRegressionSettings
    random_forest: RandomForestSettings


def load_training_settings(params_path: Path) -> TrainingSettings:
    """Load and validate minimal training policy from ``params.yaml``."""

    root = _load_yaml_mapping(params_path)
    project = _require_mapping(root.get("project"), "project")
    experiments = _require_mapping(root.get("experiments"), "experiments")
    training = _require_mapping(root.get("training"), "training")
    models = _require_mapping(training.get("models"), "training.models")

    _require_exact_keys(experiments, EXPERIMENT_KEYS, "experiments")
    _require_exact_keys(training, TRAINING_KEYS, "training")
    _require_exact_keys(models, MODEL_NAMES, "training.models")

    maximum_candidates = _require_positive_integer(
        experiments.get("maximum_candidates"),
        "experiments.maximum_candidates",
    )
    if maximum_candidates < len(ModelName):
        message = (
            "experiments.maximum_candidates must allow every fallback candidate "
            f"({len(ModelName)} required)"
        )
        raise TrainingConfigurationError(message)

    return TrainingSettings(
        random_seed=_require_non_negative_integer(
            project.get("random_seed"),
            "project.random_seed",
        ),
        validation_fraction=_require_fraction(
            training.get("validation_fraction"),
            "training.validation_fraction",
        ),
        maximum_candidates=maximum_candidates,
        primary_metric=_require_supported_metric(
            experiments.get("primary_metric"),
            "experiments.primary_metric",
        ),
        logistic_regression=_parse_logistic_regression(
            models.get(ModelName.LOGISTIC_REGRESSION.value)
        ),
        random_forest=_parse_random_forest(models.get(ModelName.RANDOM_FOREST.value)),
    )


def _parse_logistic_regression(value: object) -> LogisticRegressionSettings:
    location = "training.models.logistic_regression"
    parameters = _require_mapping(value, location)
    _require_exact_keys(parameters, LOGISTIC_REGRESSION_KEYS, location)
    regularization_strength = _require_positive_number(
        parameters.get("C"),
        f"{location}.C",
    )
    return LogisticRegressionSettings(
        regularization_strength=regularization_strength,
        maximum_iterations=_require_positive_integer(
            parameters.get("max_iter"),
            f"{location}.max_iter",
        ),
    )


def _parse_random_forest(value: object) -> RandomForestSettings:
    location = "training.models.random_forest"
    parameters = _require_mapping(value, location)
    _require_exact_keys(parameters, RANDOM_FOREST_KEYS, location)
    return RandomForestSettings(
        estimator_count=_require_positive_integer(
            parameters.get("n_estimators"),
            f"{location}.n_estimators",
        ),
        maximum_depth=_require_positive_integer(
            parameters.get("max_depth"),
            f"{location}.max_depth",
        ),
        minimum_samples_per_leaf=_require_positive_integer(
            parameters.get("min_samples_leaf"),
            f"{location}.min_samples_leaf",
        ),
    )


def _load_yaml_mapping(params_path: Path) -> StringMapping:
    try:
        content = params_path.read_text(encoding="utf-8")
        parsed = yaml.safe_load(content)
    except OSError as error:
        message = f"Cannot read parameters file '{params_path}': {error}"
        raise TrainingConfigurationError(message) from error
    except yaml.YAMLError as error:
        message = f"Parameters file '{params_path}' contains invalid YAML: {error}"
        raise TrainingConfigurationError(message) from error
    return _require_mapping(parsed, "configuration root")


def _require_mapping(value: object, location: str) -> StringMapping:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        message = f"{location} must be a mapping with string keys"
        raise TrainingConfigurationError(message)
    return cast(StringMapping, value)


def _require_exact_keys(
    mapping: StringMapping,
    expected_keys: frozenset[str],
    location: str,
) -> None:
    actual_keys = set(mapping)
    missing_keys = sorted(expected_keys - actual_keys)
    unexpected_keys = sorted(actual_keys - expected_keys)
    if not missing_keys and not unexpected_keys:
        return

    details: list[str] = []
    if missing_keys:
        details.append(f"missing keys: {', '.join(missing_keys)}")
    if unexpected_keys:
        details.append(f"unexpected keys: {', '.join(unexpected_keys)}")
    message = f"{location} has invalid fields ({'; '.join(details)})"
    raise TrainingConfigurationError(message)


def _require_positive_integer(value: object, location: str) -> int:
    if type(value) is not int or cast(int, value) < 1:
        message = f"{location} must be a positive integer"
        raise TrainingConfigurationError(message)
    return cast(int, value)


def _require_non_negative_integer(value: object, location: str) -> int:
    if type(value) is not int or cast(int, value) < 0:
        message = f"{location} must be a non-negative integer"
        raise TrainingConfigurationError(message)
    return cast(int, value)


def _require_positive_number(value: object, location: str) -> float:
    if (
        type(value) not in {int, float}
        or not math.isfinite(cast(float, value))
        or cast(float, value) <= 0
    ):
        message = f"{location} must be a positive finite number"
        raise TrainingConfigurationError(message)
    return float(cast(float, value))


def _require_fraction(value: object, location: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(cast(float, value)):
        message = f"{location} must be a finite number between 0 and 1"
        raise TrainingConfigurationError(message)
    fraction = float(cast(float, value))
    if not 0 < fraction < 1:
        message = f"{location} must be greater than 0 and less than 1"
        raise TrainingConfigurationError(message)
    return fraction


def _require_supported_metric(value: object, location: str) -> str:
    if not isinstance(value, str) or value not in SUPPORTED_PRIMARY_METRICS:
        supported = ", ".join(sorted(SUPPORTED_PRIMARY_METRICS))
        message = f"{location} must be one of: {supported}"
        raise TrainingConfigurationError(message)
    return value
