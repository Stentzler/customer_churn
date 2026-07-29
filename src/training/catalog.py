"""Small allowlisted catalog for deterministic baseline estimators.

The catalog owns estimator construction so neither an LLM response nor orchestration
code can import an arbitrary class. Every call returns fresh, unfitted estimators.
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from src.training.settings import ModelName, TrainingSettings

type AllowedEstimator = LogisticRegression | RandomForestClassifier


@dataclass(frozen=True, slots=True)
class CandidateEstimator:
    """Named unfitted estimator approved for one candidate experiment."""

    name: ModelName
    estimator: AllowedEstimator


def create_fallback_candidates(
    settings: TrainingSettings,
) -> tuple[CandidateEstimator, ...]:
    """Create the deterministic two-model plan used without an agent."""

    return tuple(
        create_candidate(model_name, settings)
        for model_name in (
            ModelName.LOGISTIC_REGRESSION,
            ModelName.RANDOM_FOREST,
        )
    )


def create_candidate(
    model_name: ModelName,
    settings: TrainingSettings,
) -> CandidateEstimator:
    """Construct one fresh estimator from validated versioned policy."""

    if model_name is ModelName.LOGISTIC_REGRESSION:
        model_settings = settings.logistic_regression
        estimator: AllowedEstimator = LogisticRegression(
            C=model_settings.regularization_strength,
            max_iter=model_settings.maximum_iterations,
            class_weight="balanced",
            random_state=settings.random_seed,
        )
    elif model_name is ModelName.RANDOM_FOREST:
        model_settings = settings.random_forest
        estimator = RandomForestClassifier(
            n_estimators=model_settings.estimator_count,
            max_depth=model_settings.maximum_depth,
            min_samples_leaf=model_settings.minimum_samples_per_leaf,
            class_weight="balanced",
            random_state=settings.random_seed,
            n_jobs=1,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return CandidateEstimator(name=model_name, estimator=estimator)
