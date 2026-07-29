"""Small allowlisted catalog for deterministic baseline estimators.

The catalog owns estimator construction so neither an LLM response nor orchestration
code can import an arbitrary class. Every call returns fresh, unfitted estimators.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import cast

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from src.training.settings import ModelName, TrainingSettings

type AllowedEstimator = LogisticRegression | RandomForestClassifier
type EstimatorConstructor = type[LogisticRegression] | type[RandomForestClassifier]


class ParameterType(StrEnum):
    """Primitive parameter types accepted from an experiment plan."""

    FLOAT = "float"
    INTEGER = "integer"


class CatalogValidationError(ValueError):
    """Raised when proposed model parameters violate the catalog contract."""


@dataclass(frozen=True, slots=True)
class ParameterDefinition:
    """Allowlisted parameter type, inclusive bounds, and versioned default."""

    parameter_type: ParameterType
    minimum: int | float
    maximum: int | float
    default: int | float


@dataclass(frozen=True, slots=True)
class ModelDefinition:
    """Complete immutable safety contract for one allowed estimator."""

    name: ModelName
    constructor: EstimatorConstructor
    parameters: Mapping[str, ParameterDefinition]


@dataclass(frozen=True, slots=True)
class CandidateEstimator:
    """Named unfitted estimator approved for one candidate experiment."""

    name: ModelName
    estimator: AllowedEstimator


def build_model_catalog(
    settings: TrainingSettings,
) -> Mapping[ModelName, ModelDefinition]:
    """Build the allowlist using versioned defaults and code-owned safety bounds."""

    logistic = settings.logistic_regression
    forest = settings.random_forest
    definitions = {
        ModelName.LOGISTIC_REGRESSION: ModelDefinition(
            name=ModelName.LOGISTIC_REGRESSION,
            constructor=LogisticRegression,
            parameters=MappingProxyType(
                {
                    "C": ParameterDefinition(
                        ParameterType.FLOAT,
                        minimum=0.01,
                        maximum=10.0,
                        default=logistic.regularization_strength,
                    ),
                    "max_iter": ParameterDefinition(
                        ParameterType.INTEGER,
                        minimum=100,
                        maximum=5000,
                        default=logistic.maximum_iterations,
                    ),
                }
            ),
        ),
        ModelName.RANDOM_FOREST: ModelDefinition(
            name=ModelName.RANDOM_FOREST,
            constructor=RandomForestClassifier,
            parameters=MappingProxyType(
                {
                    "n_estimators": ParameterDefinition(
                        ParameterType.INTEGER,
                        minimum=10,
                        maximum=500,
                        default=forest.estimator_count,
                    ),
                    "max_depth": ParameterDefinition(
                        ParameterType.INTEGER,
                        minimum=1,
                        maximum=50,
                        default=forest.maximum_depth,
                    ),
                    "min_samples_leaf": ParameterDefinition(
                        ParameterType.INTEGER,
                        minimum=1,
                        maximum=100,
                        default=forest.minimum_samples_per_leaf,
                    ),
                }
            ),
        ),
    }
    catalog: Mapping[ModelName, ModelDefinition] = MappingProxyType(definitions)
    for definition in catalog.values():
        resolve_model_parameters(definition, {})
    return catalog


def resolve_model_parameters(
    definition: ModelDefinition,
    proposed_parameters: Mapping[str, object],
) -> dict[str, int | float]:
    """Validate proposed values and fill omitted parameters with safe defaults."""

    unknown = sorted(set(proposed_parameters) - set(definition.parameters))
    if unknown:
        message = (
            f"{definition.name.value} contains unsupported parameters: "
            f"{', '.join(unknown)}"
        )
        raise CatalogValidationError(message)

    return {
        name: _validate_parameter(
            definition.name,
            name,
            proposed_parameters.get(name, parameter.default),
            parameter,
        )
        for name, parameter in definition.parameters.items()
    }


def _validate_parameter(
    model_name: ModelName,
    parameter_name: str,
    value: object,
    definition: ParameterDefinition,
) -> int | float:
    location = f"{model_name.value}.{parameter_name}"
    if definition.parameter_type is ParameterType.INTEGER:
        if type(value) is not int:
            raise CatalogValidationError(f"{location} must be an integer")
        validated: int | float = value
    else:
        if type(value) not in {int, float}:
            raise CatalogValidationError(f"{location} must be a finite number")
        numeric_value = cast(int | float, value)
        if not math.isfinite(float(numeric_value)):
            raise CatalogValidationError(f"{location} must be a finite number")
        validated = float(numeric_value)

    if not definition.minimum <= validated <= definition.maximum:
        message = (
            f"{location} must be between {definition.minimum} and {definition.maximum}"
        )
        raise CatalogValidationError(message)
    return validated


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
    proposed_parameters: Mapping[str, object] | None = None,
) -> CandidateEstimator:
    """Construct one fresh estimator from validated catalog parameters."""

    definition = build_model_catalog(settings)[model_name]
    parameters = resolve_model_parameters(definition, proposed_parameters or {})
    if model_name is ModelName.LOGISTIC_REGRESSION:
        estimator: AllowedEstimator = LogisticRegression(
            C=parameters["C"],
            max_iter=int(parameters["max_iter"]),
            class_weight="balanced",
            random_state=settings.random_seed,
        )
    elif model_name is ModelName.RANDOM_FOREST:
        estimator = RandomForestClassifier(
            n_estimators=int(parameters["n_estimators"]),
            max_depth=int(parameters["max_depth"]),
            min_samples_leaf=int(parameters["min_samples_leaf"]),
            class_weight="balanced",
            random_state=settings.random_seed,
            n_jobs=1,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return CandidateEstimator(name=model_name, estimator=estimator)
