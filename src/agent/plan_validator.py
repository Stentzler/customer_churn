"""Deterministic validation of untrusted structured experiment plans."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError
from src.agent.schemas import ExperimentPlan
from src.training.catalog import (
    CatalogValidationError,
    build_model_catalog,
    resolve_model_parameters,
)
from src.training.settings import TrainingSettings


class ExperimentPlanValidationError(ValueError):
    """Raised when a plan violates schema or versioned execution policy."""


def load_and_validate_experiment_plan(
    plan_path: Path,
    settings: TrainingSettings,
) -> ExperimentPlan:
    """Load JSON as untrusted input and return a normalized approved plan."""

    try:
        content = plan_path.read_text(encoding="utf-8")
        proposed = ExperimentPlan.model_validate_json(content)
    except OSError as error:
        message = f"Cannot read experiment plan '{plan_path}': {error}"
        raise ExperimentPlanValidationError(message) from error
    except ValidationError as error:
        message = f"Experiment plan '{plan_path}' violates its schema: {error}"
        raise ExperimentPlanValidationError(message) from error
    return validate_experiment_plan(proposed, settings)


def validate_experiment_plan(
    proposed: ExperimentPlan,
    settings: TrainingSettings,
) -> ExperimentPlan:
    """Enforce metric, count, uniqueness, and model-catalog policy."""

    if proposed.primary_metric != settings.primary_metric:
        message = (
            "Experiment plan cannot override configured primary metric "
            f"'{settings.primary_metric}'"
        )
        raise ExperimentPlanValidationError(message)
    if len(proposed.experiments) > settings.maximum_candidates:
        raise ExperimentPlanValidationError(
            "Experiment plan exceeds the configured candidate limit"
        )

    algorithms = [experiment.algorithm for experiment in proposed.experiments]
    if len(set(algorithms)) != len(algorithms):
        raise ExperimentPlanValidationError(
            "Experiment plan cannot contain duplicate algorithms"
        )

    catalog = build_model_catalog(settings)
    try:
        approved_experiments = [
            experiment.model_copy(
                update={
                    "parameters": resolve_model_parameters(
                        catalog[experiment.algorithm],
                        experiment.parameters,
                    )
                }
            )
            for experiment in proposed.experiments
        ]
    except (CatalogValidationError, KeyError) as error:
        message = f"Experiment plan violates model-catalog policy: {error}"
        raise ExperimentPlanValidationError(message) from error

    return proposed.model_copy(update={"experiments": tuple(approved_experiments)})
