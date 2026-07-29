"""Deterministic fallback experiment-plan generation and persistence."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from src.agent.schemas import ExperimentPlan, PlannedExperiment, PlanSource
from src.training.catalog import build_model_catalog, resolve_model_parameters
from src.training.settings import (
    ModelName,
    TrainingConfigurationError,
    TrainingSettings,
    load_training_settings,
)

LOGGER = logging.getLogger(__name__)
LOG_FORMAT = "%(levelname)s %(name)s %(message)s"
DEFAULT_PARAMS_PATH = Path("params.yaml")
DEFAULT_OUTPUT_PATH = Path("artifacts/experiment-plans/fallback.json")


class ExperimentPlanError(RuntimeError):
    """Raised when an experiment plan cannot be safely persisted."""


def build_fallback_plan(settings: TrainingSettings) -> ExperimentPlan:
    """Build the no-LLM plan from validated catalog defaults."""

    catalog = build_model_catalog(settings)
    experiments = (
        PlannedExperiment(
            algorithm=ModelName.LOGISTIC_REGRESSION,
            parameters=resolve_model_parameters(
                catalog[ModelName.LOGISTIC_REGRESSION],
                {},
            ),
            reason="Interpretable linear baseline with balanced class weighting.",
        ),
        PlannedExperiment(
            algorithm=ModelName.RANDOM_FOREST,
            parameters=resolve_model_parameters(
                catalog[ModelName.RANDOM_FOREST],
                {},
            ),
            reason="Bounded nonlinear baseline for feature interactions.",
        ),
    )
    if len(experiments) > settings.maximum_candidates:
        message = "Fallback plan exceeds the configured candidate limit"
        raise ExperimentPlanError(message)

    return ExperimentPlan(
        source=PlanSource.FALLBACK,
        primary_metric=settings.primary_metric,
        experiments=experiments,
        observations=(
            "Fallback used without LLM inference.",
            "All parameters come from versioned deterministic defaults.",
        ),
    )


def write_experiment_plan(plan: ExperimentPlan, output_path: Path) -> Path:
    """Atomically persist a stable, machine-readable experiment plan."""

    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        content = f"{plan.model_dump_json(indent=2)}\n"
        temporary_path.write_text(content, encoding="utf-8")
        temporary_path.replace(output_path)
    except OSError as error:
        message = f"Cannot write experiment plan '{output_path}': {error}"
        raise ExperimentPlanError(message) from error
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path


def configure_logging() -> None:
    """Configure concise default logging for direct CLI execution."""

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def main(arguments: Sequence[str] | None = None) -> int:
    """Create and persist the deterministic fallback plan."""

    parser = argparse.ArgumentParser(description="Create the fallback experiment plan.")
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args(arguments)
    configure_logging()
    try:
        settings = load_training_settings(args.params)
        plan = build_fallback_plan(settings)
        output_path = write_experiment_plan(plan, args.output)
    except (TrainingConfigurationError, ExperimentPlanError) as error:
        LOGGER.error("fallback_plan_failed reason=%s", error)
        return 1

    LOGGER.info(
        "fallback_plan_created experiments=%d primary_metric=%s path=%s",
        len(plan.experiments),
        plan.primary_metric,
        output_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
