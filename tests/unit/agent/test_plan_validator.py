from pathlib import Path

import pytest
from src.agent.plan_validator import (
    ExperimentPlanValidationError,
    load_and_validate_experiment_plan,
    validate_experiment_plan,
)
from src.agent.planner import build_fallback_plan, write_experiment_plan
from src.agent.schemas import PlannedExperiment
from src.training.settings import ModelName, load_training_settings

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


def test_plan_loading_and_validation_resolves_omitted_defaults(tmp_path: Path) -> None:
    settings = load_training_settings(PARAMS_PATH)
    proposed = build_fallback_plan(settings)
    experiment = proposed.experiments[0].model_copy(update={"parameters": {"C": 2.0}})
    proposed = proposed.model_copy(update={"experiments": (experiment,)})
    plan_path = write_experiment_plan(proposed, tmp_path / "plan.json")

    approved = load_and_validate_experiment_plan(plan_path, settings)

    assert approved.experiments[0].parameters == {"C": 2.0, "max_iter": 1000}


def test_plan_cannot_override_primary_metric() -> None:
    settings = load_training_settings(PARAMS_PATH)
    proposed = build_fallback_plan(settings).model_copy(update={"primary_metric": "f1"})

    with pytest.raises(ExperimentPlanValidationError, match="cannot override"):
        validate_experiment_plan(proposed, settings)


def test_plan_rejects_duplicate_algorithms() -> None:
    settings = load_training_settings(PARAMS_PATH)
    proposed = build_fallback_plan(settings)
    duplicate = PlannedExperiment(
        algorithm=ModelName.LOGISTIC_REGRESSION,
        parameters={"C": 2.0},
        reason="Duplicate must not create repeated candidate runs.",
    )
    proposed = proposed.model_copy(
        update={"experiments": (proposed.experiments[0], duplicate)}
    )

    with pytest.raises(ExperimentPlanValidationError, match="duplicate"):
        validate_experiment_plan(proposed, settings)


def test_plan_rejects_out_of_bounds_catalog_parameter() -> None:
    settings = load_training_settings(PARAMS_PATH)
    proposed = build_fallback_plan(settings)
    experiment = proposed.experiments[1].model_copy(
        update={"parameters": {"n_estimators": 5000}}
    )
    proposed = proposed.model_copy(update={"experiments": (experiment,)})

    with pytest.raises(ExperimentPlanValidationError, match="model-catalog"):
        validate_experiment_plan(proposed, settings)
