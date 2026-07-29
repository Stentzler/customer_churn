import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from src.agent.planner import build_fallback_plan, main, write_experiment_plan
from src.agent.schemas import ExperimentPlan, PlanSource
from src.training.settings import load_training_settings

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


def test_fallback_plan_contains_interpretable_and_nonlinear_defaults() -> None:
    plan = build_fallback_plan(load_training_settings(PARAMS_PATH))

    assert plan.source is PlanSource.FALLBACK
    assert plan.primary_metric == "roc_auc"
    assert tuple(item.algorithm.value for item in plan.experiments) == (
        "logistic_regression",
        "random_forest",
    )
    assert plan.experiments[0].parameters == {"C": 1.0, "max_iter": 1000}
    assert plan.experiments[1].parameters == {
        "n_estimators": 200,
        "max_depth": 8,
        "min_samples_leaf": 2,
    }


def test_plan_schema_rejects_unknown_fields() -> None:
    payload = build_fallback_plan(load_training_settings(PARAMS_PATH)).model_dump()
    payload["shell_command"] = "do-not-run"

    with pytest.raises(ValidationError, match="shell_command"):
        ExperimentPlan.model_validate(payload)


def test_experiment_plan_json_round_trip_is_stable(tmp_path: Path) -> None:
    plan = build_fallback_plan(load_training_settings(PARAMS_PATH))
    output_path = tmp_path / "fallback.json"

    write_experiment_plan(plan, output_path)
    restored = ExperimentPlan.model_validate_json(
        output_path.read_text(encoding="utf-8")
    )

    assert restored == plan
    assert output_path.read_text(encoding="utf-8").endswith("\n")


def test_fallback_plan_cli_writes_structured_json(tmp_path: Path) -> None:
    output_path = tmp_path / "plans" / "fallback.json"

    exit_code = main(["--params", str(PARAMS_PATH), "--output", str(output_path)])

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["source"] == "fallback"
    assert len(payload["experiments"]) == 2
