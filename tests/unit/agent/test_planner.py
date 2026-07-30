import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from src.agent.llm import LlmProviderError
from src.agent.planner import (
    PlannerSettings,
    build_fallback_plan,
    create_experiment_plan,
    main,
    write_experiment_plan,
    write_planner_trace,
)
from src.agent.schemas import ExperimentPlan, PlanSource
from src.training.settings import load_training_settings

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"
PROMPT_PATH = PROJECT_ROOT / "prompts" / "experiment-planner.prompt.md"


class StaticPlanProvider:
    """Fake provider used to test LLM planning without network calls."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate_experiment_plan(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


class FailingPlanProvider:
    """Fake provider that exercises deterministic fallback behavior."""

    def generate_experiment_plan(self, prompt: str) -> str:
        raise LlmProviderError("provider unavailable")


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


def test_planner_trace_persists_fallback_reason(tmp_path: Path) -> None:
    plan = build_fallback_plan(load_training_settings(PARAMS_PATH))
    result = create_experiment_plan(
        load_training_settings(PARAMS_PATH),
        planner_settings=_planner_settings(llm_enabled=False),
    )
    output_path = tmp_path / "planner-trace.json"

    write_planner_trace(result, output_path, _planner_settings(llm_enabled=False))

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert result.plan == plan
    assert payload["used_fallback"] is True
    assert payload["llm_provider"] == "groq"
    assert payload["llm_model"] == "fake-model"
    assert payload["fallback_reason"] == "LLM planning is disabled."
    assert payload["validation_result"] == {
        "catalog_policy_valid": True,
        "schema_valid": True,
    }


def test_fallback_plan_cli_writes_structured_json(tmp_path: Path) -> None:
    output_path = tmp_path / "plans" / "fallback.json"
    trace_path = tmp_path / "agent" / "planner-trace.json"
    env_path = tmp_path / ".env"
    env_path.write_text("LLM_ENABLED=false\n", encoding="utf-8")

    exit_code = main(
        [
            "--params",
            str(PARAMS_PATH),
            "--env",
            str(env_path),
            "--output",
            str(output_path),
            "--trace-output",
            str(trace_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["source"] == "fallback"
    assert len(payload["experiments"]) == 2
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["plan_source"] == "fallback"


def test_disabled_llm_uses_fallback_without_calling_provider() -> None:
    settings = load_training_settings(PARAMS_PATH)
    provider = StaticPlanProvider("{}")

    result = create_experiment_plan(
        settings,
        planner_settings=_planner_settings(llm_enabled=False),
        provider=provider,
        prompt_path=PROMPT_PATH,
    )

    assert result.used_fallback is True
    assert result.plan.source is PlanSource.FALLBACK
    assert result.fallback_reason == "LLM planning is disabled."
    assert provider.prompts == []


def test_valid_llm_plan_is_approved_after_policy_validation() -> None:
    settings = load_training_settings(PARAMS_PATH)
    proposed = build_fallback_plan(settings).model_copy(
        update={"source": PlanSource.LLM}
    )
    provider = StaticPlanProvider(proposed.model_dump_json())

    result = create_experiment_plan(
        settings,
        planner_settings=_planner_settings(llm_enabled=True),
        provider=provider,
        prompt_path=PROMPT_PATH,
    )

    assert result.used_fallback is False
    assert result.plan.source is PlanSource.LLM
    assert result.plan.experiments == proposed.experiments
    assert "Allowed algorithms" in provider.prompts[0]


def test_invalid_llm_plan_uses_fallback() -> None:
    settings = load_training_settings(PARAMS_PATH)
    proposed = build_fallback_plan(settings).model_copy(update={"primary_metric": "f1"})
    provider = StaticPlanProvider(proposed.model_dump_json())

    result = create_experiment_plan(
        settings,
        planner_settings=_planner_settings(llm_enabled=True),
        provider=provider,
        prompt_path=PROMPT_PATH,
    )

    assert result.used_fallback is True
    assert result.plan.source is PlanSource.FALLBACK
    assert result.fallback_reason is not None
    assert "cannot override" in result.fallback_reason


def test_provider_failure_uses_fallback() -> None:
    settings = load_training_settings(PARAMS_PATH)

    result = create_experiment_plan(
        settings,
        planner_settings=_planner_settings(llm_enabled=True),
        provider=FailingPlanProvider(),
        prompt_path=PROMPT_PATH,
    )

    assert result.used_fallback is True
    assert result.plan.source is PlanSource.FALLBACK
    assert result.fallback_reason == "provider unavailable"


def _planner_settings(*, llm_enabled: bool) -> PlannerSettings:
    return PlannerSettings(
        llm_api_key="test-key",
        llm_base_url="https://api.groq.com/openai/v1",
        llm_enabled=llm_enabled,
        llm_model="fake-model",
        llm_provider="groq",
        max_tokens=1200,
        temperature=0.0,
        timeout_seconds=30.0,
    )
