"""Deterministic fallback experiment-plan generation and persistence."""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from pydantic import ValidationError
from src.agent.llm import (
    LlmProvider,
    LlmProviderError,
    OpenAICompatibleChatProvider,
    OpenAICompatibleProviderSettings,
)
from src.agent.plan_validator import (
    ExperimentPlanValidationError,
    validate_experiment_plan,
)
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
DEFAULT_ENV_PATH = Path(".env")
DEFAULT_PROMPT_PATH = Path("prompts/experiment-planner.prompt.md")
DEFAULT_OUTPUT_PATH = Path("artifacts/experiment-plans/fallback.json")
DEFAULT_TRACE_PATH = Path("artifacts/agent/planner-trace.json")
DEFAULT_LLM_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_LLM_MODEL = "openai/gpt-oss-20b"
DEFAULT_LLM_PROVIDER = "groq"
DEFAULT_LLM_TIMEOUT_SECONDS = 30.0
DEFAULT_LLM_TEMPERATURE = 0.0
DEFAULT_LLM_MAX_TOKENS = 1200


class ExperimentPlanError(RuntimeError):
    """Raised when an experiment plan cannot be safely persisted."""


@dataclass(frozen=True, slots=True)
class PlannerSettings:
    """Non-secret switches controlling optional LLM planning."""

    llm_enabled: bool
    llm_provider: str
    llm_model: str | None
    llm_api_key: str | None
    llm_base_url: str
    timeout_seconds: float
    temperature: float
    max_tokens: int


@dataclass(frozen=True, slots=True)
class PlannedExperimentResult:
    """Approved plan plus traceable fallback information."""

    plan: ExperimentPlan
    used_fallback: bool
    fallback_reason: str | None


def load_planner_settings(env_path: Path = DEFAULT_ENV_PATH) -> PlannerSettings:
    """Load optional LLM switches from environment or local ``.env``."""

    load_dotenv(env_path, override=False)
    return PlannerSettings(
        llm_enabled=_parse_boolean(os.getenv("LLM_ENABLED", "")),
        llm_provider=os.getenv("LLM_PROVIDER", "").strip() or DEFAULT_LLM_PROVIDER,
        llm_model=os.getenv("LLM_MODEL", "").strip() or DEFAULT_LLM_MODEL,
        llm_api_key=os.getenv("LLM_API_KEY", "").strip() or None,
        llm_base_url=os.getenv("LLM_BASE_URL", "").strip() or DEFAULT_LLM_BASE_URL,
        timeout_seconds=_parse_positive_float(
            os.getenv("LLM_TIMEOUT_SECONDS", ""),
            DEFAULT_LLM_TIMEOUT_SECONDS,
            "LLM_TIMEOUT_SECONDS",
        ),
        temperature=_parse_non_negative_float(
            os.getenv("LLM_TEMPERATURE", ""),
            DEFAULT_LLM_TEMPERATURE,
            "LLM_TEMPERATURE",
        ),
        max_tokens=_parse_positive_integer(
            os.getenv("LLM_MAX_TOKENS", ""),
            DEFAULT_LLM_MAX_TOKENS,
            "LLM_MAX_TOKENS",
        ),
    )


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


def create_experiment_plan(
    settings: TrainingSettings,
    *,
    planner_settings: PlannerSettings | None = None,
    provider: LlmProvider | None = None,
    prompt_path: Path = DEFAULT_PROMPT_PATH,
) -> PlannedExperimentResult:
    """Create an approved experiment plan or safely fall back.

    The provider result is intentionally treated as untrusted. It must pass the
    Pydantic schema and the deterministic model-catalog policy before training
    can use it.
    """

    active_planner_settings = planner_settings or PlannerSettings(
        llm_enabled=False,
        llm_provider=DEFAULT_LLM_PROVIDER,
        llm_model=None,
        llm_api_key=None,
        llm_base_url=DEFAULT_LLM_BASE_URL,
        timeout_seconds=DEFAULT_LLM_TIMEOUT_SECONDS,
        temperature=DEFAULT_LLM_TEMPERATURE,
        max_tokens=DEFAULT_LLM_MAX_TOKENS,
    )
    fallback_plan = build_fallback_plan(settings)
    if not active_planner_settings.llm_enabled:
        return PlannedExperimentResult(
            plan=fallback_plan,
            used_fallback=True,
            fallback_reason="LLM planning is disabled.",
        )

    active_provider = provider or create_llm_provider(active_planner_settings)
    try:
        prompt = build_experiment_planner_prompt(
            settings=settings,
            planner_settings=active_planner_settings,
            prompt_path=prompt_path,
        )
        raw_plan = active_provider.generate_experiment_plan(prompt)
        proposed_plan = ExperimentPlan.model_validate_json(raw_plan).model_copy(
            update={"source": PlanSource.LLM},
        )
        approved_plan = validate_experiment_plan(proposed_plan, settings)
    except (
        OSError,
        ValidationError,
        ExperimentPlanValidationError,
        LlmProviderError,
    ) as error:
        return PlannedExperimentResult(
            plan=fallback_plan,
            used_fallback=True,
            fallback_reason=str(error),
        )

    return PlannedExperimentResult(
        plan=approved_plan,
        used_fallback=False,
        fallback_reason=None,
    )


def build_experiment_planner_prompt(
    *,
    settings: TrainingSettings,
    planner_settings: PlannerSettings,
    prompt_path: Path = DEFAULT_PROMPT_PATH,
) -> str:
    """Build a bounded prompt with policy summaries and no raw customer rows."""

    prompt_template = prompt_path.read_text(encoding="utf-8")
    catalog = build_model_catalog(settings)
    allowed_models = ", ".join(model_name.value for model_name in catalog)
    return (
        f"{prompt_template.strip()}\n\n"
        "Execution policy:\n"
        f"- LLM model: {planner_settings.llm_model or 'not configured'}\n"
        f"- Primary metric: {settings.primary_metric}\n"
        f"- Maximum candidates: {settings.maximum_candidates}\n"
        f"- Allowed algorithms: {allowed_models}\n"
        "- Output must be a JSON object matching ExperimentPlan schema version 1.0.\n"
        "- Do not include code, shell commands, package names, file paths, "
        "or raw rows.\n"
    )


def create_llm_provider(settings: PlannerSettings) -> LlmProvider:
    """Create the configured provider without exposing secrets."""

    if not settings.llm_api_key:
        raise LlmProviderError("LLM_API_KEY must be configured when LLM is enabled")
    if not settings.llm_model:
        raise LlmProviderError("LLM_MODEL must be configured when LLM is enabled")
    return OpenAICompatibleChatProvider(
        OpenAICompatibleProviderSettings(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            timeout_seconds=settings.timeout_seconds,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )
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


def write_planner_trace(
    result: PlannedExperimentResult,
    output_path: Path,
    planner_settings: PlannerSettings | None = None,
) -> Path:
    """Persist a machine-readable explanation of plan selection."""

    payload = {
        "fallback_reason": result.fallback_reason,
        "llm_enabled": planner_settings.llm_enabled if planner_settings else None,
        "llm_model": planner_settings.llm_model if planner_settings else None,
        "llm_provider": planner_settings.llm_provider if planner_settings else None,
        "plan_source": result.plan.source.value,
        "primary_metric": result.plan.primary_metric,
        "proposed_experiments": [
            {
                "algorithm": experiment.algorithm.value,
                "parameters": experiment.parameters,
                "reason": experiment.reason,
            }
            for experiment in result.plan.experiments
        ],
        "schema_version": result.plan.schema_version,
        "used_fallback": result.used_fallback,
        "validation_result": {
            "catalog_policy_valid": True,
            "schema_valid": True,
        },
    }
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path.write_text(
            f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
            encoding="utf-8",
        )
        temporary_path.replace(output_path)
    except OSError as error:
        message = f"Cannot write planner trace '{output_path}': {error}"
        raise ExperimentPlanError(message) from error
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path


def configure_logging() -> None:
    """Configure concise default logging for direct CLI execution."""

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def _parse_boolean(value: str) -> bool:
    normalized = value.strip().lower()
    return normalized in {"1", "true", "yes", "y", "on"}


def _parse_positive_float(value: str, default: float, name: str) -> float:
    if not value.strip():
        return default
    try:
        parsed = float(value)
    except ValueError as error:
        raise ExperimentPlanError(f"{name} must be a number") from error
    if parsed <= 0:
        raise ExperimentPlanError(f"{name} must be greater than zero")
    return parsed


def _parse_non_negative_float(value: str, default: float, name: str) -> float:
    if not value.strip():
        return default
    try:
        parsed = float(value)
    except ValueError as error:
        raise ExperimentPlanError(f"{name} must be a number") from error
    if parsed < 0:
        raise ExperimentPlanError(f"{name} cannot be negative")
    return parsed


def _parse_positive_integer(value: str, default: int, name: str) -> int:
    if not value.strip():
        return default
    try:
        parsed = int(value)
    except ValueError as error:
        raise ExperimentPlanError(f"{name} must be an integer") from error
    if parsed < 1:
        raise ExperimentPlanError(f"{name} must be greater than zero")
    return parsed


def main(arguments: Sequence[str] | None = None) -> int:
    """Create and persist the deterministic fallback plan."""

    parser = argparse.ArgumentParser(description="Create the fallback experiment plan.")
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH)
    parser.add_argument("--env", type=Path, default=DEFAULT_ENV_PATH)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--trace-output", type=Path, default=DEFAULT_TRACE_PATH)
    args = parser.parse_args(arguments)
    configure_logging()
    try:
        settings = load_training_settings(args.params)
        planner_settings = load_planner_settings(args.env)
        result = create_experiment_plan(
            settings,
            planner_settings=planner_settings,
            prompt_path=args.prompt,
        )
        output_path = write_experiment_plan(result.plan, args.output)
        trace_path = write_planner_trace(
            result,
            args.trace_output,
            planner_settings,
        )
    except (TrainingConfigurationError, ExperimentPlanError) as error:
        LOGGER.error("fallback_plan_failed reason=%s", error)
        return 1

    LOGGER.info(
        "experiment_plan_created source=%s experiments=%d primary_metric=%s "
        "used_fallback=%s fallback_reason=%s path=%s trace=%s",
        result.plan.source.value,
        len(result.plan.experiments),
        result.plan.primary_metric,
        result.used_fallback,
        result.fallback_reason,
        output_path,
        trace_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
