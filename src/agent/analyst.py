"""Human-readable analysis based on verified pipeline artifacts."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

LOGGER = logging.getLogger(__name__)
LOG_FORMAT = "%(levelname)s %(name)s %(message)s"
DEFAULT_PLAN_PATH = Path("artifacts/experiment-plans/fallback.json")
DEFAULT_TRACE_PATH = Path("artifacts/agent/planner-trace.json")
DEFAULT_PROFILE_PATH = Path("reports/data-profile/training.profile.json")
DEFAULT_METRICS_DIRECTORY = Path("artifacts/metrics")
DEFAULT_OUTPUT_PATH = Path("artifacts/agent/agent-analysis.md")


class AgentAnalysisError(RuntimeError):
    """Raised when the agent analysis report cannot be generated."""


def write_agent_analysis(
    *,
    plan_path: Path = DEFAULT_PLAN_PATH,
    trace_path: Path = DEFAULT_TRACE_PATH,
    profile_path: Path = DEFAULT_PROFILE_PATH,
    metrics_directory: Path = DEFAULT_METRICS_DIRECTORY,
    promotion_path: Path | None = None,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Create a Markdown report from verified machine-readable artifacts."""

    plan = _read_required_json(plan_path)
    selection = _read_required_json(metrics_directory / "selection.json")
    profile = _read_optional_json(profile_path)
    trace = _read_optional_json(trace_path)
    promotion = (
        _read_optional_json(promotion_path) if promotion_path is not None else None
    )
    candidate_metrics = _read_candidate_metrics(metrics_directory)
    content = build_agent_analysis_markdown(
        plan=plan,
        trace=trace,
        profile=profile,
        selection=selection,
        candidate_metrics=candidate_metrics,
        promotion=promotion,
    )
    _write_text_atomically(output_path, content)
    LOGGER.info("agent_analysis_created path=%s", output_path)
    return output_path


def build_agent_analysis_markdown(
    *,
    plan: Mapping[str, object],
    trace: Mapping[str, object] | None,
    profile: Mapping[str, object] | None,
    selection: Mapping[str, object],
    candidate_metrics: tuple[Mapping[str, object], ...],
    promotion: Mapping[str, object] | None,
) -> str:
    """Build the deterministic human-readable agent report."""

    selected_model = _string(selection, "selected_model", default="unknown")
    primary_metric = _string(
        selection,
        "primary_metric",
        default=_string(plan, "primary_metric", default="unknown"),
    )
    selected_value = _number(selection, "selected_value")
    lines = [
        "# Agent Analysis",
        "",
        "## Input Context",
        f"- Data profile: {_profile_summary(profile)}",
        f"- Plan source: {_string(plan, 'source', default='unknown')}",
        f"- Primary metric: {primary_metric}",
        f"- Candidate limit observed: {len(_experiments(plan))}",
        "",
        "## Agent Decision",
        f"- Used fallback: {_used_fallback(trace, plan)}",
        f"- Fallback reason: {_fallback_reason(trace, plan)}",
        "- Training decision authority: deterministic training code",
        "- Promotion decision authority: deterministic promotion gates",
        "",
        "## Proposed Experiments",
        *_experiment_lines(plan),
        "",
        "## Validation Result",
        f"- Schema valid: {_trace_validation(trace, 'schema_valid')}",
        f"- Catalog policy valid: {_trace_validation(trace, 'catalog_policy_valid')}",
        "- Raw customer rows sent to LLM: no",
        "",
        "## Training Result",
        f"- Selected model: {selected_model}",
        f"- Selected {primary_metric}: {_format_optional_number(selected_value)}",
        *_candidate_metric_lines(candidate_metrics),
        "",
        "## Promotion Result",
        *_promotion_lines(promotion),
        "",
        "## Final Recommendation",
        _final_recommendation(
            selected_model=selected_model,
            primary_metric=primary_metric,
            selected_value=selected_value,
            promotion=promotion,
        ),
        "",
    ]
    return "\n".join(lines)


def _read_candidate_metrics(
    metrics_directory: Path,
) -> tuple[Mapping[str, object], ...]:
    ignored = {
        "failures.json",
        "mlflow-tracking.json",
        "promotion.json",
        "selection.json",
    }
    metrics = []
    for path in sorted(metrics_directory.glob("*.json")):
        if path.name in ignored:
            continue
        metrics.append(_read_required_json(path))
    return tuple(metrics)


def _experiment_lines(plan: Mapping[str, object]) -> list[str]:
    lines = []
    for index, experiment in enumerate(_experiments(plan), start=1):
        algorithm = _string(experiment, "algorithm", default="unknown")
        reason = _string(experiment, "reason", default="No reason provided.")
        parameters = experiment.get("parameters")
        lines.append(f"{index}. `{algorithm}` parameters={parameters} reason={reason}")
    return lines or ["- No experiments found in plan artifact."]


def _candidate_metric_lines(metrics: tuple[Mapping[str, object], ...]) -> list[str]:
    if not metrics:
        return ["- Candidate metrics: not available"]
    lines = ["", "Candidate metrics:"]
    for candidate in metrics:
        model_name = _string(candidate, "model_name", default="unknown")
        roc_auc = _format_optional_number(_number(candidate, "roc_auc"))
        f1 = _format_optional_number(_number(candidate, "f1"))
        recall = _format_optional_number(_number(candidate, "recall"))
        lines.append(f"- `{model_name}` roc_auc={roc_auc} f1={f1} recall={recall}")
    return lines


def _promotion_lines(promotion: Mapping[str, object] | None) -> list[str]:
    if promotion is None:
        return ["- Promotion report: not available"]
    passed = promotion.get("passed")
    promoted = promotion.get("promoted")
    version = _string(promotion, "candidate_version", default="unknown")
    reasons = promotion.get("reasons")
    return [
        f"- Candidate version: {version}",
        f"- Promotion gates passed: {passed}",
        f"- Champion alias updated: {promoted}",
        f"- Reasons: {reasons}",
    ]


def _final_recommendation(
    *,
    selected_model: str,
    primary_metric: str,
    selected_value: float | None,
    promotion: Mapping[str, object] | None,
) -> str:
    metric_summary = _format_optional_number(selected_value)
    if promotion is None:
        return (
            f"`{selected_model}` was selected by deterministic training using "
            f"`{primary_metric}`={metric_summary}. Review promotion separately."
        )
    if promotion.get("promoted") is True:
        return (
            f"`{selected_model}` was selected and promoted after deterministic "
            "quality gates passed."
        )
    if promotion.get("passed") is True:
        return (
            f"`{selected_model}` passed promotion gates, but alias movement was "
            "not requested."
        )
    return (
        f"`{selected_model}` was selected by training, but promotion gates did "
        "not approve it as champion."
    )


def _profile_summary(profile: Mapping[str, object] | None) -> str:
    if profile is None:
        return "not available"
    dataset_name = _string(profile, "dataset_name", default="unknown")
    rows = profile.get("row_count", "unknown")
    features = profile.get("feature_count", "unknown")
    data_version = _string(profile, "data_version", default="unknown")
    return f"{dataset_name} rows={rows} features={features} data_version={data_version}"


def _used_fallback(
    trace: Mapping[str, object] | None,
    plan: Mapping[str, object],
) -> object:
    if trace is not None and "used_fallback" in trace:
        return trace["used_fallback"]
    return _string(plan, "source", default="unknown") == "fallback"


def _fallback_reason(
    trace: Mapping[str, object] | None,
    plan: Mapping[str, object],
) -> object:
    if trace is not None:
        return trace.get("fallback_reason") or "not used"
    observations = plan.get("observations")
    return observations if observations else "not available"


def _trace_validation(trace: Mapping[str, object] | None, key: str) -> object:
    if trace is None:
        return "not available"
    validation = trace.get("validation_result")
    if not isinstance(validation, dict):
        return "not available"
    return validation.get(key, "not available")


def _experiments(plan: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    experiments = plan.get("experiments")
    if not isinstance(experiments, list):
        return ()
    return tuple(
        cast(Mapping[str, object], item)
        for item in experiments
        if isinstance(item, dict)
    )


def _read_required_json(path: Path) -> Mapping[str, object]:
    if not path.is_file():
        raise AgentAnalysisError(f"Required analysis input does not exist: {path}")
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise AgentAnalysisError(f"Analysis input must be a JSON object: {path}")
    return cast(Mapping[str, object], payload)


def _read_optional_json(path: Path) -> Mapping[str, object] | None:
    if not path.is_file():
        return None
    return _read_required_json(path)


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AgentAnalysisError(
            f"Cannot read JSON analysis input '{path}': {error}"
        ) from error


def _write_text_atomically(path: Path, content: str) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path.write_text(content, encoding="utf-8")
        temporary_path.replace(path)
    except OSError as error:
        raise AgentAnalysisError(
            f"Cannot write agent analysis '{path}': {error}"
        ) from error
    finally:
        temporary_path.unlink(missing_ok=True)


def _string(mapping: Mapping[str, object], key: str, *, default: str) -> str:
    value = mapping.get(key)
    return value if isinstance(value, str) and value else default


def _number(mapping: Mapping[str, object], key: str) -> float | None:
    value = mapping.get(key)
    if type(value) in {int, float}:
        return float(value)
    return None


def _format_optional_number(value: float | None) -> str:
    if value is None:
        return "not available"
    return f"{value:.6f}"


def configure_logging() -> None:
    """Configure concise default logging for direct CLI execution."""

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def main(arguments: Sequence[str] | None = None) -> int:
    """Generate the local agent analysis report."""

    parser = argparse.ArgumentParser(
        description="Create a human-readable agent analysis report."
    )
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN_PATH)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE_PATH)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--metrics-dir", type=Path, default=DEFAULT_METRICS_DIRECTORY)
    parser.add_argument("--promotion", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args(arguments)
    configure_logging()
    try:
        write_agent_analysis(
            plan_path=args.plan,
            trace_path=args.trace,
            profile_path=args.profile,
            metrics_directory=args.metrics_dir,
            promotion_path=args.promotion,
            output_path=args.output,
        )
    except AgentAnalysisError as error:
        LOGGER.error("agent_analysis_failed reason=%s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
