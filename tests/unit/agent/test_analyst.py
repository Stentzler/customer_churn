import json
from pathlib import Path

from src.agent.analyst import build_agent_analysis_markdown, main, write_agent_analysis


def test_agent_analysis_markdown_summarizes_verified_artifacts() -> None:
    content = build_agent_analysis_markdown(
        plan=_plan(),
        trace=_trace(),
        profile=_profile(),
        selection=_selection(),
        candidate_metrics=(_candidate("logistic_regression", 0.91),),
        promotion=_promotion(promoted=False, passed=True),
    )

    assert "# Agent Analysis" in content
    assert "- Used fallback: True" in content
    assert "`logistic_regression` parameters={'C': 1.0} reason=Baseline." in content
    assert "- Selected model: logistic_regression" in content
    assert "- Promotion gates passed: True" in content
    assert "passed promotion gates, but alias movement was not requested" in content


def test_write_agent_analysis_creates_markdown_report(tmp_path: Path) -> None:
    plan_path = _write_json(tmp_path / "plan.json", _plan())
    trace_path = _write_json(tmp_path / "trace.json", _trace())
    profile_path = _write_json(tmp_path / "profile.json", _profile())
    metrics_directory = tmp_path / "metrics"
    metrics_directory.mkdir()
    _write_json(metrics_directory / "selection.json", _selection())
    _write_json(
        metrics_directory / "logistic_regression.json",
        _candidate("logistic_regression", 0.91),
    )
    output_path = tmp_path / "agent" / "agent-analysis.md"

    result_path = write_agent_analysis(
        plan_path=plan_path,
        trace_path=trace_path,
        profile_path=profile_path,
        metrics_directory=metrics_directory,
        promotion_path=tmp_path / "missing-promotion.json",
        output_path=output_path,
    )

    assert result_path == output_path
    assert output_path.read_text(encoding="utf-8").endswith("\n")
    assert "Promotion report: not available" in output_path.read_text(encoding="utf-8")


def test_agent_analysis_cli_writes_report(tmp_path: Path) -> None:
    plan_path = _write_json(tmp_path / "plan.json", _plan())
    trace_path = _write_json(tmp_path / "trace.json", _trace())
    profile_path = _write_json(tmp_path / "profile.json", _profile())
    metrics_directory = tmp_path / "metrics"
    metrics_directory.mkdir()
    _write_json(metrics_directory / "selection.json", _selection())
    _write_json(
        metrics_directory / "logistic_regression.json",
        _candidate("logistic_regression", 0.91),
    )
    output_path = tmp_path / "agent-analysis.md"

    exit_code = main(
        [
            "--plan",
            str(plan_path),
            "--trace",
            str(trace_path),
            "--profile",
            str(profile_path),
            "--metrics-dir",
            str(metrics_directory),
            "--promotion",
            str(tmp_path / "missing-promotion.json"),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert "Agent Decision" in output_path.read_text(encoding="utf-8")


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(f"{json.dumps(payload)}\n", encoding="utf-8")
    return path


def _plan() -> dict[str, object]:
    return {
        "experiments": [
            {
                "algorithm": "logistic_regression",
                "parameters": {"C": 1.0},
                "reason": "Baseline.",
            }
        ],
        "observations": ["Fallback used."],
        "primary_metric": "roc_auc",
        "schema_version": "1.0",
        "source": "fallback",
    }


def _trace() -> dict[str, object]:
    return {
        "fallback_reason": "LLM planning is disabled.",
        "plan_source": "fallback",
        "used_fallback": True,
        "validation_result": {
            "catalog_policy_valid": True,
            "schema_valid": True,
        },
    }


def _profile() -> dict[str, object]:
    return {
        "data_version": "data-123",
        "dataset_name": "training.csv",
        "feature_count": 8,
        "row_count": 100,
    }


def _selection() -> dict[str, object]:
    return {
        "primary_metric": "roc_auc",
        "selected_model": "logistic_regression",
        "selected_value": 0.91,
    }


def _candidate(model_name: str, roc_auc: float) -> dict[str, object]:
    return {
        "f1": 0.8,
        "model_name": model_name,
        "recall": 0.7,
        "roc_auc": roc_auc,
    }


def _promotion(*, promoted: bool, passed: bool) -> dict[str, object]:
    return {
        "candidate_version": "3",
        "passed": passed,
        "promoted": promoted,
        "reasons": ["promotion_not_requested"],
    }
