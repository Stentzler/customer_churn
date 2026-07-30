import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from src.data.ingest import BatchDisposition
from src.workflow import local_pipeline
from src.workflow.local_pipeline import (
    LocalPipelineError,
    LocalPipelineStatus,
    run_local_pipeline,
)


def test_rejected_batch_stops_before_drift_and_dvc(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "data" / "incoming" / "invalid.csv"
    monkeypatch.setattr(
        local_pipeline,
        "process_incoming_batch",
        lambda **_: SimpleNamespace(disposition=BatchDisposition.REJECTED),
    )
    monkeypatch.setattr(
        local_pipeline,
        "analyze_drift",
        lambda **_: pytest.fail("Rejected data must not reach drift analysis"),
    )

    result = run_local_pipeline(
        input_path,
        dvc_command_runner=lambda _: pytest.fail("Rejected data must not reach DVC"),
    )

    assert result.status is LocalPipelineStatus.REJECTED
    assert result.data_version is None


def test_new_data_version_runs_dvc_and_registers_selected_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _pipeline_paths(tmp_path)
    commands: list[tuple[str, ...]] = []
    comparison_calls: list[dict[str, object]] = []
    analysis_calls: list[dict[str, object]] = []
    _patch_accepted_data_steps(monkeypatch, paths)
    _patch_post_registration_steps(
        monkeypatch,
        comparison_calls=comparison_calls,
        analysis_calls=analysis_calls,
    )
    monkeypatch.setattr(
        local_pipeline,
        "load_tracking_settings",
        lambda *_: SimpleNamespace(tracking_uri="file:///test-mlflow"),
    )
    monkeypatch.setattr(
        local_pipeline,
        "track_and_register_candidates",
        lambda *_args, **_kwargs: SimpleNamespace(
            registered_model_name="customer-churn",
            registered_model_version="2",
        ),
    )

    def run_dvc(arguments: tuple[str, ...]) -> None:
        commands.append(arguments)
        if arguments == ("repro", "profile"):
            _write_data_version(paths["profile"], "new-version")

    result = run_local_pipeline(
        paths["input"],
        params_path=paths["params"],
        env_path=paths["env"],
        data_root=paths["data_root"],
        quality_report_directory=paths["quality_reports"],
        profile_path=paths["profile"],
        drift_directory=paths["drift_directory"],
        model_directory=paths["models"],
        metrics_directory=paths["metrics"],
        plan_path=paths["plan"],
        planner_trace_path=paths["trace"],
        tracking_output_path=paths["tracking"],
        promotion_output_path=paths["promotion"],
        analysis_output_path=paths["analysis"],
        fixed_test_path=paths["fixed_test"],
        dvc_command_runner=run_dvc,
    )

    assert result.status is LocalPipelineStatus.REGISTERED
    assert result.data_version == "new-version"
    assert result.registered_model_version == "2"
    assert result.promotion_passed is True
    assert commands[0][0] == "add"
    assert commands[1:] == [
        ("repro", "profile"),
        ("repro", "--force", "fallback_plan"),
        ("repro", "train"),
    ]
    assert comparison_calls[0]["candidate_version"] == "2"
    assert comparison_calls[0]["promote"] is False
    assert comparison_calls[0]["fixed_test_path"] == paths["fixed_test"]
    assert analysis_calls[0]["promotion_path"] == paths["promotion"]


def test_no_significant_drift_stops_before_training_and_registration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _pipeline_paths(tmp_path)
    commands: list[tuple[str, ...]] = []
    _patch_accepted_data_steps(monkeypatch, paths, is_significant=False)
    monkeypatch.setattr(
        local_pipeline,
        "track_and_register_candidates",
        lambda *_args, **_kwargs: pytest.fail(
            "No-drift data must stop before registration"
        ),
    )

    result = run_local_pipeline(
        paths["input"],
        params_path=paths["params"],
        data_root=paths["data_root"],
        quality_report_directory=paths["quality_reports"],
        drift_directory=paths["drift_directory"],
        dvc_command_runner=lambda arguments: commands.append(tuple(arguments)),
    )

    assert result.status is LocalPipelineStatus.SKIPPED_NO_SIGNIFICANT_DRIFT
    assert result.data_version == "batch-version"
    assert len(commands) == 1
    assert commands[0][0] == "add"


def test_force_retrain_overrides_no_significant_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _pipeline_paths(tmp_path)
    commands: list[tuple[str, ...]] = []
    _patch_accepted_data_steps(monkeypatch, paths, is_significant=False)
    _patch_post_registration_steps(monkeypatch)
    monkeypatch.setattr(
        local_pipeline,
        "load_tracking_settings",
        lambda *_: SimpleNamespace(tracking_uri="file:///test-mlflow"),
    )
    monkeypatch.setattr(
        local_pipeline,
        "track_and_register_candidates",
        lambda *_args, **_kwargs: SimpleNamespace(
            registered_model_name="customer-churn",
            registered_model_version="3",
        ),
    )

    def run_dvc(arguments: tuple[str, ...]) -> None:
        commands.append(arguments)
        if arguments == ("repro", "profile"):
            _write_data_version(paths["profile"], "forced-version")

    result = run_local_pipeline(
        paths["input"],
        params_path=paths["params"],
        env_path=paths["env"],
        data_root=paths["data_root"],
        quality_report_directory=paths["quality_reports"],
        profile_path=paths["profile"],
        drift_directory=paths["drift_directory"],
        model_directory=paths["models"],
        metrics_directory=paths["metrics"],
        plan_path=paths["plan"],
        planner_trace_path=paths["trace"],
        tracking_output_path=paths["tracking"],
        promotion_output_path=paths["promotion"],
        analysis_output_path=paths["analysis"],
        fixed_test_path=paths["fixed_test"],
        dvc_command_runner=run_dvc,
        force_retrain=True,
    )

    assert result.status is LocalPipelineStatus.REGISTERED
    assert result.data_version == "forced-version"
    assert commands[1:] == [
        ("repro", "profile"),
        ("repro", "--force", "fallback_plan"),
        ("repro", "train"),
    ]


def test_unchanged_curated_data_skips_remote_registration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _pipeline_paths(tmp_path)
    _patch_accepted_data_steps(monkeypatch, paths)
    _write_data_version(paths["tracking"], "same-version")
    monkeypatch.setattr(
        local_pipeline,
        "track_and_register_candidates",
        lambda *_args, **_kwargs: pytest.fail(
            "Unchanged data must not create another remote model version"
        ),
    )

    def run_dvc(arguments: tuple[str, ...]) -> None:
        if arguments == ("repro", "profile"):
            _write_data_version(paths["profile"], "same-version")

    result = run_local_pipeline(
        paths["input"],
        params_path=paths["params"],
        data_root=paths["data_root"],
        quality_report_directory=paths["quality_reports"],
        profile_path=paths["profile"],
        drift_directory=paths["drift_directory"],
        tracking_output_path=paths["tracking"],
        dvc_command_runner=run_dvc,
    )

    assert result.status is LocalPipelineStatus.SKIPPED_UNCHANGED
    assert result.data_version == "same-version"


def test_dvc_failure_blocks_registration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _pipeline_paths(tmp_path)
    _patch_accepted_data_steps(monkeypatch, paths)
    monkeypatch.setattr(
        local_pipeline,
        "track_and_register_candidates",
        lambda *_args, **_kwargs: pytest.fail(
            "A DVC failure must block remote registration"
        ),
    )

    with pytest.raises(LocalPipelineError, match="cache unavailable"):
        run_local_pipeline(
            paths["input"],
            params_path=paths["params"],
            data_root=paths["data_root"],
            quality_report_directory=paths["quality_reports"],
            profile_path=paths["profile"],
            drift_directory=paths["drift_directory"],
            tracking_output_path=paths["tracking"],
            dvc_command_runner=lambda _: (_ for _ in ()).throw(
                LocalPipelineError("cache unavailable")
            ),
        )


def _patch_accepted_data_steps(
    monkeypatch: pytest.MonkeyPatch,
    paths: dict[str, Path],
    *,
    is_significant: bool = True,
) -> None:
    reports = SimpleNamespace(
        json_path=paths["quality_reports"] / "batch.validation.json",
        markdown_path=paths["quality_reports"] / "batch.validation.md",
    )
    monkeypatch.setattr(
        local_pipeline,
        "process_incoming_batch",
        lambda **_: SimpleNamespace(
            disposition=BatchDisposition.ACCEPTED,
            routed_path=paths["accepted"],
            reports=reports,
        ),
    )
    monkeypatch.setattr(
        local_pipeline,
        "analyze_drift",
        lambda **_: SimpleNamespace(
            json_path=paths["drift_directory"] / "batch.drift.json",
            html_path=paths["drift_directory"] / "batch.drift.html",
            result=SimpleNamespace(
                current_data_version="batch-version",
                feature_drift=SimpleNamespace(is_significant=is_significant),
            ),
        ),
    )


def _patch_post_registration_steps(
    monkeypatch: pytest.MonkeyPatch,
    *,
    comparison_calls: list[dict[str, object]] | None = None,
    analysis_calls: list[dict[str, object]] | None = None,
) -> None:
    """Replace remote comparison and report writing with observable test doubles."""

    monkeypatch.setattr(
        local_pipeline,
        "load_promotion_policy",
        lambda *_: "policy",
    )

    def compare(**kwargs: object) -> SimpleNamespace:
        if comparison_calls is not None:
            comparison_calls.append(kwargs)
        return SimpleNamespace(passed=True)

    def analyze(**kwargs: object) -> Path:
        if analysis_calls is not None:
            analysis_calls.append(kwargs)
        output_path = kwargs["output_path"]
        assert isinstance(output_path, Path)
        return output_path

    monkeypatch.setattr(local_pipeline, "compare_and_maybe_promote", compare)
    monkeypatch.setattr(local_pipeline, "write_agent_analysis", analyze)


def _pipeline_paths(tmp_path: Path) -> dict[str, Path]:
    data_root = tmp_path / "data"
    return {
        "accepted": data_root / "accepted" / "batch.csv",
        "analysis": tmp_path / "artifacts" / "agent" / "analysis.md",
        "data_root": data_root,
        "drift_directory": tmp_path / "reports" / "drift",
        "env": tmp_path / ".env",
        "fixed_test": data_root / "test" / "fixed_test.csv",
        "input": data_root / "incoming" / "batch.csv",
        "metrics": tmp_path / "artifacts" / "metrics",
        "models": tmp_path / "artifacts" / "models",
        "params": tmp_path / "params.yaml",
        "plan": tmp_path / "artifacts" / "plans" / "fallback.json",
        "profile": tmp_path / "reports" / "profile.json",
        "promotion": tmp_path / "artifacts" / "metrics" / "promotion.json",
        "quality_reports": tmp_path / "reports" / "quality",
        "trace": tmp_path / "artifacts" / "agent" / "trace.json",
        "tracking": tmp_path / "artifacts" / "metrics" / "tracking.json",
    }


def _write_data_version(path: Path, data_version: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"data_version": data_version}), encoding="utf-8")
