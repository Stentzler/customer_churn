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
    _patch_accepted_data_steps(monkeypatch, paths)
    monkeypatch.setattr(local_pipeline, "load_tracking_settings", lambda *_: "settings")
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
        if arguments == ("repro", "profile", "train"):
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
        tracking_output_path=paths["tracking"],
        dvc_command_runner=run_dvc,
    )

    assert result.status is LocalPipelineStatus.REGISTERED
    assert result.data_version == "new-version"
    assert result.registered_model_version == "2"
    assert commands[0][0] == "add"
    assert commands[1] == ("repro", "profile", "train")


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
        if arguments == ("repro", "profile", "train"):
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
        ),
    )


def _pipeline_paths(tmp_path: Path) -> dict[str, Path]:
    data_root = tmp_path / "data"
    return {
        "accepted": data_root / "accepted" / "batch.csv",
        "data_root": data_root,
        "drift_directory": tmp_path / "reports" / "drift",
        "env": tmp_path / ".env",
        "input": data_root / "incoming" / "batch.csv",
        "metrics": tmp_path / "artifacts" / "metrics",
        "models": tmp_path / "artifacts" / "models",
        "params": tmp_path / "params.yaml",
        "plan": tmp_path / "artifacts" / "plans" / "fallback.json",
        "profile": tmp_path / "reports" / "profile.json",
        "quality_reports": tmp_path / "reports" / "quality",
        "tracking": tmp_path / "artifacts" / "metrics" / "tracking.json",
    }


def _write_data_version(path: Path, data_version: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"data_version": data_version}), encoding="utf-8")
