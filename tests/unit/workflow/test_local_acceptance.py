from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from scripts import local_acceptance
from src.data.generate import DatasetScenario
from src.workflow.local_pipeline import LocalPipelineStatus


def test_acceptance_batch_uses_unique_customer_ids(
    tmp_path: Path,
) -> None:
    output_path = local_acceptance.write_acceptance_batch(
        rows=50,
        seed=2026073001,
        filename="acceptance-test.csv",
        params_path=Path("params.yaml"),
        data_root=tmp_path / "data",
    )

    customer_ids = [
        line.split(",", maxsplit=1)[0]
        for line in output_path.read_text(encoding="utf-8").splitlines()[1:]
    ]

    assert output_path == tmp_path / "data" / "incoming" / "acceptance-test.csv"
    assert len(customer_ids) == 50
    assert len(set(customer_ids)) == 50
    assert all(
        customer_id.startswith("CUST-2026073001-") for customer_id in customer_ids
    )


def test_acceptance_batch_rejects_nested_filename(tmp_path: Path) -> None:
    with pytest.raises(local_acceptance.LocalAcceptanceError, match="directories"):
        local_acceptance.write_acceptance_batch(
            rows=50,
            seed=2026073001,
            filename="nested/batch.csv",
            params_path=Path("params.yaml"),
            data_root=tmp_path / "data",
        )


def test_drifted_acceptance_batch_shifts_expected_features(tmp_path: Path) -> None:
    normal_path = local_acceptance.write_acceptance_batch(
        rows=200,
        seed=2026073002,
        filename="normal.csv",
        params_path=Path("params.yaml"),
        data_root=tmp_path / "normal-data",
    )
    drifted_path = local_acceptance.write_acceptance_batch(
        rows=200,
        seed=2026073002,
        filename="drifted.csv",
        params_path=Path("params.yaml"),
        data_root=tmp_path / "drifted-data",
        scenario=DatasetScenario.DRIFTED,
    )

    normal = pd.read_csv(normal_path)
    drifted = pd.read_csv(drifted_path)

    assert drifted["customer_id"].tolist() == normal["customer_id"].tolist()
    assert drifted["monthly_spend"].mean() > normal["monthly_spend"].mean()
    assert drifted["support_tickets_90d"].mean() > normal["support_tickets_90d"].mean()
    assert drifted["usage_hours_monthly"].mean() < normal["usage_hours_monthly"].mean()


def test_local_acceptance_uses_local_mlflow_and_does_not_push_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[object] = []
    batch_path = tmp_path / "data" / "incoming" / "acceptance.csv"
    monkeypatch.setattr(
        local_acceptance,
        "write_acceptance_batch",
        lambda **_: batch_path,
    )
    monkeypatch.setattr(
        local_acceptance,
        "write_local_mlflow_env",
        lambda: tmp_path / "local.env",
    )
    monkeypatch.setattr(
        local_acceptance,
        "push_dvc_remote",
        lambda: pytest.fail("local acceptance must not push DVC by default"),
    )

    def run_pipeline(_input_path: Path, **kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(
            status=LocalPipelineStatus.REGISTERED,
            input_path=batch_path,
            data_version="version",
            registered_model_name="customer-churn",
            registered_model_version="1",
        )

    monkeypatch.setattr(local_acceptance, "run_local_pipeline", run_pipeline)

    result = local_acceptance.run_acceptance_scenario(
        rows=50,
        seed=2026073001,
        filename="acceptance.csv",
        params_path=tmp_path / "params.yaml",
        data_root=tmp_path / "data",
        remote=False,
        push_dvc=False,
    )

    assert result is LocalPipelineStatus.REGISTERED
    assert calls[0]["env_path"] == tmp_path / "local.env"
    assert calls[0]["force_retrain"] is True


def test_local_mlflow_environment_disables_optional_llm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("LLM_ENABLED", "true")

    env_path = local_acceptance.write_local_mlflow_env(
        output_path=tmp_path / "local.env",
        backend_path=tmp_path / "mlflow.db",
    )

    assert "LLM_ENABLED=false" in env_path.read_text(encoding="utf-8")
    assert local_acceptance.os.environ["LLM_ENABLED"] == "false"


def test_remote_acceptance_uses_env_and_pushes_dvc(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pushed = False
    batch_path = tmp_path / "data" / "incoming" / "acceptance.csv"
    monkeypatch.setattr(
        local_acceptance,
        "write_acceptance_batch",
        lambda **_: batch_path,
    )

    def run_pipeline(_input_path: Path, **kwargs: object) -> SimpleNamespace:
        assert kwargs["env_path"] == Path(".env")
        return SimpleNamespace(
            status=LocalPipelineStatus.REGISTERED,
            input_path=batch_path,
            data_version="version",
            registered_model_name="customer-churn",
            registered_model_version="2",
        )

    def push() -> None:
        nonlocal pushed
        pushed = True

    monkeypatch.setattr(local_acceptance, "run_local_pipeline", run_pipeline)
    monkeypatch.setattr(local_acceptance, "push_dvc_remote", push)

    result = local_acceptance.run_acceptance_scenario(
        rows=50,
        seed=2026073001,
        filename="acceptance.csv",
        params_path=tmp_path / "params.yaml",
        data_root=tmp_path / "data",
        remote=True,
        push_dvc=True,
    )

    assert result is LocalPipelineStatus.REGISTERED
    assert pushed is True


def test_rejected_acceptance_result_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    batch_path = tmp_path / "data" / "incoming" / "acceptance.csv"
    monkeypatch.setattr(
        local_acceptance,
        "write_acceptance_batch",
        lambda **_: batch_path,
    )
    monkeypatch.setattr(
        local_acceptance,
        "write_local_mlflow_env",
        lambda: tmp_path / "local.env",
    )
    monkeypatch.setattr(
        local_acceptance,
        "run_local_pipeline",
        lambda _input_path, **_: SimpleNamespace(
            status=LocalPipelineStatus.REJECTED,
            input_path=batch_path,
            data_version=None,
            registered_model_name=None,
            registered_model_version=None,
        ),
    )

    with pytest.raises(local_acceptance.LocalAcceptanceError, match="rejected"):
        local_acceptance.run_acceptance_scenario(
            rows=50,
            seed=2026073001,
            filename="acceptance.csv",
            params_path=tmp_path / "params.yaml",
            data_root=tmp_path / "data",
            remote=False,
            push_dvc=False,
        )


def test_create_only_cli_does_not_run_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    created: list[object] = []
    monkeypatch.setattr(
        local_acceptance,
        "write_acceptance_batch",
        lambda **kwargs: created.append(kwargs) or tmp_path / "data/incoming/batch.csv",
    )
    monkeypatch.setattr(
        local_acceptance,
        "run_acceptance_scenario",
        lambda **_: pytest.fail("create-only must not run the pipeline"),
    )

    status_code = local_acceptance.main(
        [
            "--create-only",
            "--rows",
            "50",
            "--seed",
            "2026073001",
            "--filename",
            "batch.csv",
            "--scenario",
            "drifted",
            "--data-root",
            str(tmp_path / "data"),
        ]
    )

    assert status_code == 0
    assert created[0]["filename"] == "batch.csv"
    assert created[0]["scenario"] is DatasetScenario.DRIFTED
