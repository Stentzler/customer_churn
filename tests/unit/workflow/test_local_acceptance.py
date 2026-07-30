from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts import local_acceptance
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
