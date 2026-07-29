import json
import logging
from pathlib import Path

import pytest
import yaml
from src.data.drift import DataDriftError, analyze_drift, main
from src.data.generate import DatasetScenario, generate_synthetic_dataset
from src.data.settings import (
    DataContractConfig,
    load_data_generation,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


def test_analyze_drift_writes_structured_and_visual_reports(
    tmp_path: Path,
    data_contract_config: DataContractConfig,
) -> None:
    params_path, current_path = _write_drift_inputs(
        tmp_path,
        data_contract_config,
        DatasetScenario.DRIFTED,
    )

    artifacts = analyze_drift(current_path, params_path, tmp_path / "reports")
    payload = json.loads(artifacts.json_path.read_text(encoding="utf-8"))

    assert artifacts.json_path.is_file()
    assert artifacts.html_path.is_file()
    assert artifacts.html_path.stat().st_size > 0
    assert payload["feature_drift"]["is_significant"] is True
    assert payload["feature_drift"]["drifted_features"] == [
        "monthly_spend",
        "support_tickets_90d",
        "usage_hours_monthly",
    ]
    assert payload["target_drift"]["is_drifted"] is False


def test_drift_rejects_current_data_outside_accepted_directory(
    tmp_path: Path,
    data_contract_config: DataContractConfig,
) -> None:
    params_path, accepted_path = _write_drift_inputs(
        tmp_path,
        data_contract_config,
        DatasetScenario.NORMAL,
    )
    incoming_path = tmp_path / "data" / "incoming" / accepted_path.name
    incoming_path.parent.mkdir(parents=True)
    incoming_path.write_bytes(accepted_path.read_bytes())

    with pytest.raises(DataDriftError, match="must be located directly under"):
        analyze_drift(incoming_path, params_path, tmp_path / "reports")

    assert not (tmp_path / "reports").exists()


def test_drift_cli_logs_separate_decisions(
    tmp_path: Path,
    data_contract_config: DataContractConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="src.data.drift")
    params_path, current_path = _write_drift_inputs(
        tmp_path,
        data_contract_config,
        DatasetScenario.NORMAL,
    )

    exit_code = main(
        [
            "--current",
            str(current_path),
            "--params",
            str(params_path),
            "--report-dir",
            str(tmp_path / "reports"),
        ]
    )

    assert exit_code == 0
    assert any(
        record.levelno == logging.INFO
        and "feature_drift=" in record.getMessage()
        and "target_drift=" in record.getMessage()
        for record in caplog.records
    )


def _write_drift_inputs(
    tmp_path: Path,
    contract: DataContractConfig,
    scenario: DatasetScenario,
) -> tuple[Path, Path]:
    generation = load_data_generation(PARAMS_PATH)
    data_root = tmp_path / "data"
    reference_path = data_root / "reference" / "reference.csv"
    current_path = data_root / "accepted" / f"{scenario.value}.csv"
    reference_path.parent.mkdir(parents=True)
    current_path.parent.mkdir(parents=True)

    reference = generate_synthetic_dataset(
        DatasetScenario.REFERENCE,
        generation,
        contract,
    )
    current = generate_synthetic_dataset(scenario, generation, contract)
    reference.to_csv(reference_path, index=False)
    current.to_csv(current_path, index=False)

    params = yaml.safe_load(PARAMS_PATH.read_text(encoding="utf-8"))
    assert isinstance(params, dict)
    drift = params["drift"]
    assert isinstance(drift, dict)
    drift["reference_path"] = str(reference_path)
    params_path = tmp_path / "params.yaml"
    params_path.write_text(
        yaml.safe_dump(params, sort_keys=False),
        encoding="utf-8",
    )
    return params_path, current_path
