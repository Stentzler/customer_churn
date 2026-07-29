import json
import logging
from pathlib import Path

import pandas as pd
import pytest
from src.data.profile import DataProfileError, main, profile_csv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


def test_profile_csv_writes_deterministic_aggregate_json(
    tmp_path: Path,
    valid_customer_dataframe: pd.DataFrame,
) -> None:
    input_path = tmp_path / "training.csv"
    valid_customer_dataframe.to_csv(input_path, index=False)

    first_artifact = profile_csv(input_path, PARAMS_PATH, tmp_path / "reports")
    first_content = first_artifact.output_path.read_bytes()
    second_artifact = profile_csv(input_path, PARAMS_PATH, tmp_path / "reports")
    payload = json.loads(second_artifact.output_path.read_text(encoding="utf-8"))

    assert second_artifact.output_path.read_bytes() == first_content
    assert len(payload["data_version"]) == 64
    assert payload["dataset_name"] == "training.csv"
    assert "CUST-" not in second_artifact.output_path.read_text(encoding="utf-8")


def test_invalid_csv_is_not_profiled(
    tmp_path: Path,
    valid_customer_dataframe: pd.DataFrame,
) -> None:
    dataframe = valid_customer_dataframe.copy()
    dataframe.loc[0, "region"] = "unsupported-region"
    input_path = tmp_path / "invalid.csv"
    dataframe.to_csv(input_path, index=False)

    with pytest.raises(DataProfileError, match="Cannot profile invalid dataset"):
        profile_csv(input_path, PARAMS_PATH, tmp_path / "reports")

    assert not (tmp_path / "reports").exists()


def test_profile_cli_logs_artifact_metadata(
    tmp_path: Path,
    valid_customer_dataframe: pd.DataFrame,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="src.data.profile")
    input_path = tmp_path / "training.csv"
    valid_customer_dataframe.to_csv(input_path, index=False)

    exit_code = main(
        [
            "--input",
            str(input_path),
            "--params",
            str(PARAMS_PATH),
            "--report-dir",
            str(tmp_path / "reports"),
        ]
    )

    assert exit_code == 0
    assert any(
        record.levelno == logging.INFO and "profile_created" in record.getMessage()
        for record in caplog.records
    )
