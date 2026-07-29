import logging
from pathlib import Path

import pandas as pd
import pytest
from src.data.ingest import (
    BatchDisposition,
    IncomingBatchOperationalError,
    main,
    process_incoming_batch,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


def test_valid_incoming_batch_is_copied_only_to_accepted(
    tmp_path: Path,
    valid_customer_dataframe: pd.DataFrame,
) -> None:
    input_path = _write_incoming_csv(
        tmp_path,
        "normal.csv",
        valid_customer_dataframe,
    )

    result = process_incoming_batch(
        input_path,
        PARAMS_PATH,
        tmp_path,
        tmp_path / "reports",
    )

    assert result.disposition is BatchDisposition.ACCEPTED
    assert result.routed_path == tmp_path / "accepted" / "normal.csv"
    assert result.routed_path.read_bytes() == input_path.read_bytes()
    assert not (tmp_path / "rejected" / "normal.csv").exists()
    assert result.reports.json_path.is_file()
    assert result.reports.markdown_path.is_file()


def test_invalid_incoming_batch_is_copied_only_to_rejected(
    tmp_path: Path,
    valid_customer_dataframe: pd.DataFrame,
) -> None:
    dataframe = valid_customer_dataframe.copy()
    dataframe.loc[0, "region"] = "unsupported-region"
    input_path = _write_incoming_csv(tmp_path, "invalid.csv", dataframe)

    result = process_incoming_batch(
        input_path,
        PARAMS_PATH,
        tmp_path,
        tmp_path / "reports",
    )

    assert result.disposition is BatchDisposition.REJECTED
    assert result.routed_path == tmp_path / "rejected" / "invalid.csv"
    assert result.routed_path.read_bytes() == input_path.read_bytes()
    assert not (tmp_path / "accepted" / "invalid.csv").exists()
    assert result.validation.is_valid is False
    assert result.reports.json_path.is_file()
    assert result.reports.markdown_path.is_file()


def test_batch_outside_incoming_directory_is_rejected(
    tmp_path: Path,
    valid_customer_dataframe: pd.DataFrame,
) -> None:
    input_path = tmp_path / "test" / "fixed_test.csv"
    input_path.parent.mkdir(parents=True)
    valid_customer_dataframe.to_csv(input_path, index=False)

    with pytest.raises(IncomingBatchOperationalError, match="must be located"):
        process_incoming_batch(
            input_path,
            PARAMS_PATH,
            tmp_path,
            tmp_path / "reports",
        )

    assert not (tmp_path / "accepted").exists()
    assert not (tmp_path / "rejected").exists()
    assert not (tmp_path / "reports").exists()


@pytest.mark.parametrize(
    ("is_valid", "expected_exit_code", "expected_level", "disposition"),
    [
        (True, 0, logging.INFO, "accepted"),
        (False, 1, logging.WARNING, "rejected"),
    ],
)
def test_cli_returns_routing_exit_code_and_log_level(
    tmp_path: Path,
    valid_customer_dataframe: pd.DataFrame,
    caplog: pytest.LogCaptureFixture,
    *,
    is_valid: bool,
    expected_exit_code: int,
    expected_level: int,
    disposition: str,
) -> None:
    caplog.set_level(logging.INFO, logger="src.data.ingest")
    dataframe = valid_customer_dataframe.copy()
    if not is_valid:
        dataframe.loc[0, "age"] = 17
    input_path = _write_incoming_csv(tmp_path, "batch.csv", dataframe)

    exit_code = main(
        [
            "--input",
            str(input_path),
            "--params",
            str(PARAMS_PATH),
            "--data-root",
            str(tmp_path),
            "--report-dir",
            str(tmp_path / "reports"),
        ]
    )

    assert exit_code == expected_exit_code
    assert any(
        record.levelno == expected_level
        and f"disposition={disposition}" in record.getMessage()
        for record in caplog.records
    )


def _write_incoming_csv(
    data_root: Path,
    filename: str,
    dataframe: pd.DataFrame,
) -> Path:
    input_path = data_root / "incoming" / filename
    input_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(input_path, index=False)
    return input_path
