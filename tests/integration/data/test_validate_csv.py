import logging
from pathlib import Path

import pandas as pd
import pytest
from src.data.validate import (
    DataValidationOperationalError,
    main,
    validate_csv,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


def test_validate_csv_accepts_valid_data_and_writes_reports(
    tmp_path: Path,
    valid_customer_dataframe: pd.DataFrame,
) -> None:
    input_path = _write_csv(tmp_path, "valid.csv", valid_customer_dataframe)
    report_directory = tmp_path / "reports"

    result = validate_csv(input_path, PARAMS_PATH, report_directory)

    assert result.is_valid is True
    assert (report_directory / "valid.validation.json").is_file()
    assert (report_directory / "valid.validation.md").is_file()


def test_validate_csv_rejects_invalid_data_and_still_writes_reports(
    tmp_path: Path,
    valid_customer_dataframe: pd.DataFrame,
) -> None:
    dataframe = valid_customer_dataframe.copy()
    dataframe.loc[0, "region"] = "central"
    input_path = _write_csv(tmp_path, "invalid.csv", dataframe)
    report_directory = tmp_path / "reports"

    result = validate_csv(input_path, PARAMS_PATH, report_directory)

    assert result.is_valid is False
    assert (report_directory / "invalid.validation.json").is_file()
    assert (report_directory / "invalid.validation.md").is_file()


def test_validate_csv_raises_operational_error_for_missing_input(
    tmp_path: Path,
) -> None:
    with pytest.raises(DataValidationOperationalError, match="does not exist"):
        validate_csv(
            tmp_path / "missing.csv",
            PARAMS_PATH,
            tmp_path / "reports",
        )


def test_validate_csv_raises_operational_error_for_empty_input(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "empty.csv"
    input_path.write_text("", encoding="utf-8")

    with pytest.raises(DataValidationOperationalError, match="Cannot read input CSV"):
        validate_csv(input_path, PARAMS_PATH, tmp_path / "reports")


@pytest.mark.parametrize(
    ("is_valid", "expected_exit_code", "expected_level", "expected_status"),
    [
        (True, 0, logging.INFO, "validation_status=accepted"),
        (False, 1, logging.WARNING, "validation_status=rejected"),
    ],
)
def test_cli_returns_data_quality_exit_codes(
    tmp_path: Path,
    valid_customer_dataframe: pd.DataFrame,
    caplog: pytest.LogCaptureFixture,
    *,
    is_valid: bool,
    expected_exit_code: int,
    expected_level: int,
    expected_status: str,
) -> None:
    caplog.set_level(logging.INFO, logger="src.data.validate")
    dataframe = valid_customer_dataframe.copy()
    if not is_valid:
        dataframe.loc[0, "age"] = 17
    input_path = _write_csv(tmp_path, "batch.csv", dataframe)

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

    assert exit_code == expected_exit_code
    assert any(
        record.levelno == expected_level and expected_status in record.getMessage()
        for record in caplog.records
    )
    assert any(
        record.levelno == logging.INFO
        and "validation_report format=json" in record.getMessage()
        for record in caplog.records
    )


def test_cli_returns_operational_error_without_traceback(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="src.data.validate")
    exit_code = main(
        [
            "--input",
            str(tmp_path / "missing.csv"),
            "--params",
            str(PARAMS_PATH),
            "--report-dir",
            str(tmp_path / "reports"),
        ]
    )

    assert exit_code == 2
    assert any(
        record.levelno == logging.ERROR
        and "validation_operational_error" in record.getMessage()
        and "does not exist" in record.getMessage()
        for record in caplog.records
    )
    assert all(record.exc_info is None for record in caplog.records)


def _write_csv(
    tmp_path: Path,
    filename: str,
    dataframe: pd.DataFrame,
) -> Path:
    input_path = tmp_path / filename
    dataframe.to_csv(input_path, index=False)
    return input_path
