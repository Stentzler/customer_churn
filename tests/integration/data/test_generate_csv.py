import logging
from pathlib import Path

import pandas as pd
import pytest
from src.data.generate import (
    DatasetScenario,
    generated_dataset_path,
    main,
    write_synthetic_dataset,
)
from src.data.settings import (
    DataContractConfig,
    load_data_generation,
)
from src.data.validate import validate_dataframe

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


@pytest.mark.parametrize("scenario", list(DatasetScenario))
def test_write_synthetic_dataset_uses_the_expected_path(
    scenario: DatasetScenario,
    tmp_path: Path,
    data_contract_config: DataContractConfig,
) -> None:
    generation = load_data_generation(PARAMS_PATH)

    output_path = write_synthetic_dataset(
        scenario,
        generation,
        data_contract_config,
        tmp_path,
    )

    assert output_path == generated_dataset_path(scenario, tmp_path)
    assert output_path.is_file()
    dataframe = pd.read_csv(output_path)
    result = validate_dataframe(dataframe, data_contract_config)
    assert result.is_valid is (scenario is not DatasetScenario.INVALID)


def test_persisted_csv_is_reproducible(
    tmp_path: Path,
    data_contract_config: DataContractConfig,
) -> None:
    generation = load_data_generation(PARAMS_PATH)
    output_path = write_synthetic_dataset(
        DatasetScenario.REFERENCE,
        generation,
        data_contract_config,
        tmp_path,
    )
    first_content = output_path.read_bytes()

    write_synthetic_dataset(
        DatasetScenario.REFERENCE,
        generation,
        data_contract_config,
        tmp_path,
    )

    assert output_path.read_bytes() == first_content


def test_cli_generates_all_scenarios_and_logs_artifacts(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="src.data.generate")

    exit_code = main(
        [
            "--scenario",
            "all",
            "--params",
            str(PARAMS_PATH),
            "--data-root",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert all(
        generated_dataset_path(scenario, tmp_path).is_file()
        for scenario in DatasetScenario
    )
    generated_records = [
        record
        for record in caplog.records
        if "dataset_generated" in record.getMessage()
    ]
    assert len(generated_records) == len(DatasetScenario)
    assert all(record.levelno == logging.INFO for record in generated_records)


def test_cli_reports_configuration_error_without_traceback(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="src.data.generate")

    exit_code = main(
        [
            "--scenario",
            "reference",
            "--params",
            str(tmp_path / "missing.yaml"),
            "--data-root",
            str(tmp_path / "data"),
        ]
    )

    assert exit_code == 2
    assert any(
        record.levelno == logging.ERROR
        and "generation_operational_error" in record.getMessage()
        for record in caplog.records
    )
    assert all(record.exc_info is None for record in caplog.records)
