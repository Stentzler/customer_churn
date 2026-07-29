import logging
from pathlib import Path

import pandas as pd
import pytest
from src.data.curate import DataCurationError, curate_training_dataset, main
from src.data.generate import generate_valid_customer_dataframe
from src.data.schema import CUSTOMER_CHURN_COLUMNS
from src.data.settings import DataContractConfig
from src.data.validate import validate_dataframe

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


def test_curation_merges_reference_and_sorted_accepted_batches(
    tmp_path: Path,
    data_contract_config: DataContractConfig,
) -> None:
    reference = _write_valid_csv(
        tmp_path,
        "reference/reference.csv",
        42,
        data_contract_config,
    )
    first_batch = _write_valid_csv(
        tmp_path,
        "accepted/2026-01.csv",
        43,
        data_contract_config,
    )
    second_path = _write_valid_csv(
        tmp_path,
        "accepted/2026-02.csv",
        44,
        data_contract_config,
    )
    second_dataframe = pd.read_csv(second_path)
    duplicated_identifier = pd.read_csv(reference).iloc[0]["customer_id"]
    second_dataframe.loc[0, "customer_id"] = duplicated_identifier
    second_dataframe.loc[0, "monthly_spend"] = 321.0
    second_dataframe.to_csv(second_path, index=False)

    result = curate_training_dataset(tmp_path, data_contract_config)
    curated = pd.read_csv(result.output_path)

    assert result.source_paths == (reference, first_batch, second_path)
    assert result.input_row_count == 150
    assert result.output_row_count == 149
    assert result.duplicate_customer_count == 1
    retained_row = curated.loc[curated["customer_id"] == duplicated_identifier].iloc[0]
    assert retained_row["monthly_spend"] == 321.0
    assert tuple(curated.columns) == CUSTOMER_CHURN_COLUMNS
    assert curated["customer_id"].is_monotonic_increasing
    assert validate_dataframe(curated, data_contract_config).is_valid is True


def test_fixed_test_data_is_never_included(
    tmp_path: Path,
    data_contract_config: DataContractConfig,
) -> None:
    _write_valid_csv(
        tmp_path,
        "reference/reference.csv",
        42,
        data_contract_config,
    )
    _write_valid_csv(
        tmp_path,
        "accepted/normal.csv",
        43,
        data_contract_config,
    )
    fixed_test_path = _write_valid_csv(
        tmp_path,
        "test/fixed_test.csv",
        44,
        data_contract_config,
    )
    fixed_test_ids = set(pd.read_csv(fixed_test_path)["customer_id"])

    result = curate_training_dataset(tmp_path, data_contract_config)
    curated_ids = set(pd.read_csv(result.output_path)["customer_id"])

    assert curated_ids.isdisjoint(fixed_test_ids)
    assert fixed_test_path not in result.source_paths


def test_invalid_accepted_source_blocks_curation(
    tmp_path: Path,
    data_contract_config: DataContractConfig,
) -> None:
    _write_valid_csv(
        tmp_path,
        "reference/reference.csv",
        42,
        data_contract_config,
    )
    invalid_path = _write_valid_csv(
        tmp_path,
        "accepted/invalid.csv",
        43,
        data_contract_config,
    )
    dataframe = pd.read_csv(invalid_path)
    dataframe.loc[0, "region"] = "unsupported-region"
    dataframe.to_csv(invalid_path, index=False)

    with pytest.raises(DataCurationError, match="is invalid"):
        curate_training_dataset(tmp_path, data_contract_config)

    assert not (tmp_path / "curated" / "training.csv").exists()


def test_curated_csv_is_reproducible(
    tmp_path: Path,
    data_contract_config: DataContractConfig,
) -> None:
    _write_valid_csv(
        tmp_path,
        "reference/reference.csv",
        42,
        data_contract_config,
    )
    _write_valid_csv(
        tmp_path,
        "accepted/normal.csv",
        43,
        data_contract_config,
    )

    first_result = curate_training_dataset(tmp_path, data_contract_config)
    first_content = first_result.output_path.read_bytes()
    second_result = curate_training_dataset(tmp_path, data_contract_config)

    assert second_result.output_path.read_bytes() == first_content


def test_cli_logs_curation_summary(
    tmp_path: Path,
    data_contract_config: DataContractConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="src.data.curate")
    _write_valid_csv(
        tmp_path,
        "reference/reference.csv",
        42,
        data_contract_config,
    )
    _write_valid_csv(
        tmp_path,
        "accepted/normal.csv",
        43,
        data_contract_config,
    )

    exit_code = main(
        [
            "--params",
            str(PARAMS_PATH),
            "--data-root",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert any(
        record.levelno == logging.INFO and "curation_completed" in record.getMessage()
        for record in caplog.records
    )


def _write_valid_csv(
    data_root: Path,
    relative_path: str,
    seed: int,
    contract: DataContractConfig,
) -> Path:
    output_path = data_root / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Fifty rows satisfy the current minimum contract while keeping tests fast.
    dataframe = generate_valid_customer_dataframe(
        row_count=50,
        seed=seed,
        contract=contract,
    )
    dataframe.to_csv(output_path, index=False)
    return output_path
