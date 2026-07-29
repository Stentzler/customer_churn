from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal
from src.data.schema import CUSTOMER_CHURN_COLUMNS
from src.training.preprocessing import MODEL_FEATURE_COLUMNS
from src.training.settings import TrainingSettings, load_training_settings
from src.training.train import (
    TrainingDataError,
    load_and_split_training_data,
    split_training_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


@pytest.fixture
def settings() -> TrainingSettings:
    return load_training_settings(PARAMS_PATH)


def test_split_is_deterministic_and_excludes_identifier_and_target(
    settings: TrainingSettings,
) -> None:
    dataframe = _training_dataframe()

    first = split_training_data(dataframe, settings)
    second = split_training_data(dataframe, settings)

    assert tuple(first.training_features.columns) == MODEL_FEATURE_COLUMNS
    assert "customer_id" not in first.training_features
    assert "churned" not in first.training_features
    assert len(first.training_features) == 80
    assert len(first.validation_features) == 20
    assert_frame_equal(first.training_features, second.training_features)
    assert_series_equal(first.training_target, second.training_target)


def test_split_preserves_target_ratio_and_does_not_mutate_source(
    settings: TrainingSettings,
) -> None:
    dataframe = _training_dataframe()
    original = dataframe.copy(deep=True)

    result = split_training_data(dataframe, settings)

    assert result.training_target.mean() == pytest.approx(0.5)
    assert result.validation_target.mean() == pytest.approx(0.5)
    assert_frame_equal(dataframe, original)


def test_loader_reads_curated_csv_before_splitting(
    tmp_path: Path,
    settings: TrainingSettings,
) -> None:
    curated_directory = tmp_path / "curated"
    curated_directory.mkdir()
    curated_path = curated_directory / "training.csv"
    _training_dataframe().to_csv(curated_path, index=False)

    result = load_and_split_training_data(curated_path, settings)

    assert len(result.training_features) == 80
    assert len(result.validation_features) == 20


def test_missing_curated_dataset_has_an_actionable_error(
    tmp_path: Path,
    settings: TrainingSettings,
) -> None:
    curated_directory = tmp_path / "curated"

    with pytest.raises(TrainingDataError, match="does not exist"):
        load_and_split_training_data(curated_directory / "missing.csv", settings)


def test_fixed_test_directory_cannot_be_loaded_for_training(
    tmp_path: Path,
    settings: TrainingSettings,
) -> None:
    fixed_test_path = tmp_path / "test" / "fixed_test.csv"

    with pytest.raises(TrainingDataError, match="directory named 'curated'"):
        load_and_split_training_data(fixed_test_path, settings)


def test_incorrect_columns_are_rejected(settings: TrainingSettings) -> None:
    dataframe = _training_dataframe().drop(columns="region")

    with pytest.raises(TrainingDataError, match="exactly match"):
        split_training_data(dataframe, settings)


def test_both_target_classes_are_required(settings: TrainingSettings) -> None:
    dataframe = _training_dataframe()
    dataframe["churned"] = 0

    with pytest.raises(TrainingDataError, match="both binary classes"):
        split_training_data(dataframe, settings)


def _training_dataframe(row_count: int = 100) -> pd.DataFrame:
    dataframe = pd.DataFrame(
        {
            "customer_id": [f"customer-{index:03d}" for index in range(row_count)],
            "age": [30 + index % 20 for index in range(row_count)],
            "tenure_months": [12 + index % 24 for index in range(row_count)],
            "monthly_spend": [50.0 + index % 30 for index in range(row_count)],
            "support_tickets_90d": [index % 4 for index in range(row_count)],
            "late_payments_12m": [index % 3 for index in range(row_count)],
            "usage_hours_monthly": [20.0 + index % 50 for index in range(row_count)],
            "plan_type": [
                ("basic", "standard", "premium")[index % 3]
                for index in range(row_count)
            ],
            "region": [
                ("north", "south", "east", "west")[index % 4]
                for index in range(row_count)
            ],
            "churned": [index % 2 for index in range(row_count)],
        }
    )
    return dataframe.loc[:, CUSTOMER_CHURN_COLUMNS]
