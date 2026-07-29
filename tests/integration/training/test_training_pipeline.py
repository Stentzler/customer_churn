from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from src.agent.plan_validator import load_and_validate_experiment_plan
from src.agent.planner import build_fallback_plan, write_experiment_plan
from src.data.generate import generate_valid_customer_dataframe
from src.data.settings import load_data_contract
from src.training.artifacts import persist_training_run
from src.training.preprocessing import NUMERIC_FEATURE_COLUMNS
from src.training.settings import load_training_settings
from src.training.train import (
    run_training_from_plan,
    split_training_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


@pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array has been deprecated:DeprecationWarning"
)
def test_curated_data_to_reloadable_prediction_artifacts(tmp_path: Path) -> None:
    settings = load_training_settings(PARAMS_PATH)
    dataframe = generate_valid_customer_dataframe(
        row_count=300,
        seed=91,
        contract=load_data_contract(PARAMS_PATH),
    )
    curated_directory = tmp_path / "curated"
    curated_directory.mkdir()
    curated_path = curated_directory / "training.csv"
    dataframe.to_csv(curated_path, index=False)

    proposed_path = write_experiment_plan(
        build_fallback_plan(settings),
        tmp_path / "plans" / "fallback.json",
    )
    approved_plan = load_and_validate_experiment_plan(proposed_path, settings)
    training_run = run_training_from_plan(curated_path, settings, approved_plan)
    artifacts = persist_training_run(
        training_run,
        tmp_path / "models",
        tmp_path / "metrics",
        approved_plan.primary_metric,
    )

    expected_split = split_training_data(dataframe, settings)
    expected_numeric_means = expected_split.training_features.loc[
        :, NUMERIC_FEATURE_COLUMNS
    ].mean()
    for model_path in artifacts.model_paths:
        pipeline = joblib.load(model_path)
        assert isinstance(pipeline, Pipeline)
        scaler = (
            pipeline.named_steps["preprocessing"]
            .named_transformers_["numerical"]
            .named_steps["scale"]
        )
        np.testing.assert_allclose(scaler.mean_, expected_numeric_means)

        predictions = pipeline.predict(expected_split.validation_features.iloc[:5])
        probabilities = pipeline.predict_proba(
            expected_split.validation_features.iloc[:5]
        )
        assert predictions.shape == (5,)
        assert probabilities.shape == (5, 2)
        assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))

    assert training_run.selected in training_run.candidates
    assert artifacts.selection_path.is_file()
    assert artifacts.failures_path.is_file()
