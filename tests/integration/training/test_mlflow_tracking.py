from pathlib import Path

import mlflow
import mlflow.sklearn as mlflow_sklearn
import pandas as pd
import pytest
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from src.training.preprocessing import MODEL_FEATURE_COLUMNS
from src.training.registry import TrackingSettings, track_and_register_candidates

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"
MODEL_DIRECTORY = PROJECT_ROOT / "artifacts/models"
METRICS_DIRECTORY = PROJECT_ROOT / "artifacts/metrics"
PLAN_PATH = PROJECT_ROOT / "artifacts/experiment-plans/fallback.json"
PROFILE_PATH = PROJECT_ROOT / "reports/data-profile/training.profile.json"
DRIFT_DIRECTORY = PROJECT_ROOT / "reports/drift"
CURATED_PATH = PROJECT_ROOT / "data/curated/training.csv"


@pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array has been deprecated:DeprecationWarning"
)
def test_candidates_are_tracked_and_selected_model_is_registered(
    tmp_path: Path,
) -> None:
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    settings = TrackingSettings(
        tracking_uri=tracking_uri,
        experiment_name="integration-training",
        registered_model_name="integration-customer-churn",
        artifact_root=(tmp_path / "mlartifacts").resolve().as_uri(),
    )

    result = track_and_register_candidates(
        settings,
        params_path=PARAMS_PATH,
        model_directory=MODEL_DIRECTORY,
        metrics_directory=METRICS_DIRECTORY,
        plan_path=PLAN_PATH,
        profile_path=PROFILE_PATH,
        drift_directory=DRIFT_DIRECTORY,
        output_path=tmp_path / "tracking.json",
    )

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(settings.experiment_name)
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 2
    assert {run.data.tags["algorithm"] for run in runs} == {
        "logistic_regression",
        "random_forest",
    }
    assert all("data_version" in run.data.tags for run in runs)
    assert result.selected_model == "logistic_regression"
    assert result.registered_model_version == "1"

    model_uri = (
        f"models:/{result.registered_model_name}/{result.registered_model_version}"
    )
    pipeline = mlflow_sklearn.load_model(model_uri)
    features = pd.read_csv(CURATED_PATH).loc[:4, MODEL_FEATURE_COLUMNS]
    assert pipeline.predict(features).shape == (5,)

    with pytest.raises(MlflowException):
        client.get_model_version_by_alias(
            result.registered_model_name,
            "champion",
        )
