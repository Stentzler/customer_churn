import json
from pathlib import Path

import mlflow
import mlflow.sklearn as mlflow_sklearn
import pandas as pd
import pytest
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from src.agent.plan_validator import load_and_validate_experiment_plan
from src.agent.planner import build_fallback_plan, write_experiment_plan
from src.data.generate import generate_valid_customer_dataframe
from src.data.settings import load_data_contract
from src.training.artifacts import persist_training_run
from src.training.preprocessing import MODEL_FEATURE_COLUMNS
from src.training.registry import TrackingSettings, track_and_register_candidates
from src.training.settings import load_training_settings
from src.training.train import run_training_from_plan

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


@pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array has been deprecated:DeprecationWarning"
)
def test_candidates_are_tracked_and_selected_model_is_registered(
    tmp_path: Path,
) -> None:
    fixture = _build_tracking_fixture(tmp_path)
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
        model_directory=fixture["model_directory"],
        metrics_directory=fixture["metrics_directory"],
        plan_path=fixture["plan_path"],
        profile_path=fixture["profile_path"],
        drift_directory=fixture["drift_directory"],
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
    features = pd.read_csv(fixture["curated_path"]).loc[:4, MODEL_FEATURE_COLUMNS]
    assert pipeline.predict(features).shape == (5,)

    with pytest.raises(MlflowException):
        client.get_model_version_by_alias(
            result.registered_model_name,
            "champion",
        )


def _build_tracking_fixture(tmp_path: Path) -> dict[str, Path]:
    """Create all tracking inputs without depending on DVC-managed repo outputs."""

    settings = load_training_settings(PARAMS_PATH)
    dataframe = generate_valid_customer_dataframe(
        row_count=300,
        seed=191,
        contract=load_data_contract(PARAMS_PATH),
    )
    curated_directory = tmp_path / "curated"
    curated_directory.mkdir()
    curated_path = curated_directory / "training.csv"
    dataframe.to_csv(curated_path, index=False)

    plan_path = write_experiment_plan(
        build_fallback_plan(settings),
        tmp_path / "plans" / "fallback.json",
    )
    approved_plan = load_and_validate_experiment_plan(plan_path, settings)
    training_run = run_training_from_plan(curated_path, settings, approved_plan)
    model_directory = tmp_path / "models"
    metrics_directory = tmp_path / "metrics"
    persist_training_run(
        training_run,
        model_directory,
        metrics_directory,
        approved_plan.primary_metric,
    )

    profile_path = tmp_path / "reports" / "data-profile" / "training.profile.json"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text(
        json.dumps(
            {
                "data_version": "integration-data-version",
                "dataset_name": "training.csv",
            }
        ),
        encoding="utf-8",
    )
    drift_directory = tmp_path / "reports" / "drift"
    drift_directory.mkdir(parents=True)

    return {
        "curated_path": curated_path,
        "drift_directory": drift_directory,
        "metrics_directory": metrics_directory,
        "model_directory": model_directory,
        "plan_path": plan_path,
        "profile_path": profile_path,
    }
