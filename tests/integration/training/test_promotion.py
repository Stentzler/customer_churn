import json
from pathlib import Path

import mlflow
import pytest
import yaml
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from src.agent.plan_validator import load_and_validate_experiment_plan
from src.agent.planner import build_fallback_plan, write_experiment_plan
from src.data.generate import generate_valid_customer_dataframe
from src.data.settings import load_data_contract
from src.training.artifacts import persist_training_run
from src.training.compare import (
    PromotionPolicy,
    compare_and_maybe_promote,
)
from src.training.registry import TrackingSettings, track_and_register_candidates
from src.training.settings import load_training_settings
from src.training.train import run_training_from_plan

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


@pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array has been deprecated:DeprecationWarning"
)
def test_compare_only_reports_and_promote_updates_champion_alias(
    tmp_path: Path,
) -> None:
    fixture = _build_tracking_fixture(tmp_path)
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    settings = TrackingSettings(
        tracking_uri=tracking_uri,
        experiment_name="integration-promotion",
        registered_model_name="integration-customer-churn",
        artifact_root=(tmp_path / "mlartifacts").resolve().as_uri(),
    )
    tracking = track_and_register_candidates(
        settings,
        params_path=fixture["params_path"],
        model_directory=fixture["model_directory"],
        metrics_directory=fixture["metrics_directory"],
        plan_path=fixture["plan_path"],
        profile_path=fixture["profile_path"],
        drift_directory=fixture["drift_directory"],
        output_path=fixture["tracking_path"],
    )

    report_only = compare_and_maybe_promote(
        tracking_uri=tracking_uri,
        registered_model_name=settings.registered_model_name,
        policy=_permissive_policy(),
        tracking_path=fixture["tracking_path"],
        output_path=tmp_path / "promotion-report-only.json",
        promote=False,
    )

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    assert report_only.passed is True
    assert report_only.promoted is False
    with pytest.raises(MlflowException):
        client.get_model_version_by_alias(settings.registered_model_name, "champion")

    promoted = compare_and_maybe_promote(
        tracking_uri=tracking_uri,
        registered_model_name=settings.registered_model_name,
        policy=_permissive_policy(),
        tracking_path=fixture["tracking_path"],
        output_path=tmp_path / "promotion.json",
        promote=True,
    )

    champion = client.get_model_version_by_alias(
        settings.registered_model_name,
        "champion",
    )
    report = json.loads((tmp_path / "promotion.json").read_text(encoding="utf-8"))
    assert promoted.promoted is True
    assert str(champion.version) == tracking.registered_model_version
    assert report["candidate_version"] == tracking.registered_model_version
    assert report["promoted"] is True


def _build_tracking_fixture(tmp_path: Path) -> dict[str, Path]:
    params_path = _write_temp_params(tmp_path)
    settings = load_training_settings(params_path)
    dataframe = generate_valid_customer_dataframe(
        row_count=300,
        seed=391,
        contract=load_data_contract(params_path),
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
                "data_version": "integration-promotion-data-version",
                "dataset_name": "training.csv",
            }
        ),
        encoding="utf-8",
    )
    drift_directory = tmp_path / "reports" / "drift"
    drift_directory.mkdir(parents=True)

    return {
        "drift_directory": drift_directory,
        "metrics_directory": metrics_directory,
        "model_directory": model_directory,
        "params_path": params_path,
        "plan_path": plan_path,
        "profile_path": profile_path,
        "tracking_path": tmp_path / "tracking.json",
    }


def _write_temp_params(tmp_path: Path) -> Path:
    payload = yaml.safe_load(PARAMS_PATH.read_text(encoding="utf-8"))
    payload["promotion"] = {
        "minimum_roc_auc": 0.0,
        "minimum_f1": 0.0,
        "minimum_recall": 0.0,
        "minimum_roc_auc_improvement": 0.0,
    }
    payload["registry"] = {
        "experiment_name": "integration-promotion",
        "model_name": "integration-customer-churn",
    }
    params_path = tmp_path / "params.yaml"
    params_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return params_path


def _permissive_policy() -> PromotionPolicy:
    return PromotionPolicy(
        minimum_roc_auc=0.0,
        minimum_f1=0.0,
        minimum_recall=0.0,
        minimum_roc_auc_improvement=0.0,
    )
