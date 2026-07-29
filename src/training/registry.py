"""MLflow experiment tracking and selected-candidate registration."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import joblib
import mlflow
import mlflow.sklearn as mlflow_sklearn
import yaml
from dotenv import load_dotenv
from mlflow import MlflowClient
from mlflow.entities import Run

LOGGER = logging.getLogger(__name__)
LOG_FORMAT = "%(levelname)s %(name)s %(message)s"
DEFAULT_PARAMS_PATH = Path("params.yaml")
DEFAULT_ENV_PATH = Path(".env")
DEFAULT_MODEL_DIRECTORY = Path("artifacts/models")
DEFAULT_METRICS_DIRECTORY = Path("artifacts/metrics")
DEFAULT_PLAN_PATH = Path("artifacts/experiment-plans/fallback.json")
DEFAULT_PROFILE_PATH = Path("reports/data-profile/training.profile.json")
DEFAULT_DRIFT_DIRECTORY = Path("reports/drift")
DEFAULT_OUTPUT_PATH = Path("artifacts/metrics/mlflow-tracking.json")


class TrackingConfigurationError(ValueError):
    """Raised when MLflow configuration is missing or unsafe."""


class ExperimentTrackingError(RuntimeError):
    """Raised when required MLflow tracking or registration fails."""


@dataclass(frozen=True, slots=True)
class TrackingSettings:
    """Validated non-secret MLflow and registry configuration."""

    tracking_uri: str
    experiment_name: str
    registered_model_name: str
    artifact_root: str | None


@dataclass(frozen=True, slots=True)
class CandidateRun:
    """Traceability identifiers for one successful candidate run."""

    model_name: str
    run_id: str


@dataclass(frozen=True, slots=True)
class TrackingResult:
    """Candidate runs and the registered selected-model version."""

    candidate_runs: tuple[CandidateRun, ...]
    selected_model: str
    registered_model_name: str
    registered_model_version: str
    output_path: Path


def load_tracking_settings(
    params_path: Path,
    env_path: Path = DEFAULT_ENV_PATH,
) -> TrackingSettings:
    """Load local secrets from ``.env`` and non-secret registry policy from YAML."""

    load_dotenv(env_path, override=False)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if not tracking_uri:
        raise TrackingConfigurationError("MLFLOW_TRACKING_URI must be configured")
    if not (
        tracking_uri.startswith(("https://", "http://", "sqlite:///"))
        or tracking_uri == "databricks"
    ):
        raise TrackingConfigurationError("MLFLOW_TRACKING_URI uses an unsupported URI")
    try:
        parsed = yaml.safe_load(params_path.read_text(encoding="utf-8"))
        registry = parsed["registry"]
        experiment_name = registry["experiment_name"]
        registered_model_name = registry["model_name"]
    except (OSError, TypeError, KeyError, yaml.YAMLError) as error:
        message = f"Cannot load registry policy from '{params_path}': {error}"
        raise TrackingConfigurationError(message) from error
    for key, value in {
        "experiment_name": experiment_name,
        "model_name": registered_model_name,
    }.items():
        if not isinstance(value, str) or not value.strip():
            raise TrackingConfigurationError(
                f"registry.{key} must be a non-empty string"
            )

    artifact_root = os.getenv("MLFLOW_ARTIFACT_ROOT", "").strip() or None
    return TrackingSettings(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name.strip(),
        registered_model_name=registered_model_name.strip(),
        artifact_root=artifact_root,
    )


def track_and_register_candidates(
    settings: TrackingSettings,
    *,
    params_path: Path,
    model_directory: Path,
    metrics_directory: Path,
    plan_path: Path,
    profile_path: Path,
    drift_directory: Path,
    output_path: Path,
) -> TrackingResult:
    """Log each candidate independently and register only the selected model."""

    try:
        selection = _read_json(metrics_directory / "selection.json")
        profile = _read_json(profile_path)
        plan = _read_json(plan_path)
        selected_model = _require_string(selection, "selected_model")
        data_version = _require_string(profile, "data_version")
        plan_source = _require_string(plan, "source")
        git_commit = _git_commit()

        mlflow.set_tracking_uri(settings.tracking_uri)
        client = MlflowClient()
        experiment_id = _get_or_create_experiment(client, settings)
        candidate_runs = []
        selected_model_version: str | None = None
        for metric_path in sorted(metrics_directory.glob("*.json")):
            if metric_path.name in {
                "failures.json",
                "mlflow-tracking.json",
                "selection.json",
            }:
                continue
            metrics = _read_json(metric_path)
            model_name = _require_string(metrics, "model_name")
            model_path = model_directory / f"{model_name}.joblib"
            run, registered_model_version = _log_candidate_run(
                experiment_id=experiment_id,
                model_name=model_name,
                model_path=model_path,
                metric_path=metric_path,
                failures_path=metrics_directory / "failures.json",
                metrics=metrics,
                params_path=params_path,
                plan_path=plan_path,
                profile_path=profile_path,
                drift_directory=drift_directory,
                git_commit=git_commit,
                data_version=data_version,
                plan_source=plan_source,
                is_selected=model_name == selected_model,
                registered_model_name=(
                    settings.registered_model_name
                    if model_name == selected_model
                    else None
                ),
            )
            candidate_runs.append(
                CandidateRun(model_name=model_name, run_id=run.info.run_id)
            )
            if model_name == selected_model:
                selected_model_version = registered_model_version

        if selected_model_version is None:
            raise ExperimentTrackingError(
                "Selected candidate was not registered as an MLflow model version"
            )
        for key, value in {
            "candidate_status": "selected_not_promoted",
            "data_version": data_version,
            "git_commit": git_commit,
        }.items():
            client.set_model_version_tag(
                name=settings.registered_model_name,
                version=selected_model_version,
                key=key,
                value=value,
            )
        result = TrackingResult(
            candidate_runs=tuple(candidate_runs),
            selected_model=selected_model,
            registered_model_name=settings.registered_model_name,
            registered_model_version=selected_model_version,
            output_path=output_path,
        )
        _write_tracking_result(result, data_version, git_commit)
        return result
    except ExperimentTrackingError:
        raise
    except Exception as error:
        message = f"MLflow tracking or registration failed: {error}"
        raise ExperimentTrackingError(message) from error


def _log_candidate_run(
    *,
    experiment_id: str,
    model_name: str,
    model_path: Path,
    metric_path: Path,
    failures_path: Path,
    metrics: Mapping[str, object],
    params_path: Path,
    plan_path: Path,
    profile_path: Path,
    drift_directory: Path,
    git_commit: str,
    data_version: str,
    plan_source: str,
    is_selected: bool,
    registered_model_name: str | None,
) -> tuple[Run, str | None]:
    pipeline = joblib.load(model_path)
    classifier = pipeline.named_steps["classifier"]
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=f"{model_name}-{data_version[:8]}",
        tags={
            "algorithm": model_name,
            "candidate_selected": str(is_selected).lower(),
            "data_version": data_version,
            "git_commit": git_commit,
            "plan_source": plan_source,
        },
    ) as active_run:
        estimator_parameters = classifier.get_params(deep=False)
        mlflow.log_params(
            {
                "algorithm": model_name,
                **{
                    key: value
                    for key, value in estimator_parameters.items()
                    if isinstance(value, (str, int, float, bool)) or value is None
                },
            }
        )
        mlflow.log_metrics(_scalar_metrics(metrics))
        mlflow.log_artifact(str(params_path), artifact_path="lineage")
        mlflow.log_artifact(str(plan_path), artifact_path="lineage")
        mlflow.log_artifact(str(profile_path), artifact_path="lineage")
        mlflow.log_artifact(str(metric_path), artifact_path="evaluation")
        mlflow.log_artifact(str(failures_path), artifact_path="evaluation")
        for drift_path in sorted(drift_directory.glob("*.drift.json")):
            mlflow.log_artifact(str(drift_path), artifact_path="lineage/drift")
        model_info = mlflow_sklearn.log_model(
            sk_model=pipeline,
            name="model",
            registered_model_name=registered_model_name,
            serialization_format="cloudpickle",
        )
        registered_version = model_info.registered_model_version
        return active_run, (
            str(registered_version) if registered_version is not None else None
        )


def _get_or_create_experiment(
    client: MlflowClient,
    settings: TrackingSettings,
) -> str:
    existing = client.get_experiment_by_name(settings.experiment_name)
    if existing is not None:
        return existing.experiment_id
    return client.create_experiment(
        settings.experiment_name,
        artifact_location=settings.artifact_root,
    )


def _scalar_metrics(payload: Mapping[str, object]) -> dict[str, float]:
    names = ("roc_auc", "pr_auc", "f1", "precision", "recall")
    return {
        name: float(cast(int | float, payload[name]))
        for name in names
        if type(payload.get(name)) in {int, float}
    }


def _read_json(path: Path) -> dict[str, object]:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        message = f"Cannot read JSON artifact '{path}': {error}"
        raise ExperimentTrackingError(message) from error
    if not isinstance(parsed, dict):
        raise ExperimentTrackingError(f"JSON artifact '{path}' must contain an object")
    return cast(dict[str, object], parsed)


def _require_string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ExperimentTrackingError(f"Tracking artifact field '{key}' is required")
    return value


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _write_tracking_result(
    result: TrackingResult,
    data_version: str,
    git_commit: str,
) -> None:
    payload = {
        "candidate_runs": [
            {"model_name": item.model_name, "run_id": item.run_id}
            for item in result.candidate_runs
        ],
        "data_version": data_version,
        "git_commit": git_commit,
        "registered_model_name": result.registered_model_name,
        "registered_model_version": result.registered_model_version,
        "selected_model": result.selected_model,
    }
    temporary_path = result.output_path.with_suffix(".json.tmp")
    try:
        result.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path.write_text(
            f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
            encoding="utf-8",
        )
        temporary_path.replace(result.output_path)
    except OSError as error:
        raise ExperimentTrackingError(
            f"Cannot persist MLflow traceability report: {error}"
        ) from error
    finally:
        temporary_path.unlink(missing_ok=True)


def configure_logging() -> None:
    """Configure concise default logging for direct CLI execution."""

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def main(arguments: Sequence[str] | None = None) -> int:
    """Track candidate artifacts and register the selected model."""

    parser = argparse.ArgumentParser(
        description="Track and register trained candidates."
    )
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH)
    parser.add_argument("--env", type=Path, default=DEFAULT_ENV_PATH)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIRECTORY)
    parser.add_argument("--metrics-dir", type=Path, default=DEFAULT_METRICS_DIRECTORY)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN_PATH)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--drift-dir", type=Path, default=DEFAULT_DRIFT_DIRECTORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args(arguments)
    configure_logging()
    try:
        settings = load_tracking_settings(args.params, args.env)
        result = track_and_register_candidates(
            settings,
            params_path=args.params,
            model_directory=args.model_dir,
            metrics_directory=args.metrics_dir,
            plan_path=args.plan,
            profile_path=args.profile,
            drift_directory=args.drift_dir,
            output_path=args.output,
        )
    except (TrackingConfigurationError, ExperimentTrackingError) as error:
        LOGGER.error("mlflow_tracking_failed reason=%s", error)
        return 1
    LOGGER.info(
        "mlflow_tracking_completed runs=%d model=%s version=%s report=%s",
        len(result.candidate_runs),
        result.registered_model_name,
        result.registered_model_version,
        result.output_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
