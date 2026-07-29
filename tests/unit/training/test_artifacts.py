import json
from pathlib import Path

import joblib
import pytest
from sklearn.pipeline import Pipeline
from src.training.artifacts import persist_training_run
from src.training.settings import load_training_settings
from src.training.train import run_training

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"
CURATED_PATH = PROJECT_ROOT / "data/curated/training.csv"


@pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array has been deprecated:DeprecationWarning"
)
def test_persist_training_run_writes_independent_loadable_artifacts(
    tmp_path: Path,
) -> None:
    settings = load_training_settings(PARAMS_PATH)
    training_run = run_training(CURATED_PATH, settings)

    artifacts = persist_training_run(
        training_run,
        tmp_path / "models",
        tmp_path / "metrics",
        settings.primary_metric,
    )

    assert len(artifacts.model_paths) == 2
    assert len(artifacts.metric_paths) == 2
    assert all(
        isinstance(joblib.load(path), Pipeline) for path in artifacts.model_paths
    )
    assert {
        json.loads(path.read_text(encoding="utf-8"))["model_name"]
        for path in artifacts.metric_paths
    } == {"logistic_regression", "random_forest"}
    selection = json.loads(artifacts.selection_path.read_text(encoding="utf-8"))
    assert selection["selected_model"] == training_run.selected.model_name.value
    assert selection["primary_metric"] == "roc_auc"
