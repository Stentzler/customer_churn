import json
from pathlib import Path

from src.data.drift import evaluate_drift, render_drift_json
from src.data.generate import DatasetScenario, generate_synthetic_dataset
from src.data.settings import (
    DataContractConfig,
    DriftConfig,
    load_data_generation,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


def test_drifted_scenario_has_significant_feature_drift_only(
    data_contract_config: DataContractConfig,
) -> None:
    generation = load_data_generation(PARAMS_PATH)
    reference = generate_synthetic_dataset(
        DatasetScenario.REFERENCE,
        generation,
        data_contract_config,
    )
    drifted = generate_synthetic_dataset(
        DatasetScenario.DRIFTED,
        generation,
        data_contract_config,
    )

    result, _ = evaluate_drift(
        reference,
        drifted,
        DriftConfig(
            reference_path=PARAMS_PATH,
            feature_drift_share_threshold=0.25,
        ),
        reference_name="reference.csv",
        current_name="drifted.csv",
        reference_data_version="reference-hash",
        current_data_version="current-hash",
    )

    assert result.feature_drift.is_significant is True
    assert result.feature_drift.drifted_features == (
        "monthly_spend",
        "support_tickets_90d",
        "usage_hours_monthly",
    )
    assert result.feature_drift.drift_share == 3 / 8
    assert result.target_drift.column.is_drifted is False


def test_drift_json_separates_features_and_target_without_raw_rows(
    data_contract_config: DataContractConfig,
) -> None:
    generation = load_data_generation(PARAMS_PATH)
    reference = generate_synthetic_dataset(
        DatasetScenario.REFERENCE,
        generation,
        data_contract_config,
    )
    normal = generate_synthetic_dataset(
        DatasetScenario.NORMAL,
        generation,
        data_contract_config,
    )
    result, _ = evaluate_drift(
        reference,
        normal,
        DriftConfig(
            reference_path=PARAMS_PATH,
            feature_drift_share_threshold=0.25,
        ),
        reference_name="/private/reference.csv",
        current_name="/private/normal.csv",
        reference_data_version="reference-hash",
        current_data_version="current-hash",
    )

    rendered = render_drift_json(result)
    payload = json.loads(rendered)

    assert payload["reference_dataset"] == "reference.csv"
    assert payload["current_dataset"] == "normal.csv"
    assert payload["feature_drift"]["feature_count"] == 8
    assert payload["feature_drift"]["is_significant"] is False
    assert payload["target_drift"]["column"] == "churned"
    assert "customer_id" not in rendered
    assert "CUST-" not in rendered
