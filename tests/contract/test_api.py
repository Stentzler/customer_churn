from dataclasses import dataclass

from fastapi.testclient import TestClient
from src.api.main import create_app
from src.api.model_loader import LoadedModelInfo, Prediction
from src.api.schemas import ChurnPredictionRequest


@dataclass
class FakeModelService:
    """Small test double that avoids MLflow network calls in API tests."""

    loaded: bool = False

    @property
    def is_loaded(self) -> bool:
        return self.loaded

    @property
    def info(self) -> LoadedModelInfo:
        return LoadedModelInfo(
            model_name="customer-churn",
            model_alias="champion",
            model_version="3",
            run_id="run-123",
            data_version="data-abc",
            git_commit="git-abc",
            model_uri="models:/customer-churn@champion",
        )

    def load(self) -> None:
        self.loaded = True

    def predict(self, request: ChurnPredictionRequest) -> Prediction:
        assert request.plan_type == "basic"
        return Prediction(predicted_class=1, churn_probability=0.82)


def test_health_reports_loaded_model() -> None:
    service = FakeModelService()

    with TestClient(create_app(service)) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_model_info_returns_non_sensitive_metadata() -> None:
    with TestClient(create_app(FakeModelService())) as client:
        response = client.get("/model-info")

    assert response.status_code == 200
    assert response.json() == {
        "model_name": "customer-churn",
        "model_alias": "champion",
        "model_version": "3",
        "run_id": "run-123",
        "data_version": "data-abc",
        "git_commit": "git-abc",
    }


def test_predict_validates_features_and_returns_prediction() -> None:
    payload = {
        "age": 35,
        "tenure_months": 12,
        "monthly_spend": 120.50,
        "support_tickets_90d": 3,
        "late_payments_12m": 1,
        "usage_hours_monthly": 45.0,
        "plan_type": "basic",
        "region": "north",
    }

    with TestClient(create_app(FakeModelService())) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json() == {
        "predicted_class": 1,
        "churn_probability": 0.82,
        "model_name": "customer-churn",
        "model_version": "3",
        "model_alias": "champion",
    }


def test_predict_rejects_unknown_fields() -> None:
    payload = {
        "age": 35,
        "tenure_months": 12,
        "monthly_spend": 120.50,
        "support_tickets_90d": 3,
        "late_payments_12m": 1,
        "usage_hours_monthly": 45.0,
        "plan_type": "basic",
        "region": "north",
        "customer_id": "CUST-001",
    }

    with TestClient(create_app(FakeModelService())) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422
