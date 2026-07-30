import numpy as np
import pandas as pd
import pytest
from src.api.model_loader import (
    ChampionModelService,
    ModelLoadError,
    SingletonMeta,
    predict_with_pipeline,
)
from src.api.schemas import ChurnPredictionRequest
from src.api.settings import ApiSettings


def test_champion_model_service_is_lazy_singleton() -> None:
    """Repeated construction returns the same process-local service object."""

    SingletonMeta._instances.pop(ChampionModelService, None)
    try:
        settings = ApiSettings(
            model_uri="models:/customer-churn@champion",
            model_name="customer-churn",
            model_alias="champion",
            tracking_uri="sqlite:///tmp/mlflow.db",
        )

        first_service = ChampionModelService(settings)
        second_service = ChampionModelService(settings)

        assert first_service is second_service
    finally:
        SingletonMeta._instances.pop(ChampionModelService, None)


def test_predict_with_pipeline_enforces_api_output_contract() -> None:
    request = ChurnPredictionRequest(
        age=35,
        tenure_months=12,
        monthly_spend=120.5,
        support_tickets_90d=3,
        late_payments_12m=1,
        usage_hours_monthly=45.0,
        plan_type="basic",
        region="north",
    )

    prediction = predict_with_pipeline(_ValidPipeline(), request)

    assert prediction.predicted_class == 1
    assert prediction.churn_probability == pytest.approx(0.75)


def test_predict_with_pipeline_rejects_invalid_probability_shape() -> None:
    request = ChurnPredictionRequest(
        age=35,
        tenure_months=12,
        monthly_spend=120.5,
        support_tickets_90d=3,
        late_payments_12m=1,
        usage_hours_monthly=45.0,
        plan_type="basic",
        region="north",
    )

    with pytest.raises(ModelLoadError, match="two binary classes"):
        predict_with_pipeline(_InvalidProbabilityPipeline(), request)


class _ValidPipeline:
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.array([1])

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return np.array([[0.25, 0.75]])


class _InvalidProbabilityPipeline(_ValidPipeline):
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return np.array([[0.75]])
