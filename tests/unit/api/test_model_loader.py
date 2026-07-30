from src.api.model_loader import ChampionModelService, SingletonMeta
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
