"""FastAPI application entry point for champion-model inference."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import cast

from fastapi import Depends, FastAPI, HTTPException, Request, status
from src.api.model_loader import ChampionModelService, ModelLoadError
from src.api.schemas import (
    ChurnPredictionRequest,
    ChurnPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
)
from src.api.settings import ApiConfigurationError, load_api_settings

LOGGER = logging.getLogger(__name__)
LOG_FORMAT = "%(levelname)s %(name)s %(message)s"
MODEL_SERVICE_STATE_KEY = "model_service"


def create_app(model_service: ChampionModelService | None = None) -> FastAPI:
    """Create the FastAPI app and load the champion model during lifespan startup."""

    @asynccontextmanager
    async def lifespan(app_instance: FastAPI) -> AsyncGenerator[None, None]:
        service = model_service or _build_model_service()
        try:
            LOGGER.info("Loading the model...")
            service.load()
        except ModelLoadError as error:
            LOGGER.error("model_startup_failed reason=%s", error)
            raise
        setattr(app_instance.state, MODEL_SERVICE_STATE_KEY, service)
        try:
            yield
        finally:
            if hasattr(app_instance.state, MODEL_SERVICE_STATE_KEY):
                delattr(app_instance.state, MODEL_SERVICE_STATE_KEY)

    app = FastAPI(
        title="Customer Churn API",
        version="0.1.0",
        lifespan=lifespan,
    )

    model_service_dependency = Depends(get_model_service)

    @app.get("/health", response_model=HealthResponse)
    def health(
        loaded_service: ChampionModelService = model_service_dependency,
    ) -> HealthResponse:
        """Return whether the process is alive and the model is loaded."""

        return HealthResponse(status="ok", model_loaded=loaded_service.is_loaded)

    @app.get("/model-info", response_model=ModelInfoResponse)
    def model_info(
        loaded_service: ChampionModelService = model_service_dependency,
    ) -> ModelInfoResponse:
        """Return non-sensitive metadata about the loaded champion model."""

        try:
            info = loaded_service.info
        except ModelLoadError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(error),
            ) from error
        return ModelInfoResponse(
            model_name=info.model_name,
            model_alias=info.model_alias,
            model_version=info.model_version,
            run_id=info.run_id,
            data_version=info.data_version,
            git_commit=info.git_commit,
        )

    @app.post("/predict", response_model=ChurnPredictionResponse)
    def predict(
        request: ChurnPredictionRequest,
        loaded_service: ChampionModelService = model_service_dependency,
    ) -> ChurnPredictionResponse:
        """Predict churn for one validated customer feature payload."""

        try:
            prediction = loaded_service.predict(request)
            info = loaded_service.info
        except ModelLoadError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(error),
            ) from error
        return ChurnPredictionResponse(
            predicted_class=prediction.predicted_class,
            churn_probability=prediction.churn_probability,
            model_name=info.model_name,
            model_version=info.model_version,
            model_alias=info.model_alias,
        )

    return app


def get_model_service(request: Request) -> ChampionModelService:
    """Return the model service initialized by the FastAPI lifespan hook."""

    service = getattr(request.app.state, MODEL_SERVICE_STATE_KEY, None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service is not initialized",
        )
    return cast(ChampionModelService, service)


def _build_model_service() -> ChampionModelService:
    try:
        settings = load_api_settings()
    except ApiConfigurationError as error:
        raise ModelLoadError(str(error)) from error
    return ChampionModelService(settings)


def configure_logging() -> None:
    """Configure concise logging when the module is run directly."""

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


configure_logging()
