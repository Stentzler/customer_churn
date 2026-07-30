"""Serving configuration loaded from environment and versioned parameters."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

DEFAULT_ENV_PATH = Path(".env")
DEFAULT_PARAMS_PATH = Path("params.yaml")
DEFAULT_MODEL_ALIAS = "champion"


class ApiConfigurationError(ValueError):
    """Raised when serving configuration is missing or malformed."""


@dataclass(frozen=True, slots=True)
class ApiSettings:
    """Configuration needed to load the promoted MLflow model."""

    model_uri: str
    model_name: str
    model_alias: str
    tracking_uri: str


def load_api_settings(
    *,
    params_path: Path = DEFAULT_PARAMS_PATH,
    env_path: Path = DEFAULT_ENV_PATH,
) -> ApiSettings:
    """Load serving settings without exposing credentials in application responses."""

    load_dotenv(env_path, override=False)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if not tracking_uri:
        raise ApiConfigurationError("MLFLOW_TRACKING_URI must be configured")

    model_name = os.getenv("MODEL_NAME", "").strip() or _load_model_name(params_path)
    model_alias = os.getenv("MODEL_ALIAS", "").strip() or DEFAULT_MODEL_ALIAS
    model_uri = (
        os.getenv("MODEL_URI", "").strip() or f"models:/{model_name}@{model_alias}"
    )
    return ApiSettings(
        model_uri=model_uri,
        model_name=model_name,
        model_alias=model_alias,
        tracking_uri=tracking_uri,
    )


def _load_model_name(params_path: Path) -> str:
    try:
        parsed = yaml.safe_load(params_path.read_text(encoding="utf-8"))
        model_name = parsed["registry"]["model_name"]
    except (OSError, TypeError, KeyError, yaml.YAMLError) as error:
        raise ApiConfigurationError(
            f"Cannot load registry.model_name from '{params_path}': {error}"
        ) from error
    if not isinstance(model_name, str) or not model_name.strip():
        raise ApiConfigurationError("registry.model_name must be a non-empty string")
    return model_name.strip()
