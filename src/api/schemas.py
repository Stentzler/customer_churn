"""Inference API request and response contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChurnPredictionRequest(BaseModel):
    """Raw model features required by the trained preprocessing pipeline."""

    model_config = ConfigDict(extra="forbid")

    age: int = Field(ge=18, le=100)
    tenure_months: int = Field(ge=0, le=120)
    monthly_spend: float = Field(ge=0.0, le=500.0)
    support_tickets_90d: int = Field(ge=0, le=20)
    late_payments_12m: int = Field(ge=0, le=12)
    usage_hours_monthly: float = Field(ge=0.0, le=300.0)
    plan_type: Literal["basic", "standard", "premium"]
    region: Literal["north", "south", "east", "west"]


class ChurnPredictionResponse(BaseModel):
    """Prediction result plus non-sensitive model lineage."""

    predicted_class: int
    churn_probability: float | None
    model_name: str
    model_version: str | None
    model_alias: str


class HealthResponse(BaseModel):
    """Liveness and model-readiness response."""

    status: Literal["ok"]
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    """Non-sensitive metadata for the loaded serving model."""

    model_name: str
    model_alias: str
    model_version: str | None
    run_id: str | None
    data_version: str | None
    git_commit: str | None
