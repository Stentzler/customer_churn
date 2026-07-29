"""Strict structured contracts shared by fallback and future LLM planning."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field
from src.training.settings import ModelName

type ParameterValue = int | float
type NonEmptyText = Annotated[str, Field(min_length=1, max_length=500)]


class PlanSource(StrEnum):
    """Auditable origin of an experiment plan."""

    FALLBACK = "fallback"
    LLM = "llm"


class PlannedExperiment(BaseModel):
    """One bounded candidate proposed for deterministic validation and training."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    algorithm: ModelName
    parameters: dict[str, ParameterValue]
    reason: NonEmptyText


class ExperimentPlan(BaseModel):
    """Versioned collection of candidate experiments with no executable content."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["1.0"] = "1.0"
    source: PlanSource
    primary_metric: NonEmptyText
    experiments: Annotated[tuple[PlannedExperiment, ...], Field(min_length=1)]
    observations: tuple[NonEmptyText, ...]
