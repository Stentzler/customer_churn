"""Atomic local persistence for deterministic training outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import joblib

if TYPE_CHECKING:
    from src.training.train import TrainedCandidate, TrainingRun


class TrainingArtifactError(RuntimeError):
    """Raised when a complete training result cannot be persisted."""


@dataclass(frozen=True, slots=True)
class TrainingArtifacts:
    """Paths written for candidate pipelines, metrics, and selection."""

    model_paths: tuple[Path, ...]
    metric_paths: tuple[Path, ...]
    failures_path: Path
    selection_path: Path


def persist_training_run(
    training_run: TrainingRun,
    model_directory: Path,
    metrics_directory: Path,
    primary_metric: str,
) -> TrainingArtifacts:
    """Persist every candidate independently and record the selected model."""

    model_paths: list[Path] = []
    metric_paths: list[Path] = []
    try:
        model_directory.mkdir(parents=True, exist_ok=True)
        metrics_directory.mkdir(parents=True, exist_ok=True)
        for candidate in training_run.candidates:
            model_path = model_directory / f"{candidate.model_name.value}.joblib"
            metric_path = metrics_directory / f"{candidate.model_name.value}.json"
            _write_pipeline(candidate, model_path)
            _write_json(_candidate_payload(candidate), metric_path)
            model_paths.append(model_path)
            metric_paths.append(metric_path)

        selection_path = metrics_directory / "selection.json"
        failures_path = metrics_directory / "failures.json"
        _write_json(
            {
                "failures": [
                    {
                        "model_name": failure.model_name.value,
                        "reason": failure.reason,
                    }
                    for failure in training_run.failed_candidates
                ]
            },
            failures_path,
        )
        _write_json(
            {
                "primary_metric": primary_metric,
                "selected_model": training_run.selected.model_name.value,
                "selected_value": training_run.selected.metrics.value_for(
                    primary_metric
                ),
            },
            selection_path,
        )
    except (OSError, ValueError) as error:
        message = f"Cannot persist training artifacts: {error}"
        raise TrainingArtifactError(message) from error

    return TrainingArtifacts(
        model_paths=tuple(model_paths),
        metric_paths=tuple(metric_paths),
        failures_path=failures_path,
        selection_path=selection_path,
    )


def _candidate_payload(candidate: TrainedCandidate) -> dict[str, object]:
    metrics = candidate.metrics
    return {
        "class_distribution": {
            "churned": metrics.class_distribution.churned,
            "retained": metrics.class_distribution.retained,
        },
        "confusion_matrix": [list(row) for row in metrics.confusion_matrix],
        "f1": metrics.f1,
        "model_name": candidate.model_name.value,
        "pr_auc": metrics.pr_auc,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "roc_auc": metrics.roc_auc,
    }


def _write_pipeline(candidate: TrainedCandidate, output_path: Path) -> None:
    temporary_path = output_path.with_suffix(".joblib.tmp")
    try:
        joblib.dump(candidate.pipeline, temporary_path)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_json(payload: dict[str, object], output_path: Path) -> None:
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    try:
        content = f"{json.dumps(payload, indent=2, sort_keys=True)}\n"
        temporary_path.write_text(content, encoding="utf-8")
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
