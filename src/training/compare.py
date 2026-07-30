"""Deterministic champion comparison and MLflow alias promotion.

Promotion is intentionally separate from registration. Registration gives every
selected candidate traceability; promotion decides whether that registered version
is allowed to become the serving ``champion``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import mlflow
import mlflow.sklearn as mlflow_sklearn
import pandas as pd
import yaml
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from src.api.model_loader import (
    ModelLoadError,
    PredictivePipeline,
    predict_with_pipeline,
)
from src.api.schemas import ChurnPredictionRequest
from src.data.schema import CUSTOMER_TARGET_COLUMN
from src.data.settings import load_data_contract
from src.data.validate import validate_dataframe
from src.training.evaluate import CandidateMetrics, calculate_candidate_metrics
from src.training.preprocessing import MODEL_FEATURE_COLUMNS
from src.training.registry import (
    DEFAULT_ENV_PATH,
    DEFAULT_PARAMS_PATH,
    TrackingConfigurationError,
    load_tracking_settings,
)
from src.training.registry import DEFAULT_OUTPUT_PATH as DEFAULT_TRACKING_PATH

LOGGER = logging.getLogger(__name__)
LOG_FORMAT = "%(levelname)s %(name)s %(message)s"
DEFAULT_PROMOTION_OUTPUT_PATH = Path("artifacts/metrics/promotion.json")
DEFAULT_FIXED_TEST_PATH = Path("data/test/fixed_test.csv")
CHAMPION_ALIAS = "champion"
PROMOTION_KEYS = frozenset(
    {
        "minimum_roc_auc",
        "minimum_f1",
        "minimum_recall",
        "minimum_roc_auc_improvement",
    }
)


class PromotionConfigurationError(ValueError):
    """Raised when promotion policy or candidate metadata is incomplete."""


class PromotionError(RuntimeError):
    """Raised when deterministic promotion comparison cannot complete."""


@dataclass(frozen=True, slots=True)
class PromotionPolicy:
    """Versioned quality thresholds required for champion promotion."""

    minimum_roc_auc: float
    minimum_f1: float
    minimum_recall: float
    minimum_roc_auc_improvement: float


@dataclass(frozen=True, slots=True)
class ModelMetrics:
    """Scalar metrics required by the promotion policy."""

    roc_auc: float
    f1: float
    recall: float


@dataclass(frozen=True, slots=True)
class RegisteredCandidate:
    """The registered model version being considered for promotion."""

    registered_model_name: str
    version: str


@dataclass(frozen=True, slots=True)
class RegisteredModelFacts:
    """MLflow facts used by deterministic comparison and reporting."""

    registered_model_name: str
    version: str
    run_id: str
    model_name: str
    metrics: ModelMetrics
    data_version: str | None
    git_commit: str | None


@dataclass(frozen=True, slots=True)
class FixedTestDataset:
    """Validated immutable dataset used only after candidate selection."""

    features: pd.DataFrame
    target: pd.Series
    data_version: str


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    """Pure gate result before any alias mutation is attempted."""

    passed: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PromotionResult:
    """Complete report for one comparison and optional alias update."""

    registered_model_name: str
    candidate_version: str
    selected_model: str
    previous_champion_version: str | None
    passed: bool
    promoted: bool
    reasons: tuple[str, ...]
    candidate_metrics: ModelMetrics
    champion_metrics: ModelMetrics | None
    candidate_validation_metrics: ModelMetrics
    champion_validation_metrics: ModelMetrics | None
    fixed_test_data_version: str
    api_compatible: bool
    policy: PromotionPolicy
    data_version: str | None
    git_commit: str | None
    created_at_utc: str
    output_path: Path


def load_promotion_policy(params_path: Path) -> PromotionPolicy:
    """Load and validate versioned champion-promotion thresholds."""

    try:
        parsed = yaml.safe_load(params_path.read_text(encoding="utf-8"))
        promotion = _require_mapping(parsed.get("promotion"), "promotion")
        _require_exact_keys(promotion, PROMOTION_KEYS, "promotion")
        return PromotionPolicy(
            minimum_roc_auc=_require_non_negative_number(
                promotion.get("minimum_roc_auc"),
                "promotion.minimum_roc_auc",
            ),
            minimum_f1=_require_non_negative_number(
                promotion.get("minimum_f1"),
                "promotion.minimum_f1",
            ),
            minimum_recall=_require_non_negative_number(
                promotion.get("minimum_recall"),
                "promotion.minimum_recall",
            ),
            minimum_roc_auc_improvement=_require_non_negative_number(
                promotion.get("minimum_roc_auc_improvement"),
                "promotion.minimum_roc_auc_improvement",
            ),
        )
    except (OSError, AttributeError, yaml.YAMLError) as error:
        message = f"Cannot load promotion policy from '{params_path}': {error}"
        raise PromotionConfigurationError(message) from error


def evaluate_promotion_policy(
    candidate_metrics: ModelMetrics,
    champion_metrics: ModelMetrics | None,
    policy: PromotionPolicy,
    *,
    artifact_loadable: bool,
    api_compatible: bool,
) -> PromotionDecision:
    """Evaluate promotion gates without touching MLflow state."""

    reasons: list[str] = []
    _append_minimum_reason(
        reasons,
        metric_name="roc_auc",
        actual=candidate_metrics.roc_auc,
        required=policy.minimum_roc_auc,
    )
    _append_minimum_reason(
        reasons,
        metric_name="f1",
        actual=candidate_metrics.f1,
        required=policy.minimum_f1,
    )
    _append_minimum_reason(
        reasons,
        metric_name="recall",
        actual=candidate_metrics.recall,
        required=policy.minimum_recall,
    )

    if artifact_loadable:
        reasons.append("candidate_artifact_loadable")
    else:
        reasons.append("candidate_artifact_not_loadable")

    if api_compatible:
        reasons.append("candidate_api_compatible")
    else:
        reasons.append("candidate_api_incompatible")

    if champion_metrics is None:
        reasons.append("no_champion_alias_found")
    else:
        improvement = candidate_metrics.roc_auc - champion_metrics.roc_auc
        if improvement >= policy.minimum_roc_auc_improvement:
            reasons.append(
                "roc_auc_improvement_passed "
                f"actual={improvement:.6f} "
                f"required={policy.minimum_roc_auc_improvement:.6f}"
            )
        else:
            reasons.append(
                "roc_auc_improvement_failed "
                f"actual={improvement:.6f} "
                f"required={policy.minimum_roc_auc_improvement:.6f}"
            )

    passed = (
        not any("_failed" in reason for reason in reasons)
        and "candidate_artifact_not_loadable" not in reasons
        and "candidate_api_incompatible" not in reasons
    )
    return PromotionDecision(passed=passed, reasons=tuple(reasons))


def compare_and_maybe_promote(
    *,
    tracking_uri: str,
    registered_model_name: str,
    policy: PromotionPolicy,
    tracking_path: Path,
    output_path: Path,
    fixed_test_path: Path = DEFAULT_FIXED_TEST_PATH,
    params_path: Path = DEFAULT_PARAMS_PATH,
    candidate_version: str | None = None,
    promote: bool = False,
) -> PromotionResult:
    """Compare a registered candidate with champion and optionally move alias."""

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    candidate = _resolve_candidate(
        tracking_path=tracking_path,
        registered_model_name=registered_model_name,
        candidate_version=candidate_version,
    )
    candidate_facts = _load_registered_model_facts(client, candidate)
    champion_facts = _load_champion_facts(client, registered_model_name)
    fixed_test = _load_fixed_test_dataset(fixed_test_path, params_path)
    candidate_pipeline = _load_model_pipeline(candidate_facts)
    candidate_metrics = _evaluate_fixed_test(candidate_pipeline, fixed_test)
    api_compatible = _is_api_compatible(candidate_pipeline, fixed_test)
    champion_metrics = None
    if champion_facts is not None:
        champion_pipeline = _load_model_pipeline(champion_facts)
        champion_metrics = _evaluate_fixed_test(champion_pipeline, fixed_test)
    decision = evaluate_promotion_policy(
        candidate_metrics,
        champion_metrics,
        policy,
        artifact_loadable=True,
        api_compatible=api_compatible,
    )

    promoted = False
    reasons = list(decision.reasons)
    if decision.passed and promote:
        client.set_registered_model_alias(
            name=registered_model_name,
            alias=CHAMPION_ALIAS,
            version=candidate_facts.version,
        )
        client.set_model_version_tag(
            name=registered_model_name,
            version=candidate_facts.version,
            key="candidate_status",
            value="promoted_champion",
        )
        promoted = True
        reasons.append("champion_alias_updated")
    elif decision.passed:
        reasons.append("promotion_not_requested")
    else:
        client.set_model_version_tag(
            name=registered_model_name,
            version=candidate_facts.version,
            key="candidate_status",
            value="promotion_rejected",
        )

    result = PromotionResult(
        registered_model_name=registered_model_name,
        candidate_version=candidate_facts.version,
        selected_model=candidate_facts.model_name,
        previous_champion_version=(
            champion_facts.version if champion_facts is not None else None
        ),
        passed=decision.passed,
        promoted=promoted,
        reasons=tuple(reasons),
        candidate_metrics=candidate_metrics,
        champion_metrics=champion_metrics,
        candidate_validation_metrics=candidate_facts.metrics,
        champion_validation_metrics=(
            champion_facts.metrics if champion_facts is not None else None
        ),
        fixed_test_data_version=fixed_test.data_version,
        api_compatible=api_compatible,
        policy=policy,
        data_version=candidate_facts.data_version,
        git_commit=candidate_facts.git_commit,
        created_at_utc=datetime.now(UTC).isoformat(),
        output_path=output_path,
    )
    _write_promotion_result(result)
    return result


def _append_minimum_reason(
    reasons: list[str],
    *,
    metric_name: str,
    actual: float,
    required: float,
) -> None:
    status = "passed" if actual >= required else "failed"
    reasons.append(
        f"{metric_name}_{status} actual={actual:.6f} required={required:.6f}"
    )


def _resolve_candidate(
    *,
    tracking_path: Path,
    registered_model_name: str,
    candidate_version: str | None,
) -> RegisteredCandidate:
    if candidate_version is not None:
        return RegisteredCandidate(
            registered_model_name=registered_model_name,
            version=_require_non_empty_string(candidate_version, "candidate_version"),
        )

    payload = _read_json(tracking_path)
    return RegisteredCandidate(
        registered_model_name=_require_non_empty_string(
            payload.get("registered_model_name"),
            "tracking.registered_model_name",
        ),
        version=_require_non_empty_string(
            payload.get("registered_model_version"),
            "tracking.registered_model_version",
        ),
    )


def _load_registered_model_facts(
    client: MlflowClient,
    candidate: RegisteredCandidate,
) -> RegisteredModelFacts:
    try:
        model_version = client.get_model_version(
            name=candidate.registered_model_name,
            version=candidate.version,
        )
    except MlflowException as error:
        raise PromotionError(
            "Cannot load registered candidate "
            f"{candidate.registered_model_name} version {candidate.version}"
        ) from error

    run_id = _require_model_run_id(model_version)
    run = client.get_run(run_id)
    return RegisteredModelFacts(
        registered_model_name=candidate.registered_model_name,
        version=candidate.version,
        run_id=run_id,
        model_name=_require_tag(run.data.tags, "algorithm"),
        metrics=_metrics_from_mapping(run.data.metrics, "candidate"),
        data_version=run.data.tags.get("data_version"),
        git_commit=run.data.tags.get("git_commit"),
    )


def _load_champion_facts(
    client: MlflowClient,
    registered_model_name: str,
) -> RegisteredModelFacts | None:
    try:
        model_version = client.get_model_version_by_alias(
            registered_model_name,
            CHAMPION_ALIAS,
        )
    except MlflowException:
        return None

    run_id = _require_model_run_id(model_version)
    run = client.get_run(run_id)
    return RegisteredModelFacts(
        registered_model_name=registered_model_name,
        version=model_version.version,
        run_id=run_id,
        model_name=run.data.tags.get("algorithm", "unknown"),
        metrics=_metrics_from_mapping(run.data.metrics, "champion"),
        data_version=run.data.tags.get("data_version"),
        git_commit=run.data.tags.get("git_commit"),
    )


def _load_model_pipeline(candidate: RegisteredModelFacts) -> PredictivePipeline:
    try:
        pipeline = mlflow_sklearn.load_model(
            f"models:/{candidate.registered_model_name}/{candidate.version}"
        )
    except Exception as error:
        raise PromotionError(
            "Registered model artifact cannot be loaded "
            f"model={candidate.registered_model_name} version={candidate.version}"
        ) from error
    return cast(PredictivePipeline, pipeline)


def _load_fixed_test_dataset(
    fixed_test_path: Path,
    params_path: Path,
) -> FixedTestDataset:
    if fixed_test_path.parent.name != "test":
        raise PromotionConfigurationError(
            "Promotion data must come from a directory named 'test'"
        )
    try:
        content = fixed_test_path.read_bytes()
        dataframe = pd.read_csv(fixed_test_path)
    except (OSError, pd.errors.ParserError, UnicodeError) as error:
        raise PromotionConfigurationError(
            f"Cannot read fixed-test dataset '{fixed_test_path}': {error}"
        ) from error

    validation = validate_dataframe(
        dataframe,
        load_data_contract(params_path),
        dataset_name=fixed_test_path.name,
    )
    if not validation.is_valid:
        issue_codes = ", ".join(issue.code for issue in validation.issues)
        raise PromotionConfigurationError(
            f"Fixed-test dataset is invalid: {issue_codes}"
        )
    if dataframe[CUSTOMER_TARGET_COLUMN].nunique() < 2:
        raise PromotionConfigurationError(
            "Fixed-test dataset must contain both target classes"
        )
    return FixedTestDataset(
        features=dataframe.loc[:, MODEL_FEATURE_COLUMNS],
        target=dataframe[CUSTOMER_TARGET_COLUMN],
        data_version=hashlib.sha256(content).hexdigest(),
    )


def _evaluate_fixed_test(
    pipeline: PredictivePipeline,
    fixed_test: FixedTestDataset,
) -> ModelMetrics:
    try:
        predicted = pd.Series(
            pipeline.predict(fixed_test.features),
            index=fixed_test.target.index,
        )
        probabilities = pipeline.predict_proba(fixed_test.features)
        if probabilities.ndim != 2 or probabilities.shape[1] != 2:
            raise PromotionError(
                "Fixed-test probabilities must contain two binary classes"
            )
        positive_probability = pd.Series(
            probabilities[:, 1],
            index=fixed_test.target.index,
        )
        metrics = calculate_candidate_metrics(
            fixed_test.target,
            predicted,
            positive_probability,
        )
    except PromotionError:
        raise
    except Exception as error:
        raise PromotionError("Cannot evaluate model on fixed-test data") from error
    return _promotion_metrics(metrics)


def _promotion_metrics(metrics: CandidateMetrics) -> ModelMetrics:
    return ModelMetrics(
        roc_auc=metrics.roc_auc,
        f1=metrics.f1,
        recall=metrics.recall,
    )


def _is_api_compatible(
    pipeline: PredictivePipeline,
    fixed_test: FixedTestDataset,
) -> bool:
    sample = fixed_test.features.iloc[0].to_dict()
    try:
        request = ChurnPredictionRequest.model_validate(sample)
        predict_with_pipeline(pipeline, request)
    except (ModelLoadError, ValueError, TypeError):
        return False
    return True


def _metrics_from_mapping(
    values: Mapping[str, float],
    owner: str,
) -> ModelMetrics:
    return ModelMetrics(
        roc_auc=_require_metric(values, "roc_auc", owner),
        f1=_require_metric(values, "f1", owner),
        recall=_require_metric(values, "recall", owner),
    )


def _require_metric(
    values: Mapping[str, float],
    metric_name: str,
    owner: str,
) -> float:
    value = values.get(metric_name)
    if type(value) not in {int, float}:
        raise PromotionError(f"{owner} run metric '{metric_name}' is required")
    return float(value)


def _require_model_run_id(model_version: ModelVersion) -> str:
    run_id = model_version.run_id
    if not run_id:
        raise PromotionError(
            f"Model version {model_version.name}/{model_version.version} has no run_id"
        )
    return run_id


def _require_tag(tags: Mapping[str, str], key: str) -> str:
    value = tags.get(key)
    if not value:
        raise PromotionError(f"Run tag '{key}' is required")
    return value


def _read_json(path: Path) -> dict[str, object]:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PromotionConfigurationError(
            f"Cannot read JSON '{path}': {error}"
        ) from error
    if not isinstance(parsed, dict):
        raise PromotionConfigurationError(f"JSON '{path}' must contain an object")
    return cast(dict[str, object], parsed)


def _write_promotion_result(result: PromotionResult) -> None:
    payload = {
        "api_compatible": result.api_compatible,
        "candidate_metrics": _metrics_payload(result.candidate_metrics),
        "candidate_validation_metrics": _metrics_payload(
            result.candidate_validation_metrics
        ),
        "candidate_version": result.candidate_version,
        "champion_metrics": (
            _metrics_payload(result.champion_metrics)
            if result.champion_metrics is not None
            else None
        ),
        "created_at_utc": result.created_at_utc,
        "data_version": result.data_version,
        "fixed_test_data_version": result.fixed_test_data_version,
        "git_commit": result.git_commit,
        "passed": result.passed,
        "policy": {
            "minimum_f1": result.policy.minimum_f1,
            "minimum_recall": result.policy.minimum_recall,
            "minimum_roc_auc": result.policy.minimum_roc_auc,
            "minimum_roc_auc_improvement": (result.policy.minimum_roc_auc_improvement),
        },
        "previous_champion_version": result.previous_champion_version,
        "promoted": result.promoted,
        "reasons": list(result.reasons),
        "registered_model_name": result.registered_model_name,
        "selected_model": result.selected_model,
        "champion_validation_metrics": (
            _metrics_payload(result.champion_validation_metrics)
            if result.champion_validation_metrics is not None
            else None
        ),
    }
    temporary_path = result.output_path.with_suffix(".json.tmp")
    try:
        result.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path.write_text(
            f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
            encoding="utf-8",
        )
        temporary_path.replace(result.output_path)
    except OSError as error:
        raise PromotionError(f"Cannot write promotion report: {error}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def _metrics_payload(metrics: ModelMetrics) -> dict[str, float]:
    return {
        "f1": metrics.f1,
        "recall": metrics.recall,
        "roc_auc": metrics.roc_auc,
    }


def _require_mapping(value: object, location: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise PromotionConfigurationError(f"{location} must be a mapping")
    return cast(dict[str, object], value)


def _require_exact_keys(
    mapping: Mapping[str, object],
    expected_keys: frozenset[str],
    location: str,
) -> None:
    actual_keys = set(mapping)
    missing_keys = sorted(expected_keys - actual_keys)
    unexpected_keys = sorted(actual_keys - expected_keys)
    if not missing_keys and not unexpected_keys:
        return
    details: list[str] = []
    if missing_keys:
        details.append(f"missing keys: {', '.join(missing_keys)}")
    if unexpected_keys:
        details.append(f"unexpected keys: {', '.join(unexpected_keys)}")
    raise PromotionConfigurationError(
        f"{location} has invalid fields ({'; '.join(details)})"
    )


def _require_non_negative_number(value: object, location: str) -> float:
    if type(value) not in {int, float}:
        raise PromotionConfigurationError(f"{location} must be a number")
    parsed = float(value)
    if parsed < 0:
        raise PromotionConfigurationError(f"{location} cannot be negative")
    return parsed


def _require_non_empty_string(value: object, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PromotionConfigurationError(f"{location} must be a non-empty string")
    return value.strip()


def configure_logging() -> None:
    """Configure concise default logging for direct CLI execution."""

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a registered candidate with champion and optionally promote."
        )
    )
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH)
    parser.add_argument("--env", type=Path, default=DEFAULT_ENV_PATH)
    parser.add_argument("--tracking", type=Path, default=DEFAULT_TRACKING_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_PROMOTION_OUTPUT_PATH)
    parser.add_argument(
        "--fixed-test",
        type=Path,
        default=DEFAULT_FIXED_TEST_PATH,
    )
    parser.add_argument(
        "--candidate-version",
        type=str,
        help="Registered model version to evaluate; defaults to tracking report.",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Move champion alias only if every deterministic gate passes.",
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    """Run deterministic promotion comparison from the command line."""

    args = _build_argument_parser().parse_args(arguments)
    configure_logging()
    try:
        tracking_settings = load_tracking_settings(args.params, args.env)
        policy = load_promotion_policy(args.params)
        result = compare_and_maybe_promote(
            tracking_uri=tracking_settings.tracking_uri,
            registered_model_name=tracking_settings.registered_model_name,
            policy=policy,
            tracking_path=args.tracking,
            output_path=args.output,
            fixed_test_path=args.fixed_test,
            params_path=args.params,
            candidate_version=args.candidate_version,
            promote=args.promote,
        )
    except (
        PromotionConfigurationError,
        PromotionError,
        TrackingConfigurationError,
    ) as error:
        LOGGER.error("promotion_failed reason=%s", error)
        return 1

    LOGGER.info(
        "promotion_completed model=%s candidate_version=%s passed=%s promoted=%s "
        "report=%s",
        result.registered_model_name,
        result.candidate_version,
        result.passed,
        result.promoted,
        result.output_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
