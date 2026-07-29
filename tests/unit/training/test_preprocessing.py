from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.training.catalog import create_fallback_candidates
from src.training.preprocessing import (
    CATEGORICAL_FEATURE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    create_model_pipeline,
    create_preprocessor,
)
from src.training.settings import load_training_settings

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


def test_preprocessor_declares_numeric_and_categorical_transformations() -> None:
    preprocessor = create_preprocessor()

    assert isinstance(preprocessor, ColumnTransformer)
    assert preprocessor.transformers == [
        ("numerical", preprocessor.transformers[0][1], NUMERIC_FEATURE_COLUMNS),
        ("categorical", preprocessor.transformers[1][1], CATEGORICAL_FEATURE_COLUMNS),
    ]
    assert (
        *NUMERIC_FEATURE_COLUMNS,
        *CATEGORICAL_FEATURE_COLUMNS,
    ) == MODEL_FEATURE_COLUMNS


def test_complete_pipeline_is_unfitted_and_contains_the_candidate() -> None:
    candidate = create_fallback_candidates(load_training_settings(PARAMS_PATH))[0]

    pipeline = create_model_pipeline(candidate)

    assert isinstance(pipeline, Pipeline)
    assert tuple(pipeline.named_steps) == ("preprocessing", "classifier")
    assert pipeline.named_steps["classifier"] is candidate.estimator
    assert not hasattr(pipeline.named_steps["preprocessing"], "transformers_")


def test_preprocessor_handles_a_category_absent_during_fit() -> None:
    training_features = _feature_frame(plan_types=["basic", "premium"])
    future_features = _feature_frame(plan_types=["standard"])
    preprocessor = create_preprocessor()

    preprocessor.fit(training_features)
    transformed = preprocessor.transform(future_features)

    assert transformed.shape[0] == 1


def _feature_frame(plan_types: list[str]) -> pd.DataFrame:
    row_count = len(plan_types)
    return pd.DataFrame(
        {
            "age": [30] * row_count,
            "tenure_months": [12] * row_count,
            "monthly_spend": [80.0] * row_count,
            "support_tickets_90d": [1] * row_count,
            "late_payments_12m": [0] * row_count,
            "usage_hours_monthly": [40.0] * row_count,
            "plan_type": plan_types,
            "region": ["north"] * row_count,
        }
    )
