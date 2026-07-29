"""Leakage-safe preprocessing and complete model-pipeline construction.

This module only constructs unfitted scikit-learn objects. The training module is
responsible for fitting the returned pipeline with the training partition, never
with validation or fixed-test data.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.training.catalog import CandidateEstimator

NUMERIC_FEATURE_COLUMNS = (
    "age",
    "tenure_months",
    "monthly_spend",
    "support_tickets_90d",
    "late_payments_12m",
    "usage_hours_monthly",
)
CATEGORICAL_FEATURE_COLUMNS = (
    "plan_type",
    "region",
)
MODEL_FEATURE_COLUMNS = NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS


def create_preprocessor() -> ColumnTransformer:
    """Create an unfitted transformer for the documented model features.

    Numerical values are standardized so scale-sensitive estimators such as
    logistic regression receive comparable feature magnitudes. Categorical values
    are converted to binary indicator columns because scikit-learn estimators
    cannot consume strings directly.
    """

    numerical_pipeline = Pipeline(
        steps=[
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, NUMERIC_FEATURE_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURE_COLUMNS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def create_model_pipeline(candidate: CandidateEstimator) -> Pipeline:
    """Create one unfitted, serving-ready preprocessing and estimator pipeline."""

    return Pipeline(
        steps=[
            ("preprocessing", create_preprocessor()),
            ("classifier", candidate.estimator),
        ]
    )
