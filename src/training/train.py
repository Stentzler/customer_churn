"""Trusted dataset loading and leakage-safe train/validation splitting.

Candidate fitting will be added separately. Keeping data preparation independent
allows its safeguards to be tested before training, tracking, or registry concerns
are introduced.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from src.data.schema import (
    CUSTOMER_CHURN_COLUMNS,
    CUSTOMER_IDENTIFIER_COLUMN,
    CUSTOMER_TARGET_COLUMN,
)
from src.training.preprocessing import MODEL_FEATURE_COLUMNS
from src.training.settings import TrainingSettings


class TrainingDataError(ValueError):
    """Raised when curated data cannot safely be used for candidate training."""


@dataclass(frozen=True, slots=True)
class TrainingDatasetSplit:
    """Feature and target partitions used for candidate selection."""

    training_features: pd.DataFrame
    validation_features: pd.DataFrame
    training_target: pd.Series
    validation_target: pd.Series


def load_and_split_training_data(
    curated_path: Path,
    settings: TrainingSettings,
) -> TrainingDatasetSplit:
    """Load trusted curated CSV data and create deterministic partitions."""

    if curated_path.parent.name != "curated":
        message = (
            "Training data must come from a directory named 'curated'; "
            "fixed-test and incoming datasets are not training sources"
        )
        raise TrainingDataError(message)
    dataframe = _read_curated_csv(curated_path)
    return split_training_data(dataframe, settings)


def split_training_data(
    dataframe: pd.DataFrame,
    settings: TrainingSettings,
) -> TrainingDatasetSplit:
    """Split validated labeled data before any preprocessing is fitted.

    Stratification preserves the target-class proportion in both partitions. The
    customer identifier is deliberately excluded because it is an identity, not a
    behavior that should influence churn predictions.
    """

    _validate_training_dataframe(dataframe)
    features = dataframe.loc[:, MODEL_FEATURE_COLUMNS]
    target = dataframe.loc[:, CUSTOMER_TARGET_COLUMN]

    try:
        (
            training_features,
            validation_features,
            training_target,
            validation_target,
        ) = train_test_split(
            features,
            target,
            test_size=settings.validation_fraction,
            random_state=settings.random_seed,
            shuffle=True,
            stratify=target,
        )
    except ValueError as error:
        message = f"Cannot create a stratified training split: {error}"
        raise TrainingDataError(message) from error

    return TrainingDatasetSplit(
        training_features=training_features.reset_index(drop=True),
        validation_features=validation_features.reset_index(drop=True),
        training_target=training_target.reset_index(drop=True),
        validation_target=validation_target.reset_index(drop=True),
    )


def _read_curated_csv(curated_path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(curated_path)
    except FileNotFoundError as error:
        message = f"Curated training dataset does not exist: {curated_path}"
        raise TrainingDataError(message) from error
    except (OSError, pd.errors.ParserError) as error:
        message = f"Cannot read curated training dataset '{curated_path}': {error}"
        raise TrainingDataError(message) from error


def _validate_training_dataframe(dataframe: pd.DataFrame) -> None:
    actual_columns = tuple(dataframe.columns)
    if actual_columns != CUSTOMER_CHURN_COLUMNS:
        message = (
            "Curated training columns must exactly match the data contract; "
            f"expected {CUSTOMER_CHURN_COLUMNS}, received {actual_columns}"
        )
        raise TrainingDataError(message)
    if dataframe.empty:
        raise TrainingDataError("Curated training dataset must not be empty")
    if dataframe.isna().to_numpy().any():
        raise TrainingDataError("Curated training dataset must not contain null values")

    target_values = set(dataframe[CUSTOMER_TARGET_COLUMN].unique())
    if target_values != {0, 1}:
        message = (
            f"{CUSTOMER_TARGET_COLUMN} must contain both binary classes 0 and 1; "
            f"received {sorted(target_values)}"
        )
        raise TrainingDataError(message)

    # This assertion documents the safety rule even if the data schema changes.
    if CUSTOMER_IDENTIFIER_COLUMN in MODEL_FEATURE_COLUMNS:
        raise RuntimeError("Customer identifiers cannot be model features")
