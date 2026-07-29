"""Executable Pandera schema for customer-churn datasets.

Pandera applies validation rules to pandas dataframes. This module only translates
the safe, typed policy from :mod:`src.data.settings` into those executable rules;
it does not read files, write reports, or decide what a workflow should do.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from src.data.settings import DataContractConfig

CUSTOMER_IDENTIFIER_COLUMN = "customer_id"
CUSTOMER_CHURN_COLUMNS = (
    CUSTOMER_IDENTIFIER_COLUMN,
    "age",
    "tenure_months",
    "monthly_spend",
    "support_tickets_90d",
    "late_payments_12m",
    "usage_hours_monthly",
    "plan_type",
    "region",
    "churned",
)


def build_customer_churn_schema(
    config: DataContractConfig,
) -> pa.DataFrameSchema:
    """Build the complete dataframe contract from validated configuration.

    The schema is strict and ordered when those policies are enabled. Type coercion
    is deliberately disabled because ingestion should expose incorrect source types
    instead of silently changing them.

    Args:
        config: Immutable ranges, categories, and dataframe-level policy.

    Returns:
        A Pandera schema ready to validate an in-memory pandas dataframe.
    """

    schema = pa.DataFrameSchema(
        columns={
            "customer_id": pa.Column(
                pa.String,
                checks=pa.Check(
                    _contains_non_whitespace,
                    name="non_empty_customer_id",
                ),
                nullable=False,
                unique=True,
                coerce=False,
            ),
            "age": _numeric_column(config, "age", pa.Int64),
            "tenure_months": _numeric_column(
                config,
                "tenure_months",
                pa.Int64,
            ),
            "monthly_spend": _numeric_column(
                config,
                "monthly_spend",
                pa.Float64,
            ),
            "support_tickets_90d": _numeric_column(
                config,
                "support_tickets_90d",
                pa.Int64,
            ),
            "late_payments_12m": _numeric_column(
                config,
                "late_payments_12m",
                pa.Int64,
            ),
            "usage_hours_monthly": _numeric_column(
                config,
                "usage_hours_monthly",
                pa.Float64,
            ),
            "plan_type": _category_column(config, "plan_type"),
            "region": _category_column(config, "region"),
            "churned": _target_column(config),
        },
        checks=[
            pa.Check(
                lambda dataframe: len(dataframe) >= config.minimum_batch_size,
                name="minimum_batch_size",
            ),
            pa.Check(
                _rows_are_unique,
                name="duplicate_rows",
            ),
            pa.Check(
                _tenure_is_consistent_with_age,
                name="tenure_consistent_with_age",
            ),
        ],
        strict=config.strict,
        ordered=config.ordered,
        coerce=False,
        unique_column_names=True,
        name="customer_churn_data",
        metadata={"schema_version": config.schema_version},
    )

    # Keep the feature contract visible and guard against accidental dictionary
    # reordering if the schema is edited in the future.
    if tuple(schema.columns) != CUSTOMER_CHURN_COLUMNS:
        message = "Customer-churn schema columns do not match the documented order"
        raise RuntimeError(message)
    return schema


def _numeric_column(
    config: DataContractConfig,
    feature_name: str,
    dtype: type[pa.Int64] | type[pa.Float64],
) -> pa.Column:
    accepted_range = config.numeric_ranges[feature_name]

    def is_in_accepted_range(series: pd.Series) -> pd.Series:
        return series.between(
            accepted_range.minimum,
            accepted_range.maximum,
            inclusive="both",
        )

    return pa.Column(
        dtype,
        checks=pa.Check(
            is_in_accepted_range,
            name=f"{feature_name}_range",
        ),
        nullable=False,
        coerce=False,
    )


def _category_column(
    config: DataContractConfig,
    feature_name: str,
) -> pa.Column:
    accepted_categories = config.allowed_categories[feature_name]

    def is_an_accepted_category(series: pd.Series) -> pd.Series:
        return series.isin(accepted_categories)

    return pa.Column(
        pa.String,
        checks=pa.Check(
            is_an_accepted_category,
            name=f"allowed_{feature_name}",
        ),
        nullable=False,
        coerce=False,
    )


def _target_column(config: DataContractConfig) -> pa.Column:
    def is_an_accepted_target(series: pd.Series) -> pd.Series:
        return series.isin(config.target_values)

    return pa.Column(
        pa.Int64,
        checks=pa.Check(
            is_an_accepted_target,
            name="allowed_target_values",
        ),
        nullable=False,
        coerce=False,
    )


def _contains_non_whitespace(series: pd.Series) -> pd.Series:
    return series.str.strip().str.len().gt(0)


def _rows_are_unique(dataframe: pd.DataFrame) -> pd.Series:
    return ~dataframe.duplicated(keep=False)


def _tenure_is_consistent_with_age(dataframe: pd.DataFrame) -> pd.Series:
    # A customer cannot have subscription tenure from before adulthood. Keeping this
    # vectorized makes the business rule efficient and identifies each failing row.
    maximum_possible_tenure = (dataframe["age"] - 18) * 12
    return dataframe["tenure_months"] <= maximum_possible_tenure
