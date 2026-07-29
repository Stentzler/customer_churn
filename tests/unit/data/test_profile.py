import json

import pandas as pd
import pytest
from src.data.profile import build_dataset_profile, render_profile_json


def test_profile_contains_required_aggregate_information(
    valid_customer_dataframe: pd.DataFrame,
) -> None:
    profile = build_dataset_profile(
        valid_customer_dataframe,
        schema_version="1.0",
        dataset_name="/private/data/training.csv",
        data_version="abc123",
    )
    payload = json.loads(render_profile_json(profile))

    assert payload["dataset_name"] == "training.csv"
    assert payload["data_version"] == "abc123"
    assert payload["row_count"] == 50
    assert payload["feature_count"] == 8
    assert payload["feature_names"] == [
        "age",
        "tenure_months",
        "monthly_spend",
        "support_tickets_90d",
        "late_payments_12m",
        "usage_hours_monthly",
        "plan_type",
        "region",
    ]
    assert payload["missing_value_counts"] == dict.fromkeys(
        valid_customer_dataframe.columns,
        0,
    )
    assert payload["duplicate_row_count"] == 0
    assert payload["target_distribution"] == {"0": 25, "1": 25}
    assert payload["drift_evaluated"] is False
    assert payload["drifted_features"] == []


def test_profile_statistics_and_frequencies_are_correct(
    valid_customer_dataframe: pd.DataFrame,
) -> None:
    profile = build_dataset_profile(
        valid_customer_dataframe,
        schema_version="1.0",
        dataset_name="training.csv",
        data_version="abc123",
    )
    payload = json.loads(render_profile_json(profile))

    age_summary = payload["numerical_summaries"]["age"]
    assert age_summary["minimum"] == 30.0
    assert age_summary["maximum"] == 69.0
    assert age_summary["mean"] == pytest.approx(46.5)
    assert payload["categorical_frequencies"]["plan_type"] == {
        "basic": 17,
        "premium": 16,
        "standard": 17,
    }


def test_profile_does_not_expose_customer_identifiers(
    valid_customer_dataframe: pd.DataFrame,
) -> None:
    rendered_profile = render_profile_json(
        build_dataset_profile(
            valid_customer_dataframe,
            schema_version="1.0",
            dataset_name="training.csv",
            data_version="abc123",
        )
    )

    assert "CUST-0000" not in rendered_profile
    assert '"customer_id"' in rendered_profile


def test_profile_rejects_unknown_drifted_features(
    valid_customer_dataframe: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="Unknown drifted features"):
        build_dataset_profile(
            valid_customer_dataframe,
            schema_version="1.0",
            dataset_name="training.csv",
            data_version="abc123",
            drifted_features=("unknown",),
        )
