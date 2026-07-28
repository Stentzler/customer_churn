import pandas as pd
import pandera.pandas as pa
import pytest
from src.data.schema import CUSTOMER_CHURN_COLUMNS, build_customer_churn_schema
from src.data.settings import DataContractConfig


def test_valid_customer_dataframe_passes(
    data_contract_config: DataContractConfig,
    valid_customer_dataframe: pd.DataFrame,
) -> None:
    schema = build_customer_churn_schema(data_contract_config)

    validated_dataframe = schema.validate(valid_customer_dataframe, lazy=True)

    pd.testing.assert_frame_equal(validated_dataframe, valid_customer_dataframe)


def test_documented_boundary_values_pass(
    data_contract_config: DataContractConfig,
    valid_customer_dataframe: pd.DataFrame,
) -> None:
    dataframe = valid_customer_dataframe.copy()
    dataframe.loc[0, ["age", "tenure_months"]] = [18, 0]
    dataframe.loc[1, "age"] = 100
    dataframe.loc[2, "tenure_months"] = 0
    dataframe.loc[3, ["age", "tenure_months"]] = [100, 120]
    dataframe.loc[4, "monthly_spend"] = 0.0
    dataframe.loc[5, "monthly_spend"] = 500.0
    dataframe.loc[6, "support_tickets_90d"] = 0
    dataframe.loc[7, "support_tickets_90d"] = 20
    dataframe.loc[8, "late_payments_12m"] = 0
    dataframe.loc[9, "late_payments_12m"] = 12
    dataframe.loc[10, "usage_hours_monthly"] = 0.0
    dataframe.loc[11, "usage_hours_monthly"] = 300.0

    build_customer_churn_schema(data_contract_config).validate(
        dataframe,
        lazy=True,
    )


@pytest.mark.parametrize(
    ("feature_name", "invalid_value"),
    [
        ("age", 17),
        ("age", 101),
        ("tenure_months", -1),
        ("tenure_months", 121),
        ("monthly_spend", -0.01),
        ("monthly_spend", 500.01),
        ("support_tickets_90d", -1),
        ("support_tickets_90d", 21),
        ("late_payments_12m", -1),
        ("late_payments_12m", 13),
        ("usage_hours_monthly", -0.01),
        ("usage_hours_monthly", 300.01),
    ],
)
def test_value_outside_numeric_range_fails(
    data_contract_config: DataContractConfig,
    valid_customer_dataframe: pd.DataFrame,
    feature_name: str,
    invalid_value: int | float,
) -> None:
    dataframe = valid_customer_dataframe.copy()
    dataframe.loc[0, feature_name] = invalid_value

    _assert_schema_fails(
        data_contract_config,
        dataframe,
        expected_check=f"{feature_name}_range",
    )


@pytest.mark.parametrize(
    ("feature_name", "invalid_value", "expected_check"),
    [
        ("plan_type", "enterprise", "allowed_plan_type"),
        ("region", "central", "allowed_region"),
        ("churned", 2, "allowed_target_values"),
    ],
)
def test_unsupported_category_or_target_fails(
    data_contract_config: DataContractConfig,
    valid_customer_dataframe: pd.DataFrame,
    feature_name: str,
    invalid_value: str | int,
    expected_check: str,
) -> None:
    dataframe = valid_customer_dataframe.copy()
    dataframe.loc[0, feature_name] = invalid_value

    _assert_schema_fails(
        data_contract_config,
        dataframe,
        expected_check=expected_check,
    )


def test_missing_column_fails(
    data_contract_config: DataContractConfig,
    valid_customer_dataframe: pd.DataFrame,
) -> None:
    dataframe = valid_customer_dataframe.drop(columns="region")

    _assert_schema_fails(data_contract_config, dataframe)


def test_unexpected_column_fails_in_strict_mode(
    data_contract_config: DataContractConfig,
    valid_customer_dataframe: pd.DataFrame,
) -> None:
    dataframe = valid_customer_dataframe.assign(unexpected="value")

    _assert_schema_fails(data_contract_config, dataframe)


def test_incorrect_column_order_fails(
    data_contract_config: DataContractConfig,
    valid_customer_dataframe: pd.DataFrame,
) -> None:
    reversed_columns = list(reversed(CUSTOMER_CHURN_COLUMNS))
    dataframe = valid_customer_dataframe.loc[:, reversed_columns]

    _assert_schema_fails(data_contract_config, dataframe)


def test_incorrect_dtype_fails_without_coercion(
    data_contract_config: DataContractConfig,
    valid_customer_dataframe: pd.DataFrame,
) -> None:
    dataframe = valid_customer_dataframe.astype({"monthly_spend": "string"})

    _assert_schema_fails(data_contract_config, dataframe)


def test_null_value_fails(
    data_contract_config: DataContractConfig,
    valid_customer_dataframe: pd.DataFrame,
) -> None:
    dataframe = valid_customer_dataframe.copy()
    dataframe.loc[0, "region"] = None

    _assert_schema_fails(data_contract_config, dataframe)


def test_empty_customer_id_fails(
    data_contract_config: DataContractConfig,
    valid_customer_dataframe: pd.DataFrame,
) -> None:
    dataframe = valid_customer_dataframe.copy()
    dataframe.loc[0, "customer_id"] = " "

    _assert_schema_fails(
        data_contract_config,
        dataframe,
        expected_check="non_empty_customer_id",
    )


def test_duplicate_customer_id_fails(
    data_contract_config: DataContractConfig,
    valid_customer_dataframe: pd.DataFrame,
) -> None:
    dataframe = valid_customer_dataframe.copy()
    dataframe.loc[1, "customer_id"] = dataframe.loc[0, "customer_id"]

    _assert_schema_fails(data_contract_config, dataframe)


def test_completely_duplicated_rows_fail(
    data_contract_config: DataContractConfig,
    valid_customer_dataframe: pd.DataFrame,
) -> None:
    dataframe = pd.concat(
        [valid_customer_dataframe, valid_customer_dataframe.iloc[[0]]],
        ignore_index=True,
    )

    _assert_schema_fails(
        data_contract_config,
        dataframe,
        expected_check="duplicate_rows",
    )


def test_batch_below_minimum_size_fails(
    data_contract_config: DataContractConfig,
    valid_customer_dataframe: pd.DataFrame,
) -> None:
    dataframe = valid_customer_dataframe.iloc[:-1]

    _assert_schema_fails(
        data_contract_config,
        dataframe,
        expected_check="minimum_batch_size",
    )


def test_tenure_inconsistent_with_age_fails(
    data_contract_config: DataContractConfig,
    valid_customer_dataframe: pd.DataFrame,
) -> None:
    dataframe = valid_customer_dataframe.copy()
    dataframe.loc[0, ["age", "tenure_months"]] = [18, 1]

    _assert_schema_fails(
        data_contract_config,
        dataframe,
        expected_check="tenure_consistent_with_age",
    )


def _assert_schema_fails(
    config: DataContractConfig,
    dataframe: pd.DataFrame,
    expected_check: str | None = None,
) -> None:
    schema = build_customer_churn_schema(config)

    with pytest.raises(pa.errors.SchemaErrors) as raised_error:
        schema.validate(dataframe, lazy=True)

    if expected_check is not None:
        failed_checks = set(raised_error.value.failure_cases["check"])
        assert expected_check in failed_checks
