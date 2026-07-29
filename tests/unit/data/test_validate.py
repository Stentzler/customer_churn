from dataclasses import replace

import pandas as pd
from src.data.settings import DataContractConfig
from src.data.validate import validate_dataframe
from src.data.validation_models import ValidationIssue, ValidationResult


def test_valid_dataframe_returns_accepted_result(
    data_contract_config: DataContractConfig,
    valid_customer_dataframe: pd.DataFrame,
) -> None:
    result = validate_dataframe(
        valid_customer_dataframe,
        data_contract_config,
        dataset_name="normal.csv",
    )

    assert result.is_valid is True
    assert result.dataset_name == "normal.csv"
    assert result.row_count == 50
    assert result.column_count == 10
    assert result.issues == ()


def test_lazy_validation_collects_independent_failures(
    data_contract_config: DataContractConfig,
    valid_customer_dataframe: pd.DataFrame,
) -> None:
    dataframe = valid_customer_dataframe.copy()
    dataframe.loc[0, "age"] = 17
    dataframe.loc[1, "region"] = "central"
    dataframe.loc[2, "customer_id"] = dataframe.loc[3, "customer_id"]

    result = validate_dataframe(dataframe, data_contract_config)

    issue_codes = {issue.code for issue in result.issues}
    assert result.is_valid is False
    assert {
        "out_of_range",
        "invalid_category",
        "duplicate_customer_id",
        "tenure_inconsistent_with_age",
    } <= issue_codes


def test_structural_failures_identify_affected_columns(
    data_contract_config: DataContractConfig,
    valid_customer_dataframe: pd.DataFrame,
) -> None:
    dataframe = valid_customer_dataframe.drop(columns="region").assign(extra="value")

    result = validate_dataframe(dataframe, data_contract_config)

    issues_by_code = {issue.code: issue for issue in result.issues}
    assert issues_by_code["missing_column"].column == "region"
    assert issues_by_code["unexpected_column"].column == "extra"


def test_dataframe_failure_counts_rows_instead_of_cells(
    data_contract_config: DataContractConfig,
    valid_customer_dataframe: pd.DataFrame,
) -> None:
    dataframe = valid_customer_dataframe.copy()
    dataframe.loc[0, ["age", "tenure_months"]] = [18, 1]

    result = validate_dataframe(dataframe, data_contract_config)

    issue = _find_issue(result, "tenure_inconsistent_with_age")
    assert issue.column is None
    assert issue.failure_count == 1
    assert issue.examples == ("row=0",)


def test_dtype_failure_receives_stable_code(
    data_contract_config: DataContractConfig,
    valid_customer_dataframe: pd.DataFrame,
) -> None:
    dataframe = valid_customer_dataframe.astype({"monthly_spend": "string"})

    result = validate_dataframe(dataframe, data_contract_config)

    issue_codes = {issue.code for issue in result.issues}
    assert "invalid_dtype" in issue_codes


def test_null_failure_receives_stable_code(
    data_contract_config: DataContractConfig,
    valid_customer_dataframe: pd.DataFrame,
) -> None:
    dataframe = valid_customer_dataframe.copy()
    dataframe.loc[0, "region"] = None

    result = validate_dataframe(dataframe, data_contract_config)

    issue = _find_issue(result, "null_value")
    assert issue.examples == ("row=0, value=<null>",)


def test_column_order_is_reported_as_one_dataframe_issue(
    data_contract_config: DataContractConfig,
    valid_customer_dataframe: pd.DataFrame,
) -> None:
    dataframe = valid_customer_dataframe.loc[
        :,
        list(reversed(valid_customer_dataframe.columns)),
    ]

    result = validate_dataframe(dataframe, data_contract_config)

    issue = _find_issue(result, "incorrect_column_order")
    assert issue.column is None
    assert issue.failure_count == 10


def test_failure_examples_are_limited_by_configuration(
    data_contract_config: DataContractConfig,
    valid_customer_dataframe: pd.DataFrame,
) -> None:
    dataframe = valid_customer_dataframe.copy()
    dataframe.loc[[0, 1, 2], "region"] = ["central", "unknown", "other"]
    limited_config = replace(data_contract_config, maximum_failure_examples=2)

    result = validate_dataframe(dataframe, limited_config)

    issue = _find_issue(result, "invalid_category")
    assert issue.failure_count == 3
    assert len(issue.examples) == 2


def test_repeated_validation_returns_identical_result(
    data_contract_config: DataContractConfig,
    valid_customer_dataframe: pd.DataFrame,
) -> None:
    dataframe = valid_customer_dataframe.copy()
    dataframe.loc[0, "monthly_spend"] = 501.0

    first_result = validate_dataframe(dataframe, data_contract_config)
    second_result = validate_dataframe(dataframe, data_contract_config)

    assert first_result == second_result


def _find_issue(result: ValidationResult, code: str) -> ValidationIssue:
    return next(issue for issue in result.issues if issue.code == code)
