from dataclasses import FrozenInstanceError

import pytest
from src.data.validation_models import ValidationIssue, ValidationResult


def test_validation_issue_examples_are_deterministic_and_bounded() -> None:
    issue = _create_issue(
        failure_count=4,
        failure_examples=("west", "central", "west", "east"),
        maximum_examples=2,
    )

    assert issue.examples == ("central", "east")
    assert issue.failure_count == 4


def test_validation_issue_rejects_invalid_example_limit() -> None:
    with pytest.raises(ValueError, match="maximum_examples must be positive"):
        _create_issue(maximum_examples=0)


def test_validation_issue_rejects_invalid_failure_count() -> None:
    with pytest.raises(ValueError, match="failure_count must be positive"):
        _create_issue(failure_count=0)


def test_validation_issue_is_immutable() -> None:
    issue = _create_issue()

    with pytest.raises(FrozenInstanceError):
        issue.code = "different_code"  # type: ignore[misc]


def test_result_without_issues_is_valid() -> None:
    result = ValidationResult.create(
        schema_version="1.0",
        dataset_name="normal.csv",
        row_count=50,
        column_count=10,
    )

    assert result.is_valid is True
    assert result.issues == ()


def test_result_with_issues_is_invalid_and_deterministically_sorted() -> None:
    region_issue = _create_issue(
        code="invalid_category",
        column="region",
        check="allowed_region",
    )
    age_issue = _create_issue(
        code="invalid_range",
        column="age",
        check="age_range",
    )

    result = ValidationResult.create(
        schema_version="1.0",
        dataset_name="invalid.csv",
        row_count=50,
        column_count=10,
        issues=(age_issue, region_issue),
    )

    assert result.is_valid is False
    assert result.issues == (region_issue, age_issue)


def test_direct_result_construction_rejects_contradictory_status() -> None:
    with pytest.raises(ValueError, match="Valid results cannot have issues"):
        ValidationResult(
            schema_version="1.0",
            dataset_name="invalid.csv",
            is_valid=True,
            row_count=50,
            column_count=10,
            issues=(_create_issue(),),
        )


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [("row_count", -1), ("column_count", -1)],
)
def test_result_rejects_negative_dimensions(
    field_name: str,
    invalid_value: int,
) -> None:
    values = {
        "schema_version": "1.0",
        "dataset_name": "normal.csv",
        "row_count": 50,
        "column_count": 10,
    }
    values[field_name] = invalid_value

    with pytest.raises(ValueError, match=f"{field_name} cannot be negative"):
        ValidationResult.create(**values)  # type: ignore[arg-type]


def _create_issue(
    *,
    code: str = "invalid_category",
    column: str | None = "region",
    check: str = "allowed_region",
    failure_count: int = 1,
    failure_examples: tuple[str, ...] = ("central",),
    maximum_examples: int = 10,
) -> ValidationIssue:
    return ValidationIssue.create(
        code=code,
        column=column,
        check=check,
        message="Region contains an unsupported category.",
        failure_count=failure_count,
        failure_examples=failure_examples,
        maximum_examples=maximum_examples,
    )
