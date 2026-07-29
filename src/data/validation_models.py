"""Stable result models for deterministic data validation.

Pandera is the validation engine, but its exceptions are a third-party integration
detail. These immutable models form the project-owned contract consumed by reports,
workflow routing, tests, and future observability code.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Self


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One normalized reason why a dataset failed the data contract.

    Attributes:
        code: Stable machine-readable identifier owned by this project.
        column: Affected column, or ``None`` for dataframe-level failures.
        check: Name of the schema check that failed.
        message: Human-readable explanation suitable for reports.
        failure_count: Total number of failures represented by this issue.
        examples: Bounded, deterministic examples; never a complete dataset.
    """

    code: str
    column: str | None
    check: str
    message: str
    failure_count: int
    examples: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Reject inconsistent models even when constructed without the factory."""

        _require_non_empty(self.code, "ValidationIssue.code")
        _require_non_empty(self.check, "ValidationIssue.check")
        _require_non_empty(self.message, "ValidationIssue.message")
        if self.column is not None:
            _require_non_empty(self.column, "ValidationIssue.column")
        if self.failure_count < 1:
            raise ValueError("ValidationIssue.failure_count must be positive")
        if any(not isinstance(example, str) for example in self.examples):
            raise ValueError("ValidationIssue.examples must contain only strings")
        if self.failure_count < len(self.examples):
            message = "ValidationIssue.failure_count cannot be less than its examples"
            raise ValueError(message)
        if self.examples != tuple(sorted(set(self.examples))):
            message = "ValidationIssue.examples must be unique and sorted"
            raise ValueError(message)

    @classmethod
    def create(
        cls,
        *,
        code: str,
        column: str | None,
        check: str,
        message: str,
        failure_count: int,
        failure_examples: Iterable[str] = (),
        maximum_examples: int,
    ) -> Self:
        """Create an issue with sorted, unique, and bounded examples.

        Args:
            code: Stable machine-readable issue identifier.
            column: Affected column, or ``None`` for a dataframe-level issue.
            check: Schema check that produced the issue.
            message: Human-readable failure explanation.
            failure_count: Total failures, including omitted examples.
            failure_examples: Potential examples collected from validation.
            maximum_examples: Maximum examples allowed into the result.

        Returns:
            An immutable, internally consistent validation issue.
        """

        if maximum_examples < 1:
            raise ValueError("maximum_examples must be positive")

        # Sorting removes dependency on dataframe/Pandera iteration order. Capping
        # prevents reports from becoming an accidental copy of the input dataset.
        examples = tuple(sorted(set(failure_examples)))[:maximum_examples]
        return cls(
            code=code,
            column=column,
            check=check,
            message=message,
            failure_count=failure_count,
            examples=examples,
        )


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Complete deterministic validation outcome for one dataset."""

    schema_version: str
    dataset_name: str
    is_valid: bool
    row_count: int
    column_count: int
    issues: tuple[ValidationIssue, ...] = ()

    def __post_init__(self) -> None:
        """Protect consumers from contradictory or nondeterministic results."""

        _require_non_empty(self.schema_version, "ValidationResult.schema_version")
        _require_non_empty(self.dataset_name, "ValidationResult.dataset_name")
        if self.row_count < 0:
            raise ValueError("ValidationResult.row_count cannot be negative")
        if self.column_count < 0:
            raise ValueError("ValidationResult.column_count cannot be negative")
        if self.is_valid == bool(self.issues):
            message = "Valid results cannot have issues; invalid results require issues"
            raise ValueError(message)
        if self.issues != tuple(sorted(self.issues, key=_issue_sort_key)):
            raise ValueError("ValidationResult.issues must use deterministic ordering")

    @classmethod
    def create(
        cls,
        *,
        schema_version: str,
        dataset_name: str,
        row_count: int,
        column_count: int,
        issues: Iterable[ValidationIssue] = (),
    ) -> Self:
        """Create a result and derive validity from canonically ordered issues."""

        ordered_issues = tuple(sorted(issues, key=_issue_sort_key))
        return cls(
            schema_version=schema_version,
            dataset_name=dataset_name,
            is_valid=not ordered_issues,
            row_count=row_count,
            column_count=column_count,
            issues=ordered_issues,
        )


def _issue_sort_key(issue: ValidationIssue) -> tuple[object, ...]:
    return (
        issue.code,
        issue.column or "",
        issue.check,
        issue.message,
        issue.examples,
    )


def _require_non_empty(value: str, location: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string")
