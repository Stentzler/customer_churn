"""Deterministic machine-readable and human-readable validation reports.

Rendering is separated from validation so a report-format change cannot alter a
quality decision. Reports intentionally omit timestamps and absolute paths, making
identical validation results produce byte-identical artifacts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.data.validation_models import ValidationIssue, ValidationResult


@dataclass(frozen=True, slots=True)
class ValidationReportPaths:
    """Filesystem paths created for one validation result."""

    json_path: Path
    markdown_path: Path


def render_validation_json(result: ValidationResult) -> str:
    """Serialize a validation result into the stable JSON report contract.

    The payload is constructed explicitly instead of using ``dataclasses.asdict``.
    This prevents internal model changes from silently changing an external artifact
    consumed by DVC stages, CI workflows, or future workflow nodes.
    """

    payload = {
        "column_count": result.column_count,
        "dataset_name": _safe_dataset_name(result.dataset_name),
        "is_valid": result.is_valid,
        "issue_count": len(result.issues),
        "issues": [_issue_payload(issue) for issue in result.issues],
        "row_count": result.row_count,
        "schema_version": result.schema_version,
        "status": _status(result),
    }
    return f"{json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True)}\n"


def render_validation_markdown(result: ValidationResult) -> str:
    """Render a concise report for developers and pipeline reviewers."""

    lines = [
        "# Data Validation Report",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Dataset | {_escape_markdown(_safe_dataset_name(result.dataset_name))} |",
        f"| Schema version | {_escape_markdown(result.schema_version)} |",
        f"| Status | {_status(result).upper()} |",
        f"| Rows | {result.row_count} |",
        f"| Columns | {result.column_count} |",
        f"| Issues | {len(result.issues)} |",
        "",
        "## Issues",
        "",
    ]

    if result.is_valid:
        lines.append("No data-contract violations were found.")
    else:
        for issue_number, issue in enumerate(result.issues, start=1):
            lines.extend(_render_markdown_issue(issue_number, issue))

    return "\n".join(lines).rstrip() + "\n"


def write_validation_reports(
    result: ValidationResult,
    report_directory: Path,
) -> ValidationReportPaths:
    """Write deterministic JSON and Markdown reports for one result.

    Args:
        result: Project-owned validation outcome to publish.
        report_directory: Destination directory, created when it does not exist.

    Returns:
        Paths to the two written report artifacts.
    """

    report_directory.mkdir(parents=True, exist_ok=True)
    report_stem = _report_stem(result.dataset_name)
    report_paths = ValidationReportPaths(
        json_path=report_directory / f"{report_stem}.validation.json",
        markdown_path=report_directory / f"{report_stem}.validation.md",
    )
    report_paths.json_path.write_text(
        render_validation_json(result),
        encoding="utf-8",
    )
    report_paths.markdown_path.write_text(
        render_validation_markdown(result),
        encoding="utf-8",
    )
    return report_paths


def _issue_payload(issue: ValidationIssue) -> dict[str, object]:
    return {
        "check": issue.check,
        "code": issue.code,
        "column": issue.column,
        "examples": list(issue.examples),
        "failure_count": issue.failure_count,
        "message": issue.message,
    }


def _render_markdown_issue(
    issue_number: int,
    issue: ValidationIssue,
) -> list[str]:
    column = issue.column if issue.column is not None else "dataset"
    lines = [
        f"### {issue_number}. {_escape_markdown(issue.code)}",
        "",
        f"- Column: {_escape_markdown(column)}",
        f"- Check: {_escape_markdown(issue.check)}",
        f"- Message: {_escape_markdown(issue.message)}",
        f"- Failure count: {issue.failure_count}",
    ]
    if issue.examples:
        lines.append("- Examples:")
        lines.extend(f"  - {_escape_markdown(example)}" for example in issue.examples)
    lines.append("")
    return lines


def _status(result: ValidationResult) -> str:
    return "accepted" if result.is_valid else "rejected"


def _report_stem(dataset_name: str) -> str:
    safe_name = _safe_dataset_name(dataset_name)
    report_stem = Path(safe_name).stem
    if report_stem in {"", ".", ".."}:
        raise ValueError("dataset_name must contain a usable filename")
    return report_stem


def _safe_dataset_name(dataset_name: str) -> str:
    # Treat both POSIX and Windows separators as path boundaries even when reports
    # are rendered on a different operating system.
    return dataset_name.replace("\\", "/").rsplit("/", maxsplit=1)[-1]


def _escape_markdown(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")
