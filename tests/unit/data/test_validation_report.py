import json
from pathlib import Path

from src.data.validation_models import ValidationIssue, ValidationResult
from src.data.validation_report import (
    render_validation_json,
    render_validation_markdown,
    write_validation_reports,
)


def test_accepted_json_report_has_stable_contract() -> None:
    result = _accepted_result()

    payload = json.loads(render_validation_json(result))

    assert payload == {
        "column_count": 10,
        "dataset_name": "normal.csv",
        "is_valid": True,
        "issue_count": 0,
        "issues": [],
        "row_count": 50,
        "schema_version": "1.0",
        "status": "accepted",
    }


def test_rejected_json_report_contains_bounded_issue_details() -> None:
    result = _rejected_result()

    payload = json.loads(render_validation_json(result))

    assert payload["status"] == "rejected"
    assert payload["issue_count"] == 1
    assert payload["issues"] == [
        {
            "check": "allowed_region",
            "code": "invalid_category",
            "column": "region",
            "examples": ["row=0, value=central"],
            "failure_count": 1,
            "message": "The column contains a category outside the allowlist.",
        }
    ]


def test_markdown_report_explains_accepted_and_rejected_results() -> None:
    accepted_markdown = render_validation_markdown(_accepted_result())
    rejected_markdown = render_validation_markdown(_rejected_result())

    assert "| Status | ACCEPTED |" in accepted_markdown
    assert "No data-contract violations were found." in accepted_markdown
    assert "| Status | REJECTED |" in rejected_markdown
    assert "### 1. invalid_category" in rejected_markdown
    assert "- Column: region" in rejected_markdown
    assert "row=0, value=central" in rejected_markdown


def test_report_rendering_is_byte_deterministic() -> None:
    result = _rejected_result()

    assert render_validation_json(result) == render_validation_json(result)
    assert render_validation_markdown(result) == render_validation_markdown(result)


def test_write_reports_creates_expected_files_without_leaking_input_path(
    tmp_path: Path,
) -> None:
    result = ValidationResult.create(
        schema_version="1.0",
        dataset_name="/private/workspace/data/incoming/batch.csv",
        row_count=50,
        column_count=10,
    )

    paths = write_validation_reports(result, tmp_path / "reports")

    assert paths.json_path == tmp_path / "reports" / "batch.validation.json"
    assert paths.markdown_path == tmp_path / "reports" / "batch.validation.md"
    assert paths.json_path.is_file()
    assert paths.markdown_path.is_file()
    assert "/private/workspace" not in paths.json_path.read_text(encoding="utf-8")
    assert "/private/workspace" not in paths.markdown_path.read_text(encoding="utf-8")


def test_repeated_writes_produce_identical_bytes(tmp_path: Path) -> None:
    result = _rejected_result()
    first_paths = write_validation_reports(result, tmp_path)
    first_json = first_paths.json_path.read_bytes()
    first_markdown = first_paths.markdown_path.read_bytes()

    second_paths = write_validation_reports(result, tmp_path)

    assert second_paths.json_path.read_bytes() == first_json
    assert second_paths.markdown_path.read_bytes() == first_markdown


def _accepted_result() -> ValidationResult:
    return ValidationResult.create(
        schema_version="1.0",
        dataset_name="normal.csv",
        row_count=50,
        column_count=10,
    )


def _rejected_result() -> ValidationResult:
    issue = ValidationIssue.create(
        code="invalid_category",
        column="region",
        check="allowed_region",
        message="The column contains a category outside the allowlist.",
        failure_count=1,
        failure_examples=("row=0, value=central",),
        maximum_examples=10,
    )
    return ValidationResult.create(
        schema_version="1.0",
        dataset_name="invalid.csv",
        row_count=50,
        column_count=10,
        issues=(issue,),
    )
