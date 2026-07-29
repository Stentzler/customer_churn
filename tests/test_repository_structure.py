from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_architecture_packages_exist() -> None:
    expected_packages = ("agent", "api", "data", "training", "workflow")

    for package_name in expected_packages:
        package_path = PROJECT_ROOT / "src" / package_name
        assert package_path.is_dir()
        assert (package_path / "__init__.py").is_file()


def test_generated_output_directories_exist() -> None:
    expected_directories = (
        "artifacts/experiment-plans",
        "artifacts/metrics",
        "artifacts/models",
        "data/accepted",
        "data/curated",
        "data/incoming",
        "data/reference",
        "data/rejected",
        "data/test",
        "reports/data-quality",
        "reports/data-profile",
        "reports/drift",
        "reports/training",
    )

    for relative_path in expected_directories:
        assert (PROJECT_ROOT / relative_path).is_dir()
