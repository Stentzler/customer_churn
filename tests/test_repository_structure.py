from pathlib import Path

import yaml

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


def test_dvc_lock_contains_every_declared_stage_output() -> None:
    """Prevent CI pulls from failing because dvc.yaml and dvc.lock disagree."""

    pipeline = yaml.safe_load((PROJECT_ROOT / "dvc.yaml").read_text(encoding="utf-8"))
    lock = yaml.safe_load((PROJECT_ROOT / "dvc.lock").read_text(encoding="utf-8"))
    pipeline_stages = pipeline["stages"]
    locked_stages = lock["stages"]

    for stage_name, stage in pipeline_stages.items():
        assert stage_name in locked_stages, f"DVC stage '{stage_name}' is not locked"
        declared_outputs = {
            *stage.get("outs", ()),
            *stage.get("metrics", ()),
        }
        locked_outputs = {
            output["path"] for output in locked_stages[stage_name].get("outs", ())
        }
        assert declared_outputs <= locked_outputs, (
            f"DVC stage '{stage_name}' has outputs missing from dvc.lock: "
            f"{sorted(declared_outputs - locked_outputs)}"
        )
