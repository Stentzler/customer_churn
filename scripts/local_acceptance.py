"""Run a local end-to-end acceptance scenario for a new incoming batch.

This script is intentionally a thin wrapper around the production pipeline. It
creates a valid synthetic delivery, lets the normal orchestrator process it, and
optionally publishes the resulting DVC cache objects to the configured remote.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from src.data.generate import (
    DatasetScenario,
    apply_feature_drift,
    generate_valid_customer_dataframe,
)
from src.data.settings import (
    DataContractConfigurationError,
    load_data_contract,
)
from src.workflow.local_pipeline import (
    DEFAULT_DATA_ROOT,
    DEFAULT_PARAMS_PATH,
    LOG_FORMAT,
    LocalPipelineError,
    LocalPipelineStatus,
    run_local_pipeline,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_LOCAL_ENV_PATH = Path(".tmp/local-acceptance.env")
DEFAULT_LOCAL_MLFLOW_BACKEND = Path(".tmp/mlflow/local-acceptance.db")
SUPPORTED_BATCH_SCENARIOS = (
    DatasetScenario.NORMAL,
    DatasetScenario.DRIFTED,
)


class LocalAcceptanceError(RuntimeError):
    """Raised when the local acceptance scenario cannot complete."""


def run_acceptance_scenario(
    *,
    rows: int,
    seed: int,
    filename: str | None,
    params_path: Path,
    data_root: Path,
    remote: bool,
    push_dvc: bool,
    scenario: DatasetScenario = DatasetScenario.NORMAL,
) -> LocalPipelineStatus:
    """Create one valid incoming batch and run it through the local lifecycle."""

    incoming_path = write_acceptance_batch(
        rows=rows,
        seed=seed,
        filename=filename,
        params_path=params_path,
        data_root=data_root,
        scenario=scenario,
    )
    env_path = Path(".env") if remote else write_local_mlflow_env()
    result = run_local_pipeline(
        incoming_path,
        params_path=params_path,
        env_path=env_path,
        data_root=data_root,
        force_retrain=True,
    )
    if result.status is LocalPipelineStatus.REJECTED:
        raise LocalAcceptanceError(
            f"Acceptance batch was rejected unexpectedly: {incoming_path}"
        )

    LOGGER.info(
        "acceptance_pipeline_completed status=%s input=%s data_version=%s "
        "model=%s model_version=%s remote=%s",
        result.status.value,
        result.input_path,
        result.data_version,
        result.registered_model_name,
        result.registered_model_version,
        remote,
    )
    if push_dvc:
        push_dvc_remote()
    return result.status


def write_acceptance_batch(
    *,
    rows: int,
    seed: int,
    filename: str | None,
    params_path: Path,
    data_root: Path,
    scenario: DatasetScenario = DatasetScenario.NORMAL,
) -> Path:
    """Generate a valid CSV delivery under ``data/incoming``.

    The generator embeds the seed in ``customer_id`` values. A unique seed
    therefore creates records that curation will append instead of deduplicating
    away as an already-seen batch.
    """

    if scenario not in SUPPORTED_BATCH_SCENARIOS:
        raise LocalAcceptanceError(
            "Acceptance batch scenario must be 'normal' or 'drifted'"
        )
    contract = load_data_contract(params_path)
    try:
        dataframe = generate_valid_customer_dataframe(
            row_count=rows,
            seed=seed,
            contract=contract,
        )
        if scenario is DatasetScenario.DRIFTED:
            dataframe = apply_feature_drift(dataframe, contract)
    except ValueError as error:
        raise LocalAcceptanceError(
            f"Invalid acceptance generation settings: {error}"
        ) from error
    batch_filename = filename or default_acceptance_filename(seed)
    if Path(batch_filename).name != batch_filename:
        raise LocalAcceptanceError("filename must not include directories")
    if not batch_filename.endswith(".csv"):
        raise LocalAcceptanceError("filename must end with .csv")

    output_path = data_root / "incoming" / batch_filename
    temporary_path = output_path.with_suffix(".csv.tmp")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(
            temporary_path,
            index=False,
            lineterminator="\n",
            float_format="%.2f",
        )
        temporary_path.replace(output_path)
    except OSError as error:
        temporary_path.unlink(missing_ok=True)
        raise LocalAcceptanceError(
            f"Cannot write acceptance batch '{output_path}': {error}"
        ) from error

    LOGGER.info(
        "acceptance_batch_created scenario=%s rows=%d seed=%d path=%s",
        scenario.value,
        len(dataframe),
        seed,
        output_path,
    )
    return output_path


def default_acceptance_filename(seed: int) -> str:
    """Build a deterministic, readable filename for one acceptance seed."""

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"acceptance-{timestamp}-{seed}.csv"


def write_local_mlflow_env(
    *,
    output_path: Path = DEFAULT_LOCAL_ENV_PATH,
    backend_path: Path = DEFAULT_LOCAL_MLFLOW_BACKEND,
) -> Path:
    """Create a local-only MLflow environment file for safe acceptance runs."""

    artifact_root = backend_path.parent / "artifacts"
    backend_uri = f"sqlite:///{backend_path.resolve()}"
    artifact_uri = artifact_root.resolve().as_uri()
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        backend_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_root.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            "\n".join(
                (
                    f"MLFLOW_TRACKING_URI={backend_uri}",
                    f"MLFLOW_ARTIFACT_ROOT={artifact_uri}",
                    "LLM_ENABLED=false",
                    "",
                )
            ),
            encoding="utf-8",
        )
    except OSError as error:
        raise LocalAcceptanceError(
            f"Cannot write local MLflow env '{output_path}': {error}"
        ) from error
    os.environ["MLFLOW_TRACKING_URI"] = backend_uri
    os.environ["MLFLOW_ARTIFACT_ROOT"] = artifact_uri
    os.environ["LLM_ENABLED"] = "false"
    return output_path


def push_dvc_remote() -> None:
    """Publish newly cached DVC objects to the configured remote storage."""

    command = ("uv", "run", "dvc", "push", "-r", "origin")
    LOGGER.info("dvc_push_started command=%s", " ".join(command))
    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise LocalAcceptanceError("DVC push failed") from error


def configure_logging() -> None:
    """Configure concise default logging for direct CLI execution."""

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a valid batch and run the local lifecycle.",
    )
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument(
        "--seed",
        type=int,
        default=int(datetime.now(UTC).strftime("%Y%m%d%H%M%S")),
    )
    parser.add_argument("--filename", type=str)
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--scenario",
        choices=[scenario.value for scenario in SUPPORTED_BATCH_SCENARIOS],
        default=DatasetScenario.NORMAL.value,
        help="Generate a normal or significantly drifted valid batch.",
    )
    parser.add_argument(
        "--remote",
        action="store_true",
        help="Use .env and real MLflow/DagsHub tracking instead of local SQLite.",
    )
    parser.add_argument(
        "--push-dvc",
        action="store_true",
        help="Push DVC cache objects to the configured DagsHub DVC remote.",
    )
    parser.add_argument(
        "--create-only",
        action="store_true",
        help="Only create the incoming CSV; do not validate, train, or register.",
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the acceptance scenario and return a shell-friendly status code."""

    args = _build_argument_parser().parse_args(arguments)
    configure_logging()
    try:
        if args.create_only:
            write_acceptance_batch(
                rows=args.rows,
                seed=args.seed,
                filename=args.filename,
                params_path=args.params,
                data_root=args.data_root,
                scenario=DatasetScenario(args.scenario),
            )
            return 0
        run_acceptance_scenario(
            rows=args.rows,
            seed=args.seed,
            filename=args.filename,
            params_path=args.params,
            data_root=args.data_root,
            remote=args.remote,
            push_dvc=args.push_dvc,
            scenario=DatasetScenario(args.scenario),
        )
    except (
        DataContractConfigurationError,
        LocalAcceptanceError,
        LocalPipelineError,
    ) as error:
        LOGGER.error("local_acceptance_failed reason=%s", error)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
