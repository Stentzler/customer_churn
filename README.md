# Customer Churn Agentic MLOps

A learning project for an automated and auditable customer-churn lifecycle spanning
DataOps, MLOps, and constrained LLMOps.

The project is being implemented incrementally according to [SPEC.md](./SPEC.md).
The current implementation demonstrates the main local and GitHub-driven
operational cycle: deterministic DataOps, candidate training, MLflow/DagsHub
registration, controlled promotion, FastAPI serving, and optional constrained
agent assistance.

Detailed subsystem documentation:

- [DataOps guide](./src/data/README.md)
- [Training and MLflow guide](./src/training/README.md)
- [FastAPI serving guide](./src/api/README.md)
- [Agent and LLMOps guide](./src/agent/README.md)
- [Workflow orchestration guide](./src/workflow/README.md)

## Current Lifecycle

The project is organized as an auditable lifecycle rather than only a model
training script:

```text
new CSV
  |
  v
validate contract
  |
  v
accept or reject
  |
  v
feature and target drift reports
  |
  +-- no significant feature drift --> persist reports and stop
  |
  +-- significant drift or manual force
  |
  v
curate and profile training dataset
  |
  v
optional LLM experiment plan
  |
  v
deterministic candidate training and selection
  |
  v
MLflow/DagsHub registration
  |
  v
fixed-test and API-compatible comparison
  |
  +-- candidate passes --> eligible for manual promotion
  |
  v
manual deterministic promotion to champion
  |
  v
FastAPI serves models:/customer-churn@champion
```

The important rule is that the LLM never owns operational decisions. It can
propose bounded experiments and write an analysis report, but deterministic code
validates data, validates the plan, trains candidates, selects the winner,
registers the model, and checks promotion gates.

Use these guides for subsystem-level details:

| Area | Guide |
|---|---|
| Data generation, validation, curation, profiling, drift, and DVC | [DataOps guide](./src/data/README.md) |
| Training, model artifacts, metrics, selection, MLflow, and promotion | [Training guide](./src/training/README.md) |
| Optional LLM planning and analysis | [Agent guide](./src/agent/README.md) |
| Local model serving | [FastAPI guide](./src/api/README.md) |
| Local and GitHub lifecycle orchestration | [Workflow guide](./src/workflow/README.md) |

## Development

Prerequisites:

- Python 3.12
- [uv](https://docs.astral.sh/uv/)

Install the locked environment and run the baseline checks:

```bash
make install
make lint
make test
```

Generate all synthetic DataOps scenarios:

```bash
make generate-data
```

This writes the reference dataset to `data/reference/`, the immutable evaluation
dataset to `data/test/`, and normal, drifted, and invalid deliveries to
`data/incoming/`. These generated CSV files are ignored by Git and will be versioned
with DVC in the data-pipeline phase.

Validate and route one generated delivery:

```bash
make process-batch INPUT=data/incoming/normal.csv
```

Accepted batches are copied to `data/accepted/`. Contract-breaking batches are
copied to `data/rejected/` and return exit code `1`, preventing later pipeline stages
from treating rejection as success. Both paths produce JSON and Markdown reports
under `reports/data-quality/`.

Build the training dataset from the reference dataset and all accepted batches:

```bash
make curate-data
```

Curation reads only `data/reference/reference.csv` and sorted CSV files directly
under `data/accepted/`. Duplicate customer identifiers keep the row from the last
accepted filename, and the fixed test dataset is excluded by the curation interface.

Create the aggregate profile for the current curated training dataset:

```bash
make profile-data
```

The deterministic JSON report under `reports/data-profile/` contains schema,
missing-value, distribution, summary-statistic, duplicate, and data-version
information. It never contains customer rows or identifier values.

Compare one accepted batch with the fixed reference dataset:

```bash
make drift-data CURRENT=data/accepted/normal.csv
```

Feature drift is evaluated independently from target drift. The command writes a
stable structured JSON decision and an Evidently HTML visualization under
`reports/drift/`.

Run the complete local DataOps pipeline with DVC:

```bash
make data-pipeline
```

DVC records code, data, and parameter dependencies in `dvc.yaml` and resolved
content hashes in `dvc.lock`. Generated datasets and reports are stored in the local
`.dvc/cache/`; no account or remote storage is required. Repeating the command
reuses unchanged stages.

Run one newly delivered CSV through the complete local lifecycle:

```bash
make pipeline INPUT=data/incoming/batch-001.csv
```

The command validates and routes the batch, creates drift reports, and versions
the batch artifacts in the local DVC cache. Significant feature drift continues
through curation, profiling, bounded experiment planning, training, DagsHub
registration, fixed-test comparison, and final agent analysis. Invalid data stops
before curation, and valid data without significant feature drift stops after its
reports are persisted.

Use the explicit override only when a human intentionally wants to retrain despite
the drift result:

```bash
make pipeline INPUT=data/incoming/batch-001.csv FORCE_RETRAIN=1
```

When the resulting curated `data_version` was already registered by the previous
successful execution, remote tracking is skipped to avoid duplicate model versions.

## Local End-to-End Pipeline

This repository uses four different systems together:

| System | What it stores | What it is used for |
|---|---|---|
| Git / GitHub | Source code, configs, tests, docs, `dvc.yaml`, `dvc.lock`, and small `.dvc` pointer files | Code review, history, collaboration, and GitHub Actions automation |
| DVC | Declared pipeline outputs and files explicitly added with `dvc add`, stored in `.dvc/cache/` or the DagsHub DVC remote | Versioning large or generated files without committing them directly to Git |
| MLflow | Experiment runs, metrics, parameters, artifacts, and registered model versions | Tracking what was trained and keeping model lineage |
| DagsHub | Provides DVC remote storage and MLflow tracking/registry linked to the project | One place to inspect data versions, experiment runs, and registered models |

The fastest way to simulate a new delivery is to let the project create a fresh
valid CSV and run the same local lifecycle that GitHub Actions will use later:

```bash
make acceptance-local
```

This creates a new file under `data/incoming/`, validates it, versions the batch
artifacts with DVC, rebuilds the curated training dataset, profiles the data,
trains candidates, registers the selected model in a local SQLite MLflow backend,
compares that version, and writes the final agent report. It deliberately forces
retraining so the acceptance command always exercises the full lifecycle. It does
not contact DagsHub.

Use this command when you want a complete local rehearsal without remote writes.
It is useful for studying the behavior because it leaves the generated DVC
metadata, reports, metrics, selected model artifact, and agent analysis on disk.

After this command, Git will usually show new files like:

```text
data/incoming/<batch>.csv.dvc
data/accepted/<batch>.csv.dvc
reports/data-quality/<batch>.validation.json.dvc
reports/data-quality/<batch>.validation.md.dvc
reports/drift/<batch>.drift.json.dvc
reports/drift/<batch>.drift.html.dvc
dvc.lock
```

These are metadata files. Git stores these small files, not the real CSV or HTML
contents. The real files stay ignored by Git and are stored by DVC in `.dvc/cache/`.

The batch flow is:

1. The raw generated delivery is written to `data/incoming/<batch>.csv`.
2. Validation accepts it and copies it to `data/accepted/<batch>.csv`.
3. DVC creates `.dvc` pointer files for the raw batch, accepted batch, validation
   reports, and drift reports.
4. Curation rebuilds `data/curated/training.csv` from the reference data plus all
   accepted batches.
5. DVC updates `dvc.lock` with the new hashes for curated data, profile, training
   metrics, and model artifacts.

You can confirm curation and training happened by checking:

```bash
ls -lh data/curated/training.csv
ls -lh artifacts/models/
cat artifacts/metrics/selection.json
cat artifacts/metrics/logistic_regression.json
cat artifacts/metrics/random_forest.json
cat artifacts/agent/agent-analysis.md
```

The main training result is `artifacts/metrics/selection.json`. It tells which
candidate won according to the configured primary metric. Each candidate also gets
its own metrics JSON with ROC-AUC, PR-AUC, F1, precision, recall, confusion matrix,
and class distribution.

The agent analysis report is written to `artifacts/agent/agent-analysis.md`. It
explains which plan was used, whether fallback was used, which experiments were
approved, and what deterministic training selected.

When `LLM_ENABLED=false`, the report still exists, but the plan source is the
deterministic fallback. When `LLM_ENABLED=true`, the planner calls the configured
OpenAI-compatible provider, validates the response, and still falls back safely if
the provider fails or proposes something outside policy.

The operational pipeline automatically compares the exact version it just
registered and writes:

```text
artifacts/metrics/promotion.json
```

`candidate_validation_metrics` records the held-out validation metrics used to
select the candidate during training. `candidate_metrics` records a fresh
evaluation on `data/test/fixed_test.csv`; these fixed-test values, artifact
loadability, API request compatibility, and comparison with the current champion
control promotion eligibility.

`promotion.json` is a run-specific operational artifact, not a declared DVC output.
Locally it remains ignored by Git. In GitHub Actions it is retained inside the
downloadable workflow artifact, so neither `git pull` nor `dvc pull` restores it.

The comparison can also be rerun without moving the alias:

```bash
make compare-model
```

To manually promote a candidate after reviewing the report:

```bash
make promote-model VERSION=<registered-model-version>
```

The promotion command re-runs the same deterministic gates and only then moves:

```text
customer-churn@champion -> <registered-model-version>
```

The reports created during the data part are:

| Path | Purpose |
|---|---|
| `reports/data-quality/<batch>.validation.json` | Machine-readable validation result |
| `reports/data-quality/<batch>.validation.md` | Human-readable validation summary |
| `reports/drift/<batch>.drift.json` | Structured drift decision used by automation |
| `reports/drift/<batch>.drift.html` | Visual Evidently drift report for inspection |
| `reports/data-profile/training.profile.json` | Aggregate profile of the curated training dataset, including `data_version` |

To run the same scenario against DagsHub and push DVC objects to the configured
DagsHub DVC remote:

```bash
make acceptance-remote
```

Remote mode reads credentials from `.env`, registers the selected model in the
DagsHub MLflow registry, and runs `uv run dvc push -r origin` after the pipeline
succeeds. The `.env` file and `.dvc/config.local` must remain local-only files.

After remote mode, DagsHub should show two separate things:

1. DVC data storage receives the real generated files when `dvc push` succeeds.
2. MLflow receives experiment runs and the selected registered model version.

Locally, the MLflow tracking receipt is written to:

```text
artifacts/metrics/mlflow-tracking.json
```

That file records the selected model, registered model name, registered version,
candidate run IDs, Git commit, and data version. In DagsHub, use the MLflow
experiment page to inspect candidate runs, metrics, logged artifacts, and the
registered model version.

To know whether the model was trained again, inspect:

```bash
cat artifacts/metrics/selection.json
cat artifacts/metrics/mlflow-tracking.json
```

`selection.json` shows the winning candidate and metrics for the current run.
`mlflow-tracking.json` shows whether that selected model was registered, including
the DagsHub MLflow run IDs and registered model version when remote tracking was
enabled. If the current curated `data_version` was already registered before, the
local workflow skips duplicate remote registration.

To confirm DVC remote storage is synchronized:

```bash
uv run dvc status -c -r origin
```

To push only DVC data and reports without running a new pipeline:

```bash
uv run dvc push -r origin
```

## GitHub Actions Data Pipeline

The automated version of the local command lives in
`.github/workflows/data-pipeline.yml`. It is intentionally split into jobs so the
GitHub UI shows where the run is: input detection, quality gates, pipeline
execution, and metadata commit.

It can be triggered in two ways:

```text
1. Push one DVC pointer for a CSV directly under data/incoming/
2. Run the workflow manually and provide incoming_path
```

For a manual run, the file passed as `incoming_path` must already exist in the
branch selected in the GitHub Actions UI. Because real data files are ignored by
Git, the branch normally contains `data/incoming/<batch>.csv.dvc`, and DVC restores
the real CSV during the workflow.

A typical GitHub-triggered flow is:

```bash
# Create a uniquely seeded drifted CSV. This does not validate or train yet.
make create-incoming-batch ARGS="--scenario drifted --filename drifted-batch-001.csv"

# Store the real CSV in the local DVC cache and create the Git-visible pointer.
uv run dvc add data/incoming/drifted-batch-001.csv

# Send the real CSV to the DagsHub DVC remote so GitHub Actions can pull it.
uv run dvc push -r origin

# Commit only the pointer file. The raw CSV remains ignored by Git.
git add data/incoming/drifted-batch-001.csv.dvc
git commit -m "Add drifted incoming customer batch"
git push
```

Automatic detection accepts exactly one incoming CSV across the complete Git push,
not one CSV per commit. If two incoming pointer commits are pushed together, the
workflow stops rather than guessing which batch to process. Process each delivery
with a separate push, or recover by manually running **Data Pipeline** once for each
explicit `incoming_path`.

`--scenario normal` samples from the reference distribution and will usually stop
after drift reporting. `--scenario drifted` applies the project's configured
range-relative shifts to three of eight features. With the current `0.25` gate,
two drifted features are sufficient, so this scenario is designed to open the
automatic training path. The command's timestamp-based default seed keeps customer
identifiers unique; when supplying `--seed` yourself, use a value not used by an
earlier batch.

The push can trigger the workflow automatically on `main`. You can also run the
workflow manually and use:

```text
data/incoming/drifted-batch-001.csv
```

The workflow is split into clear jobs:

| Job | Responsibility |
|---|---|
| `detect-input` | Finds the incoming CSV. Automatic runs require exactly one changed CSV under `data/incoming/`. |
| `quality` | Installs the locked environment, runs linting, and runs the test suite. |
| `run-pipeline` | Uses the `prod` GitHub environment, pulls DVC data, validates and checks drift, conditionally trains, registers and compares the selected model, pushes DVC data, and uploads reports. |
| `commit-metadata` | Commits only Git/DVC metadata back to the branch: `dvc.lock` and `.dvc` pointer files. |

The data pipeline compares the registered candidate automatically and uploads
`promotion.json`, but it does not move the `champion` alias.
For a manual workflow run, select `force_retrain=true` only when you intend to
override the normal no-significant-drift stop.

The uploaded `pipeline-reports` artifact contains the generated inspection
outputs:

```text
reports/data-quality/
reports/data-profile/
reports/drift/
artifacts/metrics/
artifacts/experiment-plans/
artifacts/agent/
```

This is where you inspect validation results, drift decisions, training metrics,
MLflow tracking receipts, the LLM or fallback experiment plan, and the agent
analysis report. Because the runner performs `dvc pull` before execution and the
artifact uploads complete directories, the bundle can also contain historical DVC
outputs. Match the validation and drift filenames to the submitted batch, and use
`data_version`, `git_commit`, and `registered_model_version` to identify current
training outputs.

Manual champion promotion lives in `.github/workflows/promote-model.yml`. Run it
from the GitHub Actions UI with the registered model version you want to promote.
That workflow pulls the immutable fixed-test data from DVC, reloads the requested
registry artifact, re-checks fixed-test thresholds and FastAPI compatibility, and
moves the MLflow `champion` alias only if the candidate still passes.

The manual workflow publishes a separate Actions artifact named
`promotion-report`. Download it from the workflow run's **Summary** page and inspect
`artifacts/metrics/promotion.json`. This report is not restored by Git or DVC.

The workflow uses GitHub Secrets instead of `.env`:

```text
DAGSHUB_USERNAME
DAGSHUB_TOKEN
DAGSHUB_REPOSITORY
MLFLOW_TRACKING_URI
MLFLOW_TRACKING_USERNAME
MLFLOW_TRACKING_PASSWORD
LLM_ENABLED
LLM_PROVIDER
LLM_MODEL
LLM_API_KEY
LLM_BASE_URL
LLM_TIMEOUT_SECONDS
LLM_TEMPERATURE
LLM_MAX_TOKENS
```

In GitHub Actions, GitHub stores the code and DVC metadata. DagsHub DVC storage
stores the real generated datasets and reports. DagsHub MLflow stores the
experiment runs, metrics, artifacts, and registered model version.
The LLM variables are optional from a pipeline-safety perspective: when the LLM
provider is missing, unavailable, or returns an invalid plan, the workflow writes
the fallback plan and continues deterministically.

In this setup, the repositories have different responsibilities:

| Repository or service | Main responsibility |
|---|---|
| GitHub | Stores source code, tests, workflow files, configuration, and DVC pointer metadata, and executes the automation. |
| DagsHub DVC remote | Stores the real DVC-managed data and report files that Git should not store directly. |
| DagsHub MLflow | Stores experiment runs, metrics, parameters, model artifacts, registered model versions, and the `champion` alias. |

This split is common in MLOps because source code history and data/model artifact
history have different sizes, access patterns, and review needs.

You can still process a manually supplied CSV:

```bash
make pipeline INPUT=data/incoming/my-new-batch.csv
```

The file must be directly under `data/incoming/`. Invalid files are routed to
`data/rejected/` and stop before curation, training, and registration.

Data flow:

```text
Synthetic or manual CSV
        |
        v
data/incoming/<batch>.csv
        |
        v
Pandera validation
   |              |
   | valid        | invalid
   v              v
data/accepted/   data/rejected/
        |
        v
Evidently drift report
        |
        v
DVC add raw, accepted, validation, and drift artifacts
        |
        +-- no significant feature drift --> stop
        |
        +-- significant drift or FORCE_RETRAIN=1
                    |
                    v
DVC repro profile + current plan + train
        |
        v
Curated training data + data profile + model artifacts
        |
        v
If curated data version changed
        |
        v
MLflow candidate runs + selected model registration
        |
        v
Fixed-test + API compatibility comparison
        |
        v
Agent analysis from verified current-run artifacts
        |
        v
Optional: DVC push to DagsHub remote storage
```

Configuration that is safe to version belongs in `params.yaml`. Copy variable names
from `.env.example` into a local `.env` file when integrations are introduced. Never
commit credentials.

## FastAPI Serving

After a model version has been promoted to the MLflow `champion` alias, the local
API can serve that model directly from DagsHub MLflow:

```bash
make run-api
```

The API reads `MODEL_ALIAS` from `.env`. For the normal flow, use:

```text
MODEL_ALIAS=champion
```

The serving layer exposes:

```text
GET  /health
GET  /model-info
POST /predict
```

An optional local-only Docker rehearsal remains available:

```bash
make docker-build
make docker-run
```

Container publication and deployment are outside this project's version-one
scope. The local container reads credentials from `.env` at runtime and does not
bake them into the image.
