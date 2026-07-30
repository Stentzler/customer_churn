# Customer Churn Agentic MLOps

A learning project for an automated and auditable customer-churn lifecycle spanning
DataOps, MLOps, and constrained LLMOps.

The project is being implemented incrementally according to [SPEC.md](./SPEC.md).
The current implementation includes the repository foundation, deterministic
synthetic-data scenarios, and data-contract validation.

Detailed subsystem documentation:

- [DataOps guide](./data/README.md)
- [Training and MLflow guide](./src/training/README.md)

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

The command validates and routes the batch, creates drift reports, versions the
batch artifacts in the local DVC cache, reproduces curation through training, and
registers the selected candidate in DagsHub. Invalid data stops before curation.
When the resulting curated `data_version` was already registered by the previous
successful execution, remote tracking is skipped to avoid duplicate model versions.

## Local End-to-End Pipeline

This repository uses four different systems together:

| System | What it stores | What it is used for |
|---|---|---|
| Git / GitHub | Source code, configs, tests, docs, `dvc.yaml`, `dvc.lock`, and small `.dvc` pointer files | Code review, history, collaboration, and future GitHub Actions automation |
| DVC | Real generated datasets, reports, profiles, and model artifacts in `.dvc/cache/` or the DagsHub DVC remote | Versioning large or generated files without committing them directly to Git |
| MLflow | Experiment runs, metrics, parameters, artifacts, and registered model versions | Tracking what was trained and keeping model lineage |
| DagsHub | Hosts the Git repository, DVC remote storage, and MLflow tracking/registry | One place to inspect code, data versions, experiment runs, and registered models |

The fastest way to simulate a new delivery is to let the project create a fresh
valid CSV and run the same local lifecycle that GitHub Actions will use later:

```bash
make acceptance-local
```

This creates a new file under `data/incoming/`, validates it, versions the batch
artifacts with DVC, rebuilds the curated training dataset, profiles the data,
trains candidates, and registers the selected model in a local SQLite MLflow
backend under `.tmp/`. It does not contact DagsHub.

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
```

The main training result is `artifacts/metrics/selection.json`. It tells which
candidate won according to the configured primary metric. Each candidate also gets
its own metrics JSON with ROC-AUC, PR-AUC, F1, precision, recall, confusion matrix,
and class distribution.

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

To confirm DVC remote storage is synchronized:

```bash
uv run dvc status -c -r origin
```

To push only DVC data and reports without running a new pipeline:

```bash
uv run dvc push -r origin
```

## GitHub Actions Data Pipeline

The future automated version of the local command lives in
`.github/workflows/data-pipeline.yml`.

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
# Create only a new incoming CSV. This does not validate, train, or register.
make create-incoming-batch ARGS="--filename my-new-batch.csv"

# Store the real CSV in the local DVC cache and create the Git-visible pointer.
uv run dvc add data/incoming/my-new-batch.csv

# Send the real CSV to the DagsHub DVC remote so GitHub Actions can pull it.
uv run dvc push -r origin

# Commit only the pointer file. The raw CSV remains ignored by Git.
git add data/incoming/my-new-batch.csv.dvc
git commit -m "Add incoming customer batch"
git push
```

Then run the workflow manually and use:

```text
data/incoming/my-new-batch.csv
```

The workflow is split into clear jobs:

| Job | Responsibility |
|---|---|
| `detect-input` | Finds the incoming CSV. Automatic runs require exactly one changed CSV under `data/incoming/`. |
| `quality` | Installs the locked environment, runs linting, and runs the test suite. |
| `run-pipeline` | Uses the `prod` GitHub environment, pulls DVC data, validates the batch, curates data, trains models, registers the selected model in DagsHub MLflow, pushes DVC data, and uploads reports. |
| `commit-metadata` | Commits only Git/DVC metadata back to the branch: `dvc.lock` and `.dvc` pointer files. |

The workflow uses GitHub Secrets instead of `.env`:

```text
DAGSHUB_USERNAME
DAGSHUB_TOKEN
DAGSHUB_REPOSITORY
MLFLOW_TRACKING_URI
MLFLOW_TRACKING_USERNAME
MLFLOW_TRACKING_PASSWORD
```

In GitHub Actions, GitHub stores the code and DVC metadata. DagsHub DVC storage
stores the real generated datasets and reports. DagsHub MLflow stores the
experiment runs, metrics, artifacts, and registered model version.

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
        v
DVC repro profile + train
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
Optional: DVC push to DagsHub remote storage
```

Configuration that is safe to version belongs in `params.yaml`. Copy variable names
from `.env.example` into a local `.env` file when integrations are introduced. Never
commit credentials.
