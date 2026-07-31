# Workflow Orchestration Guide

This directory owns the deterministic application workflow that connects DataOps,
training, MLflow registration, promotion comparison, and final reporting.

The orchestrator does not reimplement those subsystems. It calls their public
interfaces in the required order, passes paths and structured results between them,
and stops when a gate says later work is not allowed.

The central rule remains:

```text
The LLM proposes and explains.
Deterministic code validates, routes, trains, selects, registers, and compares.
```

## Responsibilities

The workflow layer is responsible for:

- Starting from one explicit CSV under `data/incoming/`.
- Running validation before drift, curation, or training.
- Routing invalid data away from accepted and curated datasets.
- Using significant feature drift as the automatic retraining trigger.
- Supporting an explicit human `force_retrain` override.
- Coordinating DVC for batch artifacts and reproducible training stages.
- Avoiding duplicate registration for an unchanged curated `data_version`.
- Registering the selected model before promotion comparison.
- Comparing without automatically moving the `champion` alias.
- Creating final analysis only from verified pipeline artifacts.

It does not own data rules, estimator construction, MLflow implementation details,
promotion thresholds, API serving, or LLM provider behavior.

## Source Map

| File | Responsibility |
|---|---|
| [`local_pipeline.py`](./local_pipeline.py) | Current executable orchestration service and CLI |
| [`conditions.py`](./conditions.py) | Reserved module boundary for deterministic routing conditions |
| [`nodes.py`](./nodes.py) | Reserved module boundary for small workflow operations |
| [`state.py`](./state.py) | Reserved module boundary for structured workflow state |
| [`graph.py`](./graph.py) | Reserved module boundary for graph assembly if the explicit service later becomes too complex |

Version one intentionally runs through `local_pipeline.py`. The other modules are
package boundaries only and contain no runtime workflow framework. LangGraph is not
required for the current branching complexity.

## Entry Point

Run one incoming batch from the repository root:

```bash
make pipeline INPUT=data/incoming/<batch>.csv
```

Force training despite a no-significant-feature-drift result:

```bash
make pipeline INPUT=data/incoming/<batch>.csv FORCE_RETRAIN=1
```

The equivalent Python entry point is:

```bash
uv run python -m src.workflow.local_pipeline \
  --input data/incoming/<batch>.csv
```

`FORCE_RETRAIN=1` is a human override of the retraining route only. It does not
weaken validation, model-selection, registration, fixed-test, API-compatibility, or
promotion gates.

## Lifecycle

```text
explicit incoming CSV
        |
        v
validate and route
   |             |
   | valid       | invalid
   v             v
accepted      rejected + stop
   |
   v
feature and target drift analysis
   |
   +-- feature drift not significant and no override
   |       -> version batch reports with DVC -> stop
   |
   +-- significant feature drift or FORCE_RETRAIN=1
           |
           v
       DVC reproduce profile
           |
           v
       generate and validate current experiment plan
           |
           v
       DVC reproduce candidate training
           |
           v
       compare curated data_version with last tracking receipt
           |
           +-- unchanged -> stop before duplicate registration
           |
           +-- changed
                   |
                   v
             MLflow candidate runs
                   |
                   v
             register selected candidate
                   |
                   v
             compare candidate with champion
                   |
                   v
             write final analysis
```

The comparison at the end never moves the alias. It records whether the candidate
is eligible. Actual alias movement requires `make promote-model` or the separate
manual **Promote Model** GitHub Actions workflow.

## Terminal Statuses

`LocalPipelineResult` exposes one of four terminal states:

| Status | Meaning | Training or registration |
|---|---|---|
| `rejected` | The data contract failed and the batch was routed to `data/rejected/` | Not allowed |
| `skipped_no_significant_drift` | Data was valid, but feature drift did not cross the configured share threshold | Skipped unless forced |
| `skipped_unchanged` | Training outputs were reproduced, but the curated `data_version` matches the last successful tracking receipt | Duplicate registration skipped |
| `registered` | A changed curated dataset produced candidate runs and a selected registered model version | Completed |

`registered` does not mean promoted. The result separately records
`promotion_passed`, and the `champion` alias remains unchanged until an explicit
promotion command succeeds.

## Step Ordering

### 1. Validation and routing

`process_incoming_batch` verifies that the file is directly under
`data/incoming/`, writes JSON and Markdown quality reports, and copies the source to
exactly one destination:

```text
data/accepted/<batch>.csv
data/rejected/<batch>.csv
```

Rejected data returns before drift, curation, training, MLflow, or promotion code
can run.

### 2. Drift and batch versioning

An accepted batch is compared with `data/reference/reference.csv`. Feature and
target drift are persisted separately. Only significant feature drift controls the
automatic training route.

The workflow runs `dvc add` for:

- The raw incoming CSV.
- The accepted copy.
- Validation JSON and Markdown.
- Drift JSON and HTML.

This writes small `.dvc` pointer files for Git and stores real contents in the DVC
cache.

### 3. Reproducible curation, planning, and training

For the training route, the workflow executes:

```text
dvc repro profile
dvc repro --force fallback_plan
dvc repro train
```

Reproducing `profile` also follows its upstream curation dependencies. The plan is
forced so the current optional LLM configuration is evaluated for this operational
run. Training still validates the resulting plan deterministically before building
allowlisted candidates.

### 4. Duplicate registration prevention

The profile contains a SHA-256 `data_version` for the complete curated dataset. The
workflow compares it with `artifacts/metrics/mlflow-tracking.json`, which records
the last successful registration performed in the current workspace.

Matching versions stop with `skipped_unchanged`. This avoids creating another
registered model version for identical curated data.

### 5. Tracking, registration, and comparison

For changed data, MLflow receives one run per successful candidate. Only the
deterministically selected candidate becomes a registered model version.

That exact version is then evaluated against the immutable fixed-test dataset and
compared with the current champion. The comparison writes:

```text
artifacts/metrics/promotion.json
```

Passing means eligible for manual promotion. Failing means the candidate stays
registered for lineage but cannot replace the champion.

### 6. Final analysis

The final report reads the approved plan, planner trace, profile, candidate metrics,
selection, and current promotion report. It cannot change any decision and writes:

```text
artifacts/agent/agent-analysis.md
```

## Storage Boundaries

| System | Workflow responsibility |
|---|---|
| Git | Stores source, configuration, workflows, `dvc.yaml`, `dvc.lock`, and `.dvc` pointers |
| DVC | Stores declared pipeline outputs and files explicitly added by the batch workflow |
| MLflow/DagsHub | Stores experiment runs, candidate artifacts, registry versions, tags, and aliases |
| GitHub Actions artifacts | Retain downloadable reports produced inside one remote runner |

`promotion.json`, `mlflow-tracking.json`, and `agent-analysis.md` are run-specific
operational outputs rather than declared outputs in the static DVC graph. In GitHub
Actions they are available through uploaded workflow artifacts. `git pull` and
`dvc pull` do not restore files that were never committed or declared to DVC.

## GitHub Actions Mapping

The remote lifecycle is implemented in
`.github/workflows/data-pipeline.yml` and has four visible jobs:

| Job | Responsibility |
|---|---|
| `detect-input` | Resolve exactly one incoming CSV without guessing |
| `quality` | Install the locked environment and run lint and tests |
| `run-pipeline` | Pull DVC data, execute this orchestrator, push DVC data, and upload reports |
| `commit-metadata` | Commit generated DVC metadata after a successful push-triggered run |

### Automatic input detection

For a push event, detection compares the complete Git push range. Exactly one
unique path matching either form is accepted:

```text
data/incoming/<batch>.csv
data/incoming/<batch>.csv.dvc
```

Two incoming commits included in one push are seen as two changed batches and the
workflow fails intentionally. Push each batch separately. Existing failed batches
can be processed through manual execution by supplying one explicit
`incoming_path` at a time.

### Manual execution

The manual workflow input is the real CSV path, even though Git normally contains
only its pointer:

```text
data/incoming/<batch>.csv
```

The `.dvc` pointer must already exist in the selected branch and the real CSV must
already have been pushed to the DagsHub DVC remote.

### Reports

The data workflow uploads `pipeline-reports`. Because the runner first restores DVC
outputs and then uploads complete directories, the download may contain historical
files. Identify current batch reports by filename and current training outputs by
`data_version`, `git_commit`, and registered model version.

The separate manual promotion workflow uploads `promotion-report`, containing:

```text
artifacts/metrics/promotion.json
```

Download Actions artifacts from the run's **Summary** page. They are not retrieved
with Git or DVC commands.

Current behavior to be aware of: a rejected batch returns a nonzero process status.
The GitHub job therefore stops before its later generic report-upload step. The
validation reports are produced inside the temporary runner, and the rejection
status is visible in logs, but the current workflow does not retain the full reports
as downloadable artifacts.

## Configuration and Secrets

Versioned workflow policy comes from `params.yaml`, including drift, experiment,
training, registry, and promotion settings.

Local integration credentials come from ignored `.env` and `.dvc/config.local`
files. GitHub uses encrypted secrets in the protected `prod` environment. No raw
customer rows, tokens, passwords, or complete environment values should appear in
workflow logs or reports.

## Exit Codes and Failures

The CLI returns:

| Code | Meaning |
|---:|---|
| `0` | Registered, intentionally skipped for drift, or skipped as unchanged |
| `1` | Incoming data was rejected by the contract |
| `2` | An operational dependency or required artifact failed |

An LLM failure is not an operational pipeline failure. Planning falls back to the
deterministic allowlisted plan. Data validation, DVC, training, MLflow, registry,
fixed-test, artifact-loading, and API-compatibility failures remain blocking.

## Testing

Run workflow unit tests:

```bash
uv run pytest tests/unit/workflow -q
```

Run the complete project suite:

```bash
make test
```

The workflow tests use injected DVC runners and mocked subsystem boundaries. They
verify routing and ordering without network access or writes to real DVC/MLflow
state.

## Current Boundaries

The workflow intentionally does not:

- Watch directories continuously.
- Process multiple incoming batches in one automatic run.
- Execute LLM-generated code or commands.
- Automatically move the `champion` alias.
- Deploy the API or publish a container image.
- Require LangChain, LangGraph, Airflow, or another workflow framework.

These limits keep the operational lifecycle explicit, deterministic, and suitable
for the project's DataOps and MLOps study goals.
