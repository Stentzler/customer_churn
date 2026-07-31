# DataOps Guide

This document explains the complete DataOps subsystem for the synthetic
customer-churn project. It covers the data contract, generated scenarios,
validation and routing, curation, profiling, drift detection, and DVC pipeline.

The project uses synthetic data so the complete lifecycle can run locally without
access to a production customer system. In a real project, generation would usually
be replaced by ingestion from a database, warehouse, object store, API, or event
stream. The validation, routing, curation, profiling, drift, and versioning stages
would still be needed.

## Core Invariants

The DataOps implementation protects these rules:

1. Incoming data is validated before it can be accepted or curated.
2. Invalid data is copied to `data/rejected/` and never enters training data.
3. The fixed test dataset never enters curation or training.
4. The reference dataset remains the fixed baseline for drift comparison.
5. Feature drift and target drift are evaluated separately.
6. Customer rows and identifier values are not written to profile or drift JSON.
7. The same configuration and seed produce byte-identical generated CSV files.
8. Curation and report outputs use deterministic ordering and formatting.
9. Filesystem writes use temporary files where partial outputs would be dangerous.
10. DVC records the code, configuration, data, and artifact versions used by each
    pipeline stage.

## Lifecycle Overview

The successful path is:

```text
params.yaml
    |
    v
Synthetic generation
    |
    +--> data/reference/reference.csv
    +--> data/test/fixed_test.csv
    +--> data/incoming/normal.csv
    +--> data/incoming/drifted.csv
    +--> data/incoming/invalid.csv
                              |
                              v
                      Contract validation
                         /          \
                        /            \
                  accepted          rejected
                     |                 |
                     v                 v
          data/accepted/*.csv  data/rejected/*.csv
                     |
             +-------+-------+
             |               |
             v               v
       Drift analysis      Curation
             |               |
             v               v
       reports/drift/  data/curated/training.csv
                             |
                             v
                   Aggregate data profile
                             |
                             v
              reports/data-profile/training.profile.json
```

The rejection path stops after validation:

```text
data/incoming/invalid.csv
    |
    v
validation reports
    |
    v
data/rejected/invalid.csv
    |
    +--> no curation
    +--> no profiling
    +--> no training
```

## Directory Responsibilities

| Directory | Responsibility |
|---|---|
| `data/reference/` | Fixed baseline distribution and initial training population |
| `data/test/` | Immutable final evaluation data; never used for fitting or selection |
| `data/incoming/` | Raw deliveries waiting for validation |
| `data/accepted/` | Validated deliveries eligible for drift analysis and curation |
| `data/rejected/` | Contract-breaking deliveries retained for investigation |
| `data/curated/` | Deterministic training dataset assembled from trusted sources |
| `reports/data-quality/` | Machine-readable JSON and human-readable Markdown validation reports |
| `reports/data-profile/` | Aggregate JSON profile for the curated dataset |
| `reports/drift/` | Stable drift-decision JSON and Evidently HTML visualizations |
| `.dvc/cache/` | Local content-addressed storage managed by DVC |

Generated CSVs and reports are ignored by Git. Their directories remain in Git
through `.gitkeep` files. DVC stores output content in `.dvc/cache/` and records
content hashes in `dvc.lock`.

## Data Contract

The customer-churn contract is defined by versioned values in
[`params.yaml`](../../params.yaml) and translated into executable Pandera checks by
[`src/data/schema.py`](./schema.py).

### Column Schema

| Column | Logical type | Purpose |
|---|---|---|
| `customer_id` | string | Unique customer identifier |
| `age` | integer | Customer age |
| `tenure_months` | integer | Number of months as a customer |
| `monthly_spend` | float | Average monthly subscription spend |
| `support_tickets_90d` | integer | Support tickets in the previous 90 days |
| `late_payments_12m` | integer | Late payments in the previous 12 months |
| `usage_hours_monthly` | float | Average monthly product usage |
| `plan_type` | string category | Subscription plan |
| `region` | string category | Customer region |
| `churned` | integer target | `0` for retained and `1` for churned |

The model feature set excludes `customer_id` and `churned`. There are therefore
eight input features: six numerical and two categorical.

### Numerical Ranges

| Column | Minimum | Maximum |
|---|---:|---:|
| `age` | 18 | 100 |
| `tenure_months` | 0 | 120 |
| `monthly_spend` | 0.0 | 500.0 |
| `support_tickets_90d` | 0 | 20 |
| `late_payments_12m` | 0 | 12 |
| `usage_hours_monthly` | 0.0 | 300.0 |

Ranges are inclusive.

### Categories and Target

Accepted plans:

```text
basic
standard
premium
```

Accepted regions:

```text
north
south
east
west
```

Accepted target values:

```text
0
1
```

### Dataset-Level Rules

Validation also enforces:

- Exactly the documented columns when strict mode is enabled.
- The documented column order when ordered mode is enabled.
- Non-null values in every column.
- Non-empty, non-whitespace customer identifiers.
- Unique `customer_id` values within each batch.
- No completely duplicated rows.
- At least 50 rows per batch.
- Tenure consistent with customer age.

The age/tenure business rule is:

```text
tenure_months <= (age - 18) * 12
```

A customer cannot have subscription tenure from before age 18.

### Configuration Validation

[`src/data/settings.py`](./settings.py) treats `params.yaml` as an
untrusted external boundary. It validates:

- Required and unexpected configuration fields.
- Boolean and integer types.
- Positive row counts.
- Finite numerical values.
- Minimum values smaller than maximum values.
- Non-empty and unique category lists.
- Exactly `[0, 1]` as target values.
- Unique, non-negative scenario seed offsets.
- A feature drift share threshold greater than `0` and at most `1`.

Loaded settings are represented by frozen dataclasses. Nested dictionaries are
exposed through read-only mapping proxies so policy cannot be mutated accidentally
after loading.

## Synthetic Data Generation

Generation is implemented in
[`src/data/generate.py`](./generate.py).

### Why Generation Exists

This repository has no real customer source. The generator simulates:

| Scenario | Production equivalent |
|---|---|
| `reference` | Historical baseline and initial training data |
| `fixed_test` | Approved isolated evaluation dataset |
| `normal` | Routine new production delivery |
| `drifted` | Valid delivery whose population has changed |
| `invalid` | Corrupted or contract-breaking delivery |

### Reproducibility

The project seed is `42`. Each scenario has a distinct offset:

| Scenario | Seed calculation | Effective seed |
|---|---:|---:|
| `reference` | `42 + 0` | 42 |
| `fixed_test` | `42 + 1` | 43 |
| `normal` | `42 + 2` | 44 |
| `drifted` | `42 + 3` | 45 |
| `invalid` | `42 + 4` | 46 |

The generator uses an isolated `random.Random` instance rather than global random
state. The same row count, seed, and contract therefore produce the same dataframe
without affecting random behavior elsewhere in the application.

Customer identifiers include the scenario seed and row number:

```text
CUST-000042-000000
```

This keeps identifiers reproducible and prevents collisions between independently
generated scenario files.

### Dataset Sizes

The current configuration generates:

| Scenario | Rows |
|---|---:|
| Reference | 1,000 |
| Fixed test | 300 |
| Normal batch | 200 |
| Drifted batch | 200 |
| Invalid batch | 200 |

The generator rejects a configured row count below the contract minimum.

### Churn Signal

The target is not a random alternating label. The generator calculates a simple
logistic churn probability:

- More support tickets increase churn risk.
- More late payments increase churn risk.
- Greater usage reduces churn risk.
- Longer tenure reduces churn risk.
- Plan type contributes a small adjustment.

This creates a learnable but intentionally simple relationship for later MLOps
experiments.

### Drifted Scenario

The drifted scenario starts from valid generated data and applies range-relative
changes:

- `monthly_spend` shifts upward.
- `support_tickets_90d` shifts upward.
- `usage_hours_monthly` shifts downward.

Values are clipped to the accepted contract ranges, so this dataset is drifted but
still valid.

### Invalid Scenario

The invalid scenario intentionally introduces:

- An unsupported region.
- An age below the accepted minimum.
- A duplicated customer identifier.

These violations are deterministic and are used to test the negative path.

### Persistence

CSV files use:

- No pandas index column.
- Unix `\n` line endings.
- Two decimal places for floats.
- Stable paths and filenames.
- A temporary sibling file followed by atomic replacement.

The final path is replaced only after the complete temporary CSV is written.

### Generation Commands

Generate all scenarios:

```bash
make generate-data
```

Equivalent command:

```bash
uv run python -m src.data.generate --scenario all
```

Generate one predefined scenario:

```bash
uv run python -m src.data.generate --scenario normal
uv run python -m src.data.generate --scenario drifted
uv run python -m src.data.generate --scenario invalid
```

Available values are:

```text
reference
fixed_test
normal
drifted
invalid
all
```

Use a different configuration or output root:

```bash
uv run python -m src.data.generate \
  --scenario all \
  --params params.yaml \
  --data-root data
```

## Validation

Validation is divided into four modules:

| Module | Responsibility |
|---|---|
| [`schema.py`](./schema.py) | Builds executable Pandera checks |
| [`validate.py`](./validate.py) | Runs validation and normalizes Pandera failures |
| [`validation_models.py`](./validation_models.py) | Defines stable immutable result models |
| [`validation_report.py`](./validation_report.py) | Renders JSON and Markdown artifacts |

### Lazy Validation

Pandera runs with `lazy=True`. It collects independent failures instead of stopping
after the first problem. One invalid batch can therefore report an invalid region,
an out-of-range age, and duplicate identifiers in a single run.

Expected Pandera schema failures become project-owned `ValidationIssue` objects.
Unexpected programming or library errors are not mislabeled as bad data.

### Stable Issue Codes

Reports use stable project codes such as:

| Code | Meaning |
|---|---|
| `missing_column` | Required column is absent |
| `unexpected_column` | Strict schema found an unknown column |
| `incorrect_column_order` | Columns are not in contract order |
| `duplicate_customer_id` | Identifier appears more than once |
| `null_value` | Non-nullable field contains null |
| `empty_customer_id` | Identifier contains no non-whitespace text |
| `minimum_batch_size` | Dataset contains fewer than 50 rows |
| `duplicate_rows` | Complete rows are duplicated |
| `tenure_inconsistent_with_age` | Tenure implies subscription before age 18 |
| `invalid_category` | Category is outside its allowlist |
| `invalid_target` | Target is not `0` or `1` |
| `out_of_range` | Numerical value violates its range |
| `invalid_dtype` | Source dtype does not match the contract |
| `contract_violation` | Generic fallback for an unstructured schema failure |

Failure examples are sorted, deduplicated, and limited to 10 by configuration.
Reports do not contain an unbounded copy of the source data.

### Validation Reports

Every validation run writes:

```text
reports/data-quality/<dataset>.validation.json
reports/data-quality/<dataset>.validation.md
```

JSON is intended for DVC stages, CI, and later workflow code. Markdown is intended
for humans. Reports omit timestamps and absolute source paths so identical results
produce identical report bytes.

### Validation Command

Validate without routing:

```bash
uv run python -m src.data.validate \
  --input data/incoming/normal.csv \
  --params params.yaml \
  --report-dir reports/data-quality
```

Exit codes:

| Code | Meaning |
|---:|---|
| `0` | Dataset satisfies the contract |
| `1` | Dataset was read successfully but rejected by the contract |
| `2` | Validation could not run because of configuration, input, or report I/O |

## Incoming Batch Routing

Routing is implemented in
[`src/data/ingest.py`](./ingest.py).

The ingestion boundary:

1. Requires the source CSV to be directly under `data/incoming/`.
2. Runs validation and writes both reports.
3. Selects exactly one disposition: accepted or rejected.
4. Atomically copies the original CSV to the selected directory.
5. Preserves the incoming source as the raw delivery.

Only these transitions are allowed:

```text
data/incoming/<batch>.csv -> data/accepted/<batch>.csv
data/incoming/<batch>.csv -> data/rejected/<batch>.csv
```

A valid file under `data/test/` cannot be passed through this interface. This
prevents the fixed test dataset from being routed into accepted history.

Process an incoming batch:

```bash
make process-batch INPUT=data/incoming/normal.csv
```

Equivalent command:

```bash
uv run python -m src.data.ingest \
  --input data/incoming/normal.csv \
  --params params.yaml \
  --data-root data \
  --report-dir reports/data-quality
```

Routing exit codes:

| Code | Meaning |
|---:|---|
| `0` | Accepted and copied to `data/accepted/` |
| `1` | Rejected and copied to `data/rejected/` |
| `2` | Operational failure; routing could not complete safely |

Rejection code `1` is intentional. Automation must not interpret rejected data as a
successful path.

## Drift Detection

Drift detection is implemented in
[`src/data/drift.py`](./drift.py) using Evidently.

### Comparison Policy

The primary comparison is:

```text
data/reference/reference.csv
              versus
data/accepted/<current-batch>.csv
```

The reference dataset defines the fixed baseline. Comparing the newest accepted
batch directly with it is more sensitive than comparing the reference with the
complete curated history, where older rows could dilute recent changes.

The current dataset must be directly under `data/accepted/`. Reference and current
data are both validated before statistical analysis.

### Feature Drift

The following eight features are evaluated:

```text
age
tenure_months
monthly_spend
support_tickets_90d
late_payments_12m
usage_hours_monthly
plan_type
region
```

`customer_id` is excluded because it is an identifier, not a model feature.
`churned` is excluded from the feature drift count because it is the target.

Evidently selects an appropriate per-column method based on type and data size. With
the current generated datasets, reports use methods such as:

- Kolmogorov-Smirnov p-value for continuous numerical columns.
- Chi-square p-value for categorical columns.
- Z-test p-value for the binary target.

For p-value methods, this project marks a column as drifted when:

```text
score < threshold
```

For distance methods, drift is detected when:

```text
score >= threshold
```

The current per-column threshold selected by Evidently is `0.05`.

### Dataset-Level Drift Gate

The versioned project threshold is:

```yaml
drift:
  feature_drift_share_threshold: 0.25
```

The calculation is:

```text
drift_share = drifted_feature_count / feature_count
```

With eight features, two or more drifted features reach the `0.25` gate:

```text
2 / 8 = 0.25
```

The decision uses `>=`, so exactly `0.25` is significant.

Current synthetic behavior:

| Batch | Drifted features | Share | Significant |
|---|---|---:|---|
| Normal | `monthly_spend` | 0.125 | No |
| Drifted | `monthly_spend`, `support_tickets_90d`, `usage_hours_monthly` | 0.375 | Yes |

An individual feature can be statistically drifted without crossing the
dataset-level gate. This reduces the chance that one noisy feature triggers the
overall workflow.

### Target Drift

`churned` is evaluated independently through Evidently `ValueDrift`. The structured
result contains:

- Reference target proportions.
- Current target proportions.
- Statistical method.
- Score.
- Threshold.
- Boolean target drift decision.

Target drift does not increase the feature drift count.

### Drift Artifacts

Each accepted batch produces:

```text
reports/drift/<batch>.drift.json
reports/drift/<batch>.drift.html
```

The JSON is a stable project-owned contract containing:

- Reference and current dataset names.
- SHA-256 versions for both CSV files.
- Per-feature method, score, threshold, and decision.
- Drifted feature names and count.
- Feature drift share and configured gate.
- Separate target distribution and target drift result.

The HTML file is Evidently's visual exploration report. The adapter passes only the
eight features and target to Evidently, so customer identifiers are excluded.

Run drift detection:

```bash
make drift-data CURRENT=data/accepted/normal.csv
```

Equivalent command:

```bash
uv run python -m src.data.drift \
  --current data/accepted/normal.csv \
  --params params.yaml \
  --report-dir reports/drift
```

Exit codes:

| Code | Meaning |
|---:|---|
| `0` | Analysis completed, whether or not drift was detected |
| `2` | Input, validation, configuration, Evidently parsing, or report failure |

Detected drift is a result, not an operational error. It does not use exit code `1`.

## Curation

Curation is implemented in
[`src/data/curate.py`](./curate.py).

The curated training dataset is rebuilt from:

```text
data/reference/reference.csv
data/accepted/*.csv
```

The fixed test and rejected directories are not accepted as function inputs.

### Curation Algorithm

1. Load the reference dataset first.
2. Find accepted CSV files.
3. Sort accepted filenames lexicographically.
4. Validate every source independently.
5. Concatenate reference and accepted rows.
6. Deduplicate by `customer_id`.
7. Keep the last occurrence according to source order.
8. Sort the final rows by `customer_id`.
9. Validate the complete curated dataframe.
10. Atomically write `data/curated/training.csv`.

The current duplicate precedence rule is:

```text
reference first
accepted files in sorted filename order
last customer occurrence wins
```

For example:

```text
reference.csv:       CUST-001, monthly_spend=50
accepted/2026-02.csv CUST-001, monthly_spend=80
```

The curated dataset retains the accepted row with spend `80`.

Stable final sorting provides reproducible file hashes and DVC cache behavior. The
complete output is validated again because independently valid inputs do not prove
that merge logic is correct.

Run curation:

```bash
make curate-data
```

Equivalent command:

```bash
uv run python -m src.data.curate \
  --params params.yaml \
  --data-root data
```

The successful result logs:

- Number of source files.
- Total input rows.
- Final output rows.
- Duplicate customer count removed.
- Output path.

## Dataset Profiling

Profiling is implemented in
[`src/data/profile.py`](./profile.py).

It validates the input CSV, computes aggregate statistics, and writes:

```text
reports/data-profile/training.profile.json
```

### Profile Contents

The JSON includes:

- Schema version.
- Dataset filename.
- SHA-256 data-version identifier.
- Row count.
- Feature count.
- Ordered feature names.
- Pandas feature dtypes.
- Missing-value count for every column.
- Numerical minimum, maximum, mean, median, and population standard deviation.
- Categorical frequencies.
- Target distribution.
- Duplicate-row count.
- Drift evaluation status and drifted feature list fields.

Population standard deviation uses `ddof=0`.

The profile contains the `customer_id` column name only as part of the missing-value
summary. It never contains customer identifier values or raw rows.

The current curated profile has `drift_evaluated: false` because batch drift is
stored in separate drift reports. The profile contract already supports verified
drifted feature names for later workflow integration.

Run profiling:

```bash
make profile-data
```

Equivalent command:

```bash
uv run python -m src.data.profile \
  --input data/curated/training.csv \
  --params params.yaml \
  --report-dir reports/data-profile
```

## DVC In Depth

DVC is installed in the project environment as a development dependency. No global
installation or online account is required.

### Git and DVC Responsibilities

Git stores:

```text
source code
params.yaml
dvc.yaml
dvc.lock
.dvc/config
small metadata and documentation
```

DVC stores or restores:

```text
generated CSV files
accepted datasets
curated training data
validation reports
profile reports
drift reports
model artifacts
training metrics
experiment plans and planner traces
```

The actual output contents are cached locally under `.dvc/cache/`. Git sees the
pipeline definition and hashes, not the large generated files.

### Important DVC Files

#### `dvc.yaml`

[`dvc.yaml`](../../dvc.yaml) is the pipeline recipe. Every stage declares:

- `cmd`: command DVC executes.
- `deps`: files whose content can invalidate the stage.
- `params`: selected versioned configuration values.
- `outs`: artifacts created and cached by the stage.

#### `dvc.lock`

[`dvc.lock`](../../dvc.lock) is the resolved execution record. It contains:

- Exact command used by each stage.
- Dependency content hashes and sizes.
- Parameter values used by the run.
- Output content hashes and sizes.

`dvc.yaml` says what should happen. `dvc.lock` records exactly what did happen.
Both files belong in Git.

#### `.dvc/cache/`

The cache is content-addressed. DVC stores an output based on its hash rather than
its filename. If two versions have the same content, DVC can reuse one cached
object. The cache is local and ignored by Git.

#### `.dvc/config`

The committed project configuration disables DVC analytics and declares the
DagsHub remote:

```ini
[core]
    analytics = false
    remote = origin
```

The remote endpoint is non-secret project metadata. Authentication never belongs
in this committed file.

#### `.dvc/config.local`

This ignored file is where machine-specific DagsHub credentials belong. GitHub
Actions writes the same settings locally from encrypted secrets. Secrets must
never be committed in `.dvc/config`, `params.yaml`, or source code.

### Current DAG

The reproducible DVC graph has nine successful stages:

```text
generate
  |
  +--> validate_normal ----+--> drift_normal
  |                        |
  |                        +--> curate --> profile
  |                        |
  +--> validate_drifted ---+--> drift_drifted

fallback_plan -------------------------+
                                      |
curate -----------------------------> train
```

Stage details:

| Stage | Main input | Main output |
|---|---|---|
| `generate` | Seed, generation config, contract | Reference, test, normal, drifted, invalid CSVs |
| `validate_normal` | Incoming normal CSV | Accepted normal CSV and validation reports |
| `validate_drifted` | Incoming drifted CSV | Accepted drifted CSV and validation reports |
| `drift_normal` | Reference and accepted normal CSV | Normal JSON and HTML drift reports |
| `drift_drifted` | Reference and accepted drifted CSV | Drifted JSON and HTML drift reports |
| `curate` | Reference, accepted normal, accepted drifted | Curated training CSV |
| `profile` | Curated training CSV | Aggregate training profile JSON |
| `fallback_plan` | Prompt, catalog, schemas, and experiment policy | Approved plan and planner trace |
| `train` | Curated data and approved plan | Candidate pipelines, metrics, and selection |

The final agent analysis is intentionally outside this static graph. The
operational workflow writes it only after MLflow registration and deterministic
promotion comparison, which are remote and run-specific operations.

The invalid scenario is generated but is not a default successful DVC stage.
Processing it correctly returns exit code `1`, which would stop `dvc repro`.
Invalid-path behavior is covered by automated tests and can be demonstrated with
the standalone ingestion command.

### How DVC Decides to Rerun a Stage

For each stage, DVC compares the current workspace with `dvc.lock`.

A stage becomes changed when, for example:

- Its command changes.
- A declared source file changes.
- An upstream output hash changes.
- A tracked `params.yaml` value changes.
- A declared output is missing or modified.

DVC then runs the changed stage and downstream stages that depend on its output.

Example:

```text
change project.random_seed
    |
    v
generate reruns
    |
    v
generated hashes change
    |
    v
validation, drift, curation, and profile rerun as needed
```

If nothing changed:

```text
Stage 'generate' didn't change, skipping
...
Data and pipelines are up to date.
```

### Running the Pipeline

Install the locked project environment:

```bash
make install
```

Run the full DataOps DAG:

```bash
make data-pipeline
```

Equivalent command:

```bash
uv run dvc repro
```

Inspect status:

```bash
uv run dvc status
```

Display the DAG:

```bash
uv run dvc dag
```

Force reproduction when debugging:

```bash
uv run dvc repro --force
```

Reproduce one target and its dependencies:

```bash
uv run dvc repro profile
uv run dvc repro drift_normal
```

Restore workspace outputs from the local cache:

```bash
uv run dvc checkout
```

### Parameter Tracking

DVC does not treat every change to `params.yaml` as relevant to every stage.
`dvc.yaml` lists the sections used by each stage.

Examples:

- Generation tracks `project.random_seed`, `data_generation`, and `data_contract`.
- Validation tracks `data_contract`.
- Drift tracks `data_contract` and `drift`.
- Curation and profile track `data_contract`.

Changing an unrelated future API port should not invalidate the data pipeline.

### DVC Remote Storage

The repository uses the DagsHub-managed DVC remote named `origin`. Its non-secret
S3-compatible endpoint is committed in `.dvc/config`; the DagsHub token is stored
only in ignored `.dvc/config.local`.

The storage workflow is:

```text
local dvc repro
    |
    v
local .dvc/cache
    |
    v
dvc push
    |
    v
DagsHub DVC storage
```

Another machine will use:

```bash
git clone <repository>
make install
uv run dvc pull
```

Credentials are stored locally or in GitHub encrypted secrets, never committed. Publish
new cache objects with `uv run dvc push`; restore them on another machine with
`uv run dvc pull`.

## Simulating New Incoming Data

There are two useful workflows.

### Reproduce a New Version of a Predefined Scenario

Change only the relevant seed offset in `params.yaml`:

```yaml
data_generation:
  seed_offsets:
    normal: 12
```

Then run:

```bash
make data-pipeline
```

DVC detects the parameter change, reruns generation, and reruns affected downstream
stages. This is the recommended way to study parameter-based reproducibility with
the current static DAG.

After the experiment, review:

```bash
uv run dvc status
git diff -- params.yaml dvc.lock
```

### Create an Additional Ad Hoc Batch

The batch helper is the preferred interface for creating a separately named
synthetic delivery:

```bash
make create-incoming-batch ARGS="--scenario normal --filename batch-001.csv"
make create-incoming-batch ARGS="--scenario drifted --filename drifted-batch-001.csv"
```

It writes only the CSV under `data/incoming/`; validation, DVC tracking, and training
do not run until a pipeline command is executed. The lower-level Python API remains
available for studying generation directly:

```bash
uv run python -c "
from pathlib import Path
from src.data.generate import generate_valid_customer_dataframe
from src.data.settings import load_data_contract

contract = load_data_contract(Path('params.yaml'))
batch = generate_valid_customer_dataframe(
    row_count=200,
    seed=100,
    contract=contract,
)
output = Path('data/incoming/batch-001.csv')
output.parent.mkdir(parents=True, exist_ok=True)
batch.to_csv(output, index=False, lineterminator='\n', float_format='%.2f')
"
```

Process and analyze it:

```bash
make pipeline INPUT=data/incoming/batch-001.csv
```

For the GitHub lifecycle, `normal` usually stops after a
no-significant-feature-drift result. The `drifted` scenario shifts three features
and is designed to cross the configured `0.25` feature-drift-share gate.

Use a unique seed for each delivery when passing `--seed` explicitly. The seed is
embedded in `customer_id`; reusing it can cause curation to replace existing rows
instead of increasing the effective training dataset.

The local orchestrator performs these operations in order:

1. Validate the exact input path and write JSON and Markdown quality reports.
2. Copy valid data to `data/accepted/`, or invalid data to `data/rejected/`.
3. Stop immediately with exit code `1` when the batch is invalid.
4. Compare an accepted batch with the reference data and persist drift reports.
5. Run `dvc add` for the raw batch, accepted copy, validation reports, and drift
   reports. Their `.dvc` files are metadata that should be committed to Git.
6. Stop after report persistence when `feature_drift.is_significant` is false,
   unless a human supplied the explicit retraining override.
7. Reproduce `profile`, force the `fallback_plan` stage so current optional LLM
   settings are evaluated, and then reproduce `train`. The `curate` stage depends
   on the complete
   `data/accepted/` directory, so a new accepted filename invalidates curation,
   profiling, and training.
8. Compare the profile's curated `data_version` with the last successful MLflow
   tracking receipt.
9. Skip remote registration when the effective curated data is unchanged.
10. Otherwise, track both candidates, register the selected candidate, compare
    that exact version with the champion, and create the final analysis report.

Feature and target drift remain separate. Only significant **feature** drift
automatically opens the training path; target drift remains recorded evidence and
does not independently trigger retraining. To demonstrate the complete lifecycle
with a valid low-drift batch, use:

```bash
make pipeline INPUT=data/incoming/batch-001.csv FORCE_RETRAIN=1
```

The `make acceptance-local` study command applies this override internally so it
always rehearses training, registration, comparison, and final reporting.

Use a unique, immutable delivery filename such as a timestamp or source batch ID:

```text
data/incoming/customer-churn-2026-07-29-001.csv
```

Reusing a filename is supported, but unique names preserve clearer audit history.
If a previously accepted filename is replaced by invalid data, ingestion removes
the stale accepted copy before stopping the training path.

### DVC and Git After a Successful Batch

`dvc add` stores file contents in `.dvc/cache/` and writes small `.dvc` pointer
files. The actual CSV and generated reports remain ignored by Git. Review and commit
the metadata and pipeline state:

```bash
git status
git add data/**/*.dvc reports/**/*.dvc dvc.lock
git commit -m "Process customer churn batch"
```

The DagsHub DVC remote is configured as `origin`. Publish the cached objects with:

```bash
uv run dvc push
```

MLflow model runs and model versions are already sent directly to DagsHub through
`MLFLOW_TRACKING_URI`; that storage path is separate from the DVC cache.

## Common Workflows

### Full Demonstration

```bash
make install
make lint
make test
make data-pipeline
uv run dvc status
```

Expected final status:

```text
Data and pipelines are up to date.
```

### Demonstrate Rejection

```bash
make generate-data
make process-batch INPUT=data/incoming/invalid.csv
```

Expected behavior:

- Validation reports are written.
- The batch is copied to `data/rejected/invalid.csv`.
- Nothing is copied to `data/accepted/invalid.csv`.
- The underlying command returns exit code `1`.
- `make` reports a failed recipe because rejection is intentionally non-successful.

### Compare Normal and Drifted Reports

```bash
make drift-data CURRENT=data/accepted/normal.csv
make drift-data CURRENT=data/accepted/drifted.csv
```

Inspect:

```text
reports/drift/normal.drift.json
reports/drift/drifted.drift.json
reports/drift/normal.drift.html
reports/drift/drifted.drift.html
```

### Rebuild Only Curated Data and Profile

```bash
uv run dvc repro profile
```

DVC follows dependencies backward. It verifies or reproduces the upstream curation,
validation, and generation stages required by `profile`.

## Logging and Failure Behavior

The CLIs use Python's standard `logging` library with parameterized messages.

| Level | Use |
|---|---|
| `INFO` | Successful generation, acceptance, curation, profiling, and drift completion |
| `WARNING` | Data was processed successfully but rejected by the contract |
| `ERROR` | Configuration, input, artifact, or filesystem operation prevented completion |

Logs contain paths, counts, status, and hashes where useful. They do not print full
datasets, secrets, or customer rows.

Operational errors use actionable domain exceptions:

- `DataGenerationOperationalError`
- `DataValidationOperationalError`
- `IncomingBatchOperationalError`
- `DataCurationError`
- `DataProfileError`
- `DataDriftError`

## Testing

Run all tests:

```bash
make test
```

Run only DataOps unit tests:

```bash
uv run pytest tests/unit/data -q
```

Run only DataOps integration tests:

```bash
uv run pytest tests/integration/data -q
```

The suite covers:

- Reproducible generation.
- All five scenarios.
- Successful and failing schema checks.
- Stable validation result models.
- JSON and Markdown report contracts.
- CSV persistence.
- Accepted and rejected routing.
- Fixed-test isolation.
- Curation ordering and deduplication.
- Invalid-source curation blocking.
- Aggregate profile contents and privacy.
- Feature and target drift separation.
- Structured and visual drift reports.
- CLI exit codes and log levels.

Run code quality checks:

```bash
make lint
```

Validate the dependency lock:

```bash
uv lock --check
```

## Source Map

| File | Purpose |
|---|---|
| [`src/data/settings.py`](./settings.py) | Typed versioned configuration loading |
| [`src/data/generate.py`](./generate.py) | Synthetic scenario generation and CSV persistence |
| [`src/data/schema.py`](./schema.py) | Pandera schema and business rules |
| [`src/data/validate.py`](./validate.py) | Validation execution and failure normalization |
| [`src/data/validation_models.py`](./validation_models.py) | Immutable validation contracts |
| [`src/data/validation_report.py`](./validation_report.py) | JSON and Markdown validation reports |
| [`src/data/ingest.py`](./ingest.py) | Incoming validation and accepted/rejected routing |
| [`src/data/drift.py`](./drift.py) | Evidently adapter and stable drift decisions |
| [`src/data/curate.py`](./curate.py) | Deterministic training-data curation |
| [`src/data/profile.py`](./profile.py) | Aggregate dataset profile |
| [`params.yaml`](../../params.yaml) | Versioned policy and thresholds |
| [`dvc.yaml`](../../dvc.yaml) | DataOps DAG definition |
| [`dvc.lock`](../../dvc.lock) | Exact pipeline dependency and output versions |
| [`Makefile`](../../Makefile) | Repeatable developer commands |

## Current Boundaries

The current DataOps subsystem intentionally does not:

- Connect to a real upstream customer data source.
- Automatically watch the incoming directory without an explicit command.
- Decide whether a model should be promoted.
- Send raw customer records to an LLM.
- Run a permanent monitoring service.

These boundaries keep the current implementation deterministic, local, auditable,
and appropriate for the study-project scope.
