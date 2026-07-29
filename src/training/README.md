# Deterministic Training Guide

This document explains the complete local model-training subsystem for the
customer-churn project. It covers experiment planning, deterministic validation,
data splitting, leakage-safe preprocessing, candidate fitting, evaluation,
selection, failure isolation, artifact persistence, and DVC reproducibility.

Model sophistication is intentionally secondary. The purpose is to demonstrate an
auditable operational path from versioned data and configuration to independently
stored, reloadable model candidates.

## Core Invariants

The training implementation protects these rules:

1. Only `data/curated/training.csv` can enter baseline training.
2. The immutable fixed-test dataset cannot be loaded through the training loader.
3. `customer_id` is never a model feature.
4. The target is separated before fitting.
5. Training and validation are split before preprocessing is fitted.
6. Scalers and encoders learn only from the training partition.
7. Every saved model contains preprocessing and prediction in one pipeline.
8. Only catalog-listed algorithms and parameters can be executed.
9. A plan cannot override versioned metrics or resource limits.
10. Candidate selection is deterministic and does not involve an LLM.
11. One candidate failure does not automatically discard successful candidates.
12. No selection occurs when successful candidates fall below policy.
13. Model and JSON writes are atomic.
14. DVC tracks code, configuration, data, plans, models, and metrics.

## Lifecycle

```text
params.yaml
    |
    +--> training configuration
    +--> catalog defaults
    +--> experiment limits
    +--> primary metric
    |
    v
fallback_plan
    |
    v
artifacts/experiment-plans/fallback.json
    |
    v
Pydantic structural validation
    |
    v
deterministic catalog validation
    |
    v
approved normalized plan
    |
    +-----------------------------+
    |                             |
    v                             v
data/curated/training.csv    approved candidates
    |                             |
    v                             |
stratified train/validation split |
    |                             |
    +-------------+---------------+
                  |
                  v
        preprocessing + classifier
                  |
          +-------+-------+
          |               |
          v               v
 logistic regression   random forest
          |               |
          v               v
    validation metrics and independent artifacts
                  |
                  v
       deterministic candidate selection
```

## Source Responsibilities

| File | Responsibility |
|---|---|
| `settings.py` | Load and validate versioned training policy |
| `catalog.py` | Define allowed estimators, parameters, types, bounds, and defaults |
| `preprocessing.py` | Construct unfitted feature transformations and full pipelines |
| `train.py` | Load data, split, fit candidates, isolate failures, and orchestrate CLI execution |
| `evaluate.py` | Calculate metrics and select the winner deterministically |
| `artifacts.py` | Atomically persist pipelines, metrics, failures, and selection |
| `../agent/schemas.py` | Define strict Pydantic experiment-plan contracts |
| `../agent/planner.py` | Build and persist the deterministic fallback plan |
| `../agent/plan_validator.py` | Approve or reject an untrusted structured plan |

## Configuration

Training policy is versioned in `params.yaml`:

```yaml
project:
  random_seed: 42

experiments:
  maximum_candidates: 3
  minimum_successful_candidates: 1
  primary_metric: roc_auc

training:
  validation_fraction: 0.20
  models:
    logistic_regression:
      C: 1.0
      max_iter: 1000
    random_forest:
      n_estimators: 200
      max_depth: 8
      min_samples_leaf: 2
```

`maximum_candidates` limits resource usage. `minimum_successful_candidates`
controls whether selection may continue after failures. With the current value of
one, either model may fail while the other remains eligible.

`primary_metric` controls candidate ranking. It cannot be changed by an experiment
plan. This prevents an LLM or edited artifact from changing policy indirectly.

## Structured Experiment Plans

The fallback plan is stored at:

```text
artifacts/experiment-plans/fallback.json
```

It contains:

- Schema version.
- Auditable source (`fallback` or eventually `llm`).
- Primary metric.
- Ordered experiments.
- Algorithm identifiers.
- Parameter mappings.
- A short reason for every experiment.
- Non-executable observations.

The plan does not contain Python, shell commands, package names to install,
arbitrary paths, credentials, or customer rows.

Create it directly:

```bash
uv run python -m src.agent.planner
```

Create it through DVC:

```bash
uv run dvc repro fallback_plan
```

### Two Validation Layers

Pydantic performs structural validation:

- Required fields must exist.
- Extra fields are forbidden.
- Enums must contain accepted values.
- Primitive types must be correct.
- At least one experiment is required.
- Text fields cannot be empty or unbounded.

Structural validity does not make a plan safe. `plan_validator.py` then enforces:

- The configured primary metric.
- Maximum experiment count.
- Unique algorithms.
- Catalog membership.
- Allowed parameter names.
- Exact parameter types.
- Inclusive minimum and maximum values.
- Default completion for omitted parameters.

An invalid plan never reaches estimator construction.

## Model Catalog

The catalog is code-owned. An LLM cannot create entries or alter bounds.

### Logistic Regression

| Parameter | Type | Minimum | Maximum | Default |
|---|---|---:|---:|---:|
| `C` | float | 0.01 | 10.0 | 1.0 |
| `max_iter` | integer | 100 | 5000 | 1000 |

Logistic Regression is the interpretable linear baseline. `C` is the inverse
regularization strength: smaller values apply stronger regularization.

### Random Forest

| Parameter | Type | Minimum | Maximum | Default |
|---|---|---:|---:|---:|
| `n_estimators` | integer | 10 | 500 | 200 |
| `max_depth` | integer | 1 | 50 | 8 |
| `min_samples_leaf` | integer | 1 | 100 | 2 |

Random Forest is the nonlinear baseline. Its bounds prevent an experiment plan
from requesting an excessive CPU or memory workload.

Both classifiers use:

- `random_state=42`.
- Balanced class weighting.
- CPU-bounded execution.

Random Forest uses `n_jobs=1` for predictable local and CI resource use.

## Dataset Loading

The default input is:

```text
data/curated/training.csv
```

The public loader requires the parent directory to be named `curated`. Passing
`data/test/fixed_test.csv` or an incoming batch fails before reading it.

Training checks:

- Exact documented columns and order.
- Non-empty dataset.
- No nulls.
- Both binary target classes.
- Identifier exclusion from model features.

These checks are defense in depth. DataOps already validates curated data, but
training does not silently trust a malformed file at its subsystem boundary.

## Split and Leakage Prevention

The dataset is split with scikit-learn `train_test_split`:

```text
test_size=0.20
random_state=42
shuffle=True
stratify=churned
```

Stratification preserves approximately the same churn ratio in training and
validation partitions.

The critical ordering is:

```text
raw curated dataframe
    |
    v
separate identifier, features, and target
    |
    v
split into training and validation
    |
    v
pipeline.fit(training_features, training_target)
```

The incorrect ordering would fit preprocessing on the entire dataset and split
afterward. That leaks validation distribution information into scaling and
encoding, producing optimistic metrics.

The integration test verifies leakage prevention directly by comparing the fitted
`StandardScaler.mean_` values with means calculated from the training partition.

## Preprocessing

There are six numerical features:

```text
age
tenure_months
monthly_spend
support_tickets_90d
late_payments_12m
usage_hours_monthly
```

They are transformed with `StandardScaler`, which learns a training mean and
standard deviation and applies:

```text
z = (x - training_mean) / training_standard_deviation
```

The categorical features are:

```text
plan_type
region
```

They use `OneHotEncoder(handle_unknown="ignore")`. A valid category absent from one
training partition therefore does not crash future inference.

No imputer is used because the data contract rejects null values.

## Complete Serving Pipeline

Each candidate is one scikit-learn `Pipeline`:

```text
Pipeline
  preprocessing: ColumnTransformer
    numerical: StandardScaler
    categorical: OneHotEncoder
  classifier: LogisticRegression or RandomForestClassifier
```

Saving the complete pipeline prevents training-serving skew. The future API will
send raw validated features to `pipeline.predict()` and `pipeline.predict_proba()`;
it will not reimplement scaling or encoding.

## Candidate Evaluation

Candidates are evaluated only on the validation partition.

### ROC-AUC

ROC-AUC measures ranking quality across classification thresholds. A value of
`0.5` is random ranking and `1.0` is perfect ranking.

### PR-AUC

PR-AUC summarizes precision-recall performance and is especially informative for
imbalanced churn data.

### Precision

```text
true positives / (true positives + false positives)
```

Precision answers: among customers predicted to churn, how many actually churned?

### Recall

```text
true positives / (true positives + false negatives)
```

Recall answers: among customers who churned, how many did the model identify?

### F1

```text
2 * precision * recall / (precision + recall)
```

F1 balances precision and recall at the current probability threshold of `0.5`.

### Confusion Matrix

The stored matrix uses labels `[0, 1]`:

```text
[[true negatives, false positives],
 [false negatives, true positives]]
```

Class counts are stored beside the metrics so results retain their evaluation
context.

## Deterministic Selection

Candidates are sorted by:

1. Configured primary metric, descending.
2. ROC-AUC, descending.
3. F1, descending.
4. Recall, descending.
5. Algorithm identifier, ascending.

The final name tie-breaker makes selection independent of input ordering. The LLM
never chooses the winner.

Current results are approximately:

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8979 | 0.7873 | 0.6746 | 0.5700 | 0.8261 |
| Random Forest | 0.8676 | 0.7320 | 0.6846 | 0.6375 | 0.7391 |

Logistic Regression wins because `roc_auc` is the primary metric.

These are validation metrics, not immutable fixed-test results and not a promotion
decision.

## Candidate Failure Isolation

Each candidate is fitted and evaluated independently.

Successful candidates become eligible for selection. Failed candidates produce a
safe summary containing:

- Model identifier.
- Actionable failure reason.

Failures do not include tracebacks, raw records, credentials, or environment
values.

With `minimum_successful_candidates: 1`:

```text
logistic succeeds + forest fails -> select logistic
logistic fails + forest succeeds -> select forest
both fail                     -> stop without selection
```

## Persisted Artifacts

Model artifacts:

```text
artifacts/models/logistic_regression.joblib
artifacts/models/random_forest.joblib
```

Metric artifacts:

```text
artifacts/metrics/logistic_regression.json
artifacts/metrics/random_forest.json
artifacts/metrics/failures.json
artifacts/metrics/selection.json
```

Writes use a temporary sibling followed by atomic replacement. An interrupted
process cannot leave a half-written final file.

`training_seconds` is logged but intentionally excluded from DVC JSON. Wall-clock
duration varies between identical runs and would otherwise produce unstable hashes.

### Joblib Safety

Joblib artifacts use Python pickle semantics. Never load a model file from an
untrusted source. Loading can execute serialized Python objects. This project loads
only artifacts produced by its controlled training pipeline or future trusted
MLflow registry.

## DVC Training Stages

The relevant DAG is:

```text
curate -----------+
                  |
fallback_plan ----+----> train
```

The `fallback_plan` stage depends on:

- Planner and schema code.
- Catalog and settings code.
- Project, experiment, and training parameters.

The `train` stage depends on:

- Curated training CSV.
- Experiment-plan artifact.
- Plan validation code.
- Catalog, preprocessing, evaluation, artifact, settings, and training code.
- Project, experiment, and training parameters.

DVC stores output hashes in `dvc.lock`. Generated models and metrics remain ignored
by Git while their versions remain reproducible through DVC metadata and cache.

## Commands

Install dependencies:

```bash
make install
```

Generate the fallback plan:

```bash
uv run dvc repro fallback_plan
```

Run the connected training path:

```bash
uv run dvc repro train
```

Run the entire DataOps and training graph:

```bash
uv run dvc repro
```

Run training directly:

```bash
make train-models
```

Show tracked metrics:

```bash
uv run dvc metrics show
```

Check whether inputs or outputs changed:

```bash
uv run dvc status
```

## Changing Experiments

Change versioned defaults in `params.yaml`, then run:

```bash
uv run dvc repro train
```

DVC will regenerate the fallback plan because it depends on training parameters,
then retrain because the approved plan changed.

Do not edit `dvc.lock` manually. Do not modify generated JSON only to make a gate
pass. Change versioned policy or code and let DVC reproduce the artifacts.

Promotion thresholds describe required quality. Lowering them solely to pass a
specific model run undermines the gate. Threshold changes should have a documented
business justification.

## Testing

Run all tests:

```bash
make test
```

Run training unit tests:

```bash
uv run pytest tests/unit/training -q
```

Run plan tests:

```bash
uv run pytest tests/unit/agent -q
```

Run the full training integration test:

```bash
uv run pytest tests/integration/training -q
```

The integration test covers:

1. Deterministic synthetic labeled data.
2. Curated CSV persistence.
3. Fallback-plan construction.
4. Plan serialization and reloading.
5. Deterministic plan approval.
6. Stratified splitting.
7. Candidate preprocessing and fitting.
8. Metric calculation and selection.
9. Atomic artifact persistence.
10. Joblib pipeline reloading.
11. Class and probability prediction.
12. Fitted scaler statistics derived only from training rows.

All test artifacts use pytest temporary directories. Tests must never overwrite
real DVC outputs.

## Troubleshooting

### Missing fallback plan

Run:

```bash
uv run dvc repro fallback_plan
```

### Training data does not exist

Run the DataOps path:

```bash
uv run dvc repro curate
```

### Plan rejected

Read the logged reason and check:

- Primary metric matches `params.yaml`.
- Candidate count is within policy.
- Algorithms are unique.
- Parameter names exist in the catalog.
- Values have exact types and remain within bounds.

### DVC reports changed outputs

Restore or reproduce them:

```bash
uv run dvc repro train
```

Tests should not change these outputs. A changed status after tests indicates an
artifact-isolation defect.

### F1 is below threshold

ROC-AUC evaluates ranking while F1 uses one classification threshold. A strong
ROC-AUC with lower F1 can indicate that the default `0.5` cutoff is not the best
operating point. Threshold selection must be deterministic and validated without
using the fixed-test dataset.

## Current Boundaries

Implemented:

- Deterministic local planning.
- Plan validation and catalog constraints.
- Leakage-safe candidate training.
- Validation metrics and selection.
- Candidate failure isolation.
- Atomic local artifacts.
- DVC versioning and caching.
- Reload-and-predict integration testing.

Not implemented yet:

- MLflow experiment runs.
- DagsHub remote tracking.
- Model registration.
- Champion alias management.
- Fixed-test promotion comparison.
- Prediction-threshold optimization.
- LLM-generated plans and automatic fallback routing.
- FastAPI serving and Docker publication.

The next major phase is MLflow and DagsHub integration. Local model files and JSON
metrics will become tracked runs with code, data, plan, parameter, metric, and
artifact lineage.
