# Customer Churn Agentic MLOps — Project Specification

## 1. Document Status

| Field | Value |
|---|---|
| Project | `customer-churn-agentic-mlops` |
| Document | `SPEC.md` |
| Status | Draft for review |
| Version | `0.1.0` |
| Architecture | Single repository |
| Primary objective | Demonstrate a complete automated DataOps, MLOps, and constrained LLMOps lifecycle |

## 2. Executive Summary

This project will implement a small but complete machine-learning lifecycle for a synthetic customer churn use case.

The system will detect a new labeled data batch, validate it, version it, analyze drift, optionally retrain a set of candidate models, track experiments, compare the best candidate with the current champion, promote only an objectively better model, package the promoted model in a FastAPI service, build a Docker image, and publish that image to Docker Hub.

An LLM will participate in the training workflow through LangChain and LangGraph. Its responsibility will be limited to proposing bounded experiment plans and explaining results. It will not validate data, execute arbitrary code, choose the winning model, approve promotion, or control deployment quality gates.

The defining principle is:

> The LLM proposes and explains. Deterministic code validates, trains, evaluates, and promotes.

The first version will prioritize the automation cycle over model sophistication. It will use synthetic data, lightweight scikit-learn models, GitHub Actions for orchestration, DVC for data and pipeline versioning, Evidently for drift analysis, Pandera for data contracts, MLflow with DagsHub for experiment tracking and model registry, FastAPI for inference, and Docker Hub for image publication.

## 3. Project Goals

The project must:

1. Demonstrate the distinction between DataOps, MLOps, and LLMOps responsibilities.
2. Keep the complete implementation in one repository.
3. Detect and process new labeled data batches automatically.
4. Validate all incoming data before it enters the training dataset.
5. Version datasets, generated reports, pipeline stages, and relevant model artifacts.
6. Evaluate feature drift and target drift independently.
7. Use an LLM to propose a constrained experiment plan.
8. Validate every LLM-generated plan before execution.
9. Provide a deterministic fallback plan when the LLM is unavailable or invalid.
10. Train and evaluate multiple lightweight candidate models.
11. Track experiments and register model versions in MLflow through DagsHub.
12. Compare candidates with the current champion through deterministic quality gates.
13. Promote only models that satisfy all configured requirements.
14. Package the full preprocessing and prediction pipeline for serving.
15. Expose the promoted model through a small FastAPI application.
16. Build and test a Docker image automatically.
17. Publish promoted model images to Docker Hub.
18. Run without paid infrastructure requirements.
19. Be reproducible locally and in GitHub Actions.
20. Be understandable enough to serve as a portfolio and learning project.

## 4. Non-Goals

The first version will not include:

- Kubernetes.
- Airflow, Prefect, or Dagster.
- GPU training.
- AutoKeras or TensorFlow.
- Arbitrary model architectures.
- Autonomous code generation.
- Deep Agents.
- Arbitrary shell execution by the LLM.
- A feature store.
- Streaming ingestion.
- Real-time model monitoring.
- Online learning.
- Cloud databases.
- Multi-environment deployment.
- Production-grade authentication for the API.
- Automatic rollback in a runtime environment.
- A paid model-hosting platform.
- A permanent cloud deployment.

Optional free container hosting may be evaluated later, but Docker Hub publication is the required deployment artifact for version one.

## 5. Business Scenario

The project will predict whether a fictional subscription customer is likely to churn.

This scenario is selected because it supports:

- Binary classification.
- Numerical and categorical features.
- Class imbalance.
- Probability predictions.
- Understandable business metrics.
- Synthetic generation of normal, drifted, and invalid batches.
- Clear demonstration of retraining and promotion decisions.

### 5.1 Initial Dataset Fields

| Field | Type | Description |
|---|---|---|
| `customer_id` | string | Unique customer identifier |
| `age` | integer | Customer age |
| `tenure_months` | integer | Number of months as a customer |
| `monthly_spend` | float | Average monthly spend |
| `support_tickets_90d` | integer | Support tickets opened in the last 90 days |
| `late_payments_12m` | integer | Late payments in the last 12 months |
| `usage_hours_monthly` | float | Average product usage per month |
| `plan_type` | category | Subscription plan |
| `region` | category | Customer region |
| `churned` | integer | Target: `0` for retained and `1` for churned |

The precise ranges, accepted categories, nullability rules, and cross-column validation rules will be defined in the Pandera schema.

### 5.2 Synthetic Batch Types

The data generator must produce at least the following scenarios:

| Batch | Purpose | Expected pipeline behavior |
|---|---|---|
| Reference dataset | Baseline distribution and initial training data | Accepted and versioned |
| Fixed test dataset | Immutable final comparison dataset | Never merged into training data |
| Normal batch | Valid data with little or no significant drift | Validate, version, report, normally skip retraining |
| Drifted batch | Valid data with intentional distribution changes | Validate, report drift, start experiment workflow |
| Invalid batch | Schema or business-rule violations | Reject before curation or training |

All synthetic generation must be deterministic when the same seed and parameters are used.

## 6. Core Design Principles

### 6.1 Deterministic Quality Gates

The following decisions must be made only by deterministic code:

- Whether incoming data satisfies the contract.
- Whether drift thresholds were exceeded.
- Whether an experiment plan is safe and valid.
- Which trained candidate achieved the best configured metric.
- Whether the candidate satisfies promotion thresholds.
- Whether the model artifact can be loaded.
- Whether API contract and smoke tests pass.
- Whether a Docker image may be published.

### 6.2 Constrained LLM Participation

The LLM may:

- Inspect structured data-profile summaries.
- Inspect structured drift summaries.
- Inspect the allowed model catalog.
- Inspect configured resource limits.
- Inspect previous experiment summaries.
- Propose a bounded experiment plan.
- Explain why selected experiments are reasonable.
- Generate a human-readable final analysis from verified metrics.

The LLM must not:

- Receive raw customer rows unless explicitly introduced in a future version.
- Generate or execute Python code.
- Generate or execute shell commands.
- Install packages.
- Select models outside the allowlist.
- Set parameters outside approved ranges.
- Change quality thresholds.
- Skip required workflow stages.
- Select the winning model.
- Promote a model.
- Publish an image.
- Invent metrics or pipeline results.

### 6.3 Reproducibility

Every training run must be traceable to:

- Git commit.
- DVC data version or hash.
- Dataset profile.
- Drift report.
- Experiment plan.
- Model parameters.
- Random seed.
- Dependency lock file.
- Training and evaluation metrics.
- Registered model version.
- Docker image tags when promoted.

### 6.4 Training-Serving Consistency

The served artifact must contain the complete preprocessing and prediction pipeline.

The API must not reproduce preprocessing manually. It must load one artifact that includes feature selection, categorical encoding, numerical transformation, and the trained classifier.

## 7. High-Level Architecture

    New labeled batch
            |
            v
    GitHub Actions or local command
            |
            v
    Pandera validation
            |
      invalid ------------------------> reject batch and publish report
            |
          valid
            v
    DVC versioning and curation
            |
            v
    Evidently drift analysis
            |
            +---- no significant drift ----> publish report and stop
            |
            +---- significant drift
                         |
                         v
                 Build dataset profile
                         |
                         v
              LangChain experiment planner
                         |
                         v
              Deterministic plan validator
                         |
                         v
                 Train approved candidates
                         |
                         v
              Log runs to DagsHub MLflow
                         |
                         v
                 Select best candidate
                         |
                         v
                 Compare with champion
                         |
                  +------+------+
                  |             |
               rejected      promoted
                  |             |
                  v             v
             log report    assign champion alias
                                |
                                v
                        export full pipeline
                                |
                                v
                         test FastAPI service
                                |
                                v
                         build Docker image
                                |
                                v
                          push Docker Hub

## 8. Repository Structure

    customer-churn-agentic-mlops/
    ├── .github/
    │   └── workflows/
    │       ├── ci.yml
    │       ├── data-pipeline.yml
    │       └── publish-image.yml
    │
    ├── data/
    │   ├── reference/
    │   ├── incoming/
    │   ├── accepted/
    │   ├── curated/
    │   └── test/
    │
    ├── artifacts/
    │   ├── models/
    │   ├── metrics/
    │   └── experiment-plans/
    │
    ├── reports/
    │   ├── data-quality/
    │   ├── drift/
    │   └── training/
    │
    ├── prompts/
    │   ├── experiment-planner.prompt.md
    │   └── result-analyst.prompt.md
    │
    ├── src/
    │   ├── data/
    │   │   ├── generate.py
    │   │   ├── schema.py
    │   │   ├── validate.py
    │   │   ├── profile.py
    │   │   ├── drift.py
    │   │   └── curate.py
    │   │
    │   ├── agent/
    │   │   ├── llm.py
    │   │   ├── schemas.py
    │   │   ├── planner.py
    │   │   ├── plan_validator.py
    │   │   └── analyst.py
    │   │
    │   ├── training/
    │   │   ├── catalog.py
    │   │   ├── preprocessing.py
    │   │   ├── train.py
    │   │   ├── evaluate.py
    │   │   ├── compare.py
    │   │   └── registry.py
    │   │
    │   ├── workflow/
    │   │   ├── state.py
    │   │   ├── nodes.py
    │   │   ├── conditions.py
    │   │   └── graph.py
    │   │
    │   └── api/
    │       ├── main.py
    │       ├── schemas.py
    │       ├── model_loader.py
    │       └── settings.py
    │
    ├── tests/
    │   ├── unit/
    │   ├── integration/
    │   └── contract/
    │
    ├── dvc.yaml
    ├── dvc.lock
    ├── params.yaml
    ├── pyproject.toml
    ├── uv.lock
    ├── Dockerfile
    ├── Makefile
    ├── SPEC.md
    └── README.md

The future `AGENTS.md` file will define implementation conventions and instructions for coding agents. It is intentionally excluded from this specification revision until separately reviewed.

## 9. Technology Stack

| Area | Technology |
|---|---|
| Language | Python |
| Dependency management | `uv` |
| Data processing | pandas |
| Data validation | Pandera |
| Drift analysis | Evidently |
| Data and pipeline versioning | DVC |
| Model training | scikit-learn |
| Experiment tracking | MLflow |
| Remote ML platform | DagsHub |
| LLM application framework | LangChain |
| Workflow graph | LangGraph |
| Initial LLM provider | GitHub Models |
| API framework | FastAPI |
| Request and response validation | Pydantic |
| ASGI server | Uvicorn |
| Containerization | Docker |
| Container registry | Docker Hub |
| Automation | GitHub Actions |
| Code quality | Ruff |
| Testing | pytest |

The LLM provider must be abstracted so it can be replaced without changing the domain workflow.

## 10. Functional Requirements

### FR-001 — Generate Synthetic Data

The system must provide commands or scripts to generate reference, test, normal, drifted, and invalid datasets.

Acceptance conditions:

- Generation supports a configurable random seed.
- Repeating the same inputs produces identical outputs.
- Drifted batches intentionally modify selected feature distributions.
- Invalid batches intentionally violate one or more documented schema rules.

### FR-002 — Detect New Incoming Data

The automated data workflow must run when a file under `data/incoming/` is added or changed.

The same workflow must be executable manually through GitHub Actions and locally through a Makefile or Python command.

### FR-003 — Validate the Data Contract

Every incoming batch must be validated before curation.

Validation must cover:

- Required columns.
- Unexpected columns when strict mode is enabled.
- Data types.
- Nullable rules.
- Numerical ranges.
- Allowed categories.
- Target values.
- Unique customer identifiers within the batch.
- Duplicate rows.
- Minimum batch size.
- Cross-column business rules where relevant.

An invalid batch must:

- Stop the training path.
- Produce a machine-readable validation report.
- Produce a human-readable summary.
- Remain outside the accepted and curated datasets.

### FR-004 — Curate Accepted Data

A valid incoming batch must be added to the accepted data history and merged into a curated training dataset according to deterministic rules.

The fixed test dataset must never be merged into training data.

The curated dataset must preserve a documented feature schema and target definition.

### FR-005 — Version Data and Pipeline Outputs

DVC must version the following where appropriate:

- Reference data.
- Fixed test data.
- Accepted batches.
- Curated data.
- Data profiles.
- Drift reports.
- Experiment plans.
- Selected model artifacts.
- Evaluation metrics.

The pipeline must be reproducible through `dvc repro` or an equivalent project command.

### FR-006 — Build a Dataset Profile

The system must generate a structured profile containing at least:

- Row count.
- Feature count.
- Feature names and types.
- Missing-value counts.
- Numerical summaries.
- Categorical frequencies.
- Target distribution.
- Duplicate count.
- Drifted feature list.
- Data-version identifier.

The profile given to the LLM must not contain raw customer rows.

### FR-007 — Evaluate Drift

Evidently must compare the current accepted batch or curated dataset against the configured reference dataset.

The system must evaluate separately:

- Feature drift.
- Target drift.

The drift result must include:

- Drift status per feature.
- Drift score or test result per feature.
- Share of drifted features.
- Target drift status.
- Configured thresholds.
- Overall retraining recommendation according to deterministic rules.

The initial policy is:

- No significant feature drift: publish reports and skip automatic retraining.
- Significant feature drift: start the candidate-training workflow.
- Manual retraining must remain available regardless of drift result.

### FR-008 — Define the Allowed Model Catalog

The first version must support:

- Logistic Regression.
- Random Forest Classifier.
- HistGradientBoosting Classifier.

The model catalog must be implemented in Python and include:

- Algorithm identifier.
- Constructor.
- Allowed parameters.
- Parameter types.
- Minimum and maximum values.
- Default configuration.

The LLM cannot add new catalog entries.

### FR-009 — Provide a Deterministic Baseline Plan

The project must include a default experiment plan that can run without an LLM.

The fallback plan must:

- Train at least one interpretable baseline.
- Train at least one nonlinear model.
- Respect the same resource limits as LLM plans.
- Be used when LLM inference fails, times out, returns invalid output, or is disabled.

### FR-010 — Generate a Constrained Experiment Plan

The LangChain planner must receive only structured summaries and must return a Pydantic-validated `ExperimentPlan`.

The plan must include:

- Primary metric.
- Selected allowed algorithms.
- Parameters for each experiment.
- A short reason for each experiment.
- Observations derived from the supplied summaries.

The plan must respect configured limits such as:

- Maximum number of models.
- Maximum search iterations.
- CPU-only execution.
- Maximum training duration where enforceable.
- Allowed metrics.

### FR-011 — Validate the Experiment Plan

A deterministic validator must check:

- Every algorithm is allowlisted.
- Every parameter is allowlisted.
- Parameter values match expected types.
- Parameter values remain within configured bounds.
- The experiment count does not exceed the limit.
- The selected primary metric is supported.
- No arbitrary code, command, package, or file path is present.

Invalid plans must be rejected and replaced by the fallback plan.

### FR-012 — Train Candidate Models

The training system must:

- Split training and validation data before fitting preprocessing transformations.
- Use a scikit-learn `Pipeline` or equivalent full serving artifact.
- Fit preprocessing only on training data.
- Use deterministic random seeds where supported.
- Train all approved candidate experiments.
- Prevent the fixed test dataset from influencing parameter selection.
- Persist each candidate result independently.

### FR-013 — Evaluate Candidate Models

Each candidate must be evaluated using at least:

- ROC-AUC.
- PR-AUC.
- F1 score.
- Precision.
- Recall.
- Confusion matrix.
- Class distribution.

Accuracy may be recorded but must not be the only promotion metric.

The system must select the best candidate deterministically using the configured primary metric and tie-breaking rules.

### FR-014 — Track Experiments

Every candidate run must log to MLflow:

- Algorithm.
- Hyperparameters.
- Git commit.
- DVC data version.
- Random seed.
- Dataset profile.
- Drift summary.
- Experiment-plan artifact.
- Validation metrics.
- Test metrics when applicable.
- Confusion matrix.
- Full preprocessing and model pipeline.
- Training duration.
- Dependency or environment metadata where practical.

DagsHub will provide the initial remote MLflow tracking server and model registry.

### FR-015 — Register Model Versions

The selected candidate must be registered for traceability even when it is not promoted.

Registered versions must retain links to their MLflow run and relevant artifacts.

The current serving model must be addressable through the `champion` alias.

A non-promoted model must not replace the `champion` alias.

### FR-016 — Compare Candidate With Champion

The promotion comparison must be deterministic.

The initial example policy is:

- Candidate ROC-AUC is at least `0.80`.
- Candidate F1 is at least `0.70`.
- Candidate recall is at least `0.65`.
- Candidate ROC-AUC exceeds champion ROC-AUC by at least `0.005`.
- The model artifact loads successfully.
- FastAPI integration tests pass.

All thresholds must be configurable through `params.yaml`.

When no champion exists, the first model may be promoted only if it satisfies all absolute minimum thresholds.

### FR-017 — Generate a Final Analysis

After evaluation, the LLM may generate a human-readable report from verified structured results.

The report may explain:

- What changed in the data.
- Which features drifted.
- Which experiments were attempted.
- Which candidate performed best.
- Why promotion passed or failed.
- Which future investigation may be useful.

All numerical values must come from deterministic pipeline outputs.

The report must be stored as an artifact and must not influence the promotion result.

### FR-018 — Serve the Promoted Model

The FastAPI application must expose:

- `GET /health`
- `GET /model-info`
- `POST /predict`

`GET /health` must confirm that the process is running and the model is loaded.

`GET /model-info` must return non-sensitive model metadata such as:

- Registered model name.
- Model version.
- MLflow run identifier when available.
- Git commit when available.
- Training-data version when available.

`POST /predict` must:

- Validate input through Pydantic.
- Preserve the expected feature order.
- Call the complete trained pipeline.
- Return the predicted class.
- Return the churn probability when supported.
- Return model-version metadata.

The serving application must not invoke the LLM.

### FR-019 — Build a Minimal Serving Image

The serving image must include only runtime requirements.

Training-only tools such as DVC, Evidently, LangChain, LangGraph, and MLflow client components not required at runtime should not be included unless technically necessary for artifact loading.

The image must:

- Start the FastAPI application.
- Load the promoted model artifact.
- Pass a container smoke test.
- Expose the configured API port.
- Use a non-root runtime user where practical.

### FR-020 — Publish Docker Images

A successful promotion must trigger image publication.

Required tags:

- `latest`
- `model-v<registered-model-version>`
- `<git-sha>`

The `latest` tag must move only after all promotion and image tests pass.

A rejected candidate must not publish or retag an image.

## 11. LangGraph Workflow Specification

The graph must contain explicit nodes equivalent to:

    validate_data
        |
        v
    version_data
        |
        v
    calculate_drift
        |
        v
    should_retrain?
        +---- false ----> create_no_retrain_report
        |
        +---- true
                 |
                 v
             build_profile
                 |
                 v
             generate_experiment_plan
                 |
                 v
             validate_experiment_plan
                 |
                 v
             train_candidates
                 |
                 v
             evaluate_candidates
                 |
                 v
             compare_with_champion
                 |
                 v
             should_promote?
                 +---- false ----> rejection_report
                 |
                 +---- true -----> promote_model
                                      |
                                      v
                              generate_final_analysis

Only these nodes may require LLM inference:

- `generate_experiment_plan`
- `generate_final_analysis`

Workflow state must contain references and structured summaries rather than full raw datasets.

Example state fields:

| Field | Purpose |
|---|---|
| `incoming_data_path` | New batch location |
| `curated_data_path` | Curated training dataset |
| `test_data_path` | Immutable test dataset |
| `validation_report_path` | Data-contract result |
| `drift_report_path` | Evidently report |
| `dataset_profile_path` | Structured profile |
| `experiment_plan_path` | Approved or fallback plan |
| `candidate_run_ids` | MLflow candidate runs |
| `best_candidate_run_id` | Selected candidate |
| `promotion_result` | Approved or rejected |
| `registered_model_version` | Registered version when created |
| `final_report_path` | Human-readable analysis |

## 12. DVC Pipeline Specification

The initial `dvc.yaml` should model stages equivalent to:

    validate
        |
        v
    curate
        |
        v
    profile
        |
        v
    drift
        |
        v
    plan
        |
        v
    train
        |
        v
    evaluate
        |
        v
    compare
        |
        v
    register

The exact division between DVC stages and LangGraph orchestration may be adjusted during implementation, but the following must remain true:

- Each deterministic stage has declared inputs and outputs.
- Re-running unchanged stages should reuse DVC caching where applicable.
- LLM outputs are stored as versioned artifacts.
- Promotion remains deterministic.
- The workflow can be executed locally and in CI.

## 13. Configuration

`params.yaml` must contain versioned non-secret configuration, including:

- Random seeds.
- Dataset-generation settings.
- Validation thresholds.
- Drift thresholds.
- Model-catalog limits.
- Experiment-count limits.
- Primary metric.
- Promotion thresholds.
- Model name.
- API metadata defaults.

Secrets must not be stored in `params.yaml`, source code, notebooks, DVC metadata, or committed environment files.

Expected CI secrets include:

- `DAGSHUB_USERNAME`
- `DAGSHUB_TOKEN`
- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

GitHub Models should use the workflow-provided GitHub token where supported. The project must support disabling LLM use and falling back to the deterministic plan.

## 14. GitHub Actions Workflows

### 14.1 `ci.yml`

Triggers:

- Pull requests.
- Pushes to relevant application and test files.

Responsibilities:

- Install locked dependencies.
- Run Ruff checks.
- Run formatting verification.
- Run unit tests.
- Run data-schema tests.
- Run experiment-plan validator tests.
- Run FastAPI contract tests.
- Verify the Docker image builds.

Security requirements:

- External publishing secrets must not be exposed to untrusted pull requests.
- CI must not register, promote, or publish models.

### 14.2 `data-pipeline.yml`

Triggers:

- Changes under `data/incoming/**`.
- Relevant changes to `params.yaml`.
- Relevant changes under `src/data/**`.
- Relevant changes under `src/training/**`.
- Relevant changes under `src/agent/**`.
- Manual execution through `workflow_dispatch`.

Responsibilities:

- Restore the Python environment.
- Pull required DVC artifacts.
- Validate incoming data.
- Run curation and drift analysis.
- Execute the LangGraph workflow.
- Generate or select an experiment plan.
- Train and evaluate candidates when required.
- Log experiments.
- Register the selected candidate.
- Apply the promotion policy.
- Publish reports and pipeline artifacts.
- Trigger the image workflow only after promotion.

### 14.3 `publish-image.yml`

Trigger:

- Successful model promotion through an explicit workflow handoff or reusable workflow call.

Responsibilities:

- Resolve the promoted model version.
- Obtain or export the complete serving artifact.
- Build the runtime image.
- Start the container.
- Execute health and prediction smoke tests.
- Push required Docker Hub tags.

## 15. Testing Strategy

### 15.1 Unit Tests

Unit tests must cover:

- Synthetic data generation.
- Pandera schema rules.
- Data-profile creation.
- Drift-result parsing.
- Model-catalog constraints.
- Experiment-plan validation.
- Fallback-plan selection.
- Metric calculations.
- Candidate selection.
- Promotion policy.
- API request and response schemas.
- Model metadata formatting.

### 15.2 Integration Tests

Integration tests must cover:

- Valid batch through curation.
- Invalid batch rejection.
- Drifted batch through experiment planning.
- LLM failure followed by fallback-plan execution.
- Training and MLflow logging against a test or mocked backend.
- Model registration logic.
- Champion comparison.
- Loading the complete serving pipeline.
- FastAPI prediction using a real generated model artifact.

### 15.3 Contract Tests

Contract tests must verify:

- Input feature names and types.
- Output response structure.
- `/health` behavior.
- `/model-info` behavior.
- Model artifact compatibility with the API image.

### 15.4 End-to-End Scenarios

The project must demonstrate at least these scenarios:

1. Invalid data is rejected before training.
2. Normal valid data produces reports and skips retraining.
3. Drifted valid data triggers candidate training.
4. A weaker candidate is registered but not promoted.
5. A stronger candidate is promoted.
6. Promotion causes a Docker image to be built and tagged.
7. LLM unavailability causes deterministic fallback rather than pipeline failure.

## 16. Observability and Artifacts

Each pipeline execution must make the following results available through logs or artifacts:

- Validation summary.
- Validation failure details when applicable.
- Data profile.
- Drift summary.
- Evidently report.
- Approved or fallback experiment plan.
- Candidate metrics table.
- Champion comparison.
- Promotion decision and reasons.
- Final LLM-generated analysis when available.
- Registered model version.
- Published Docker image tags when applicable.

Logs must not include:

- Authentication tokens.
- Full raw datasets.
- Sensitive environment-variable values.
- Unbounded LLM prompts containing raw data.

## 17. Failure Behavior

| Failure | Required behavior |
|---|---|
| Invalid incoming data | Fail data path, publish validation report, do not curate or train |
| DVC pull failure | Stop workflow with actionable error |
| Drift-report failure | Stop retraining decision; do not guess |
| LLM unavailable | Use deterministic fallback plan |
| Invalid LLM plan | Reject plan and use fallback |
| Candidate training failure | Record failed experiment and continue only when policy allows |
| All candidates fail | Stop before registration or promotion |
| MLflow logging failure | Stop promotion because traceability is incomplete |
| Registry failure | Stop image workflow |
| Candidate below threshold | Register for traceability, do not promote |
| API integration failure | Do not promote or publish |
| Docker smoke-test failure | Do not push or retag `latest` |
| Docker Hub push failure | Preserve promotion records and report publication failure |

## 18. Security Requirements

- No credentials may be committed.
- `.env.example` must contain names only, not real values.
- CI secrets must use GitHub encrypted secrets.
- Pull-request workflows must not receive publishing secrets.
- LLM input must contain summaries rather than raw data.
- LLM output must be treated as untrusted input.
- The plan validator must reject unknown fields when practical.
- The API container should run as a non-root user where practical.
- Runtime images should not contain training datasets.
- Docker Hub access should use a scoped access token rather than an account password.

## 19. Performance and Cost Constraints

The project is designed for GitHub-hosted CPU runners and local development.

Initial limits should include:

- Small synthetic datasets.
- Maximum of three candidate models per automatic run.
- Bounded hyperparameter combinations.
- No GPU requirement.
- No long-running neural-network training.
- No mandatory paid external compute.
- No permanent hosted API requirement.

The experiment-plan validator must enforce resource limits rather than trusting the LLM to respect them.

## 20. Implementation Phases

### Phase 1 — Repository Foundation

Deliverables:

- Repository structure.
- `pyproject.toml`.
- `uv.lock`.
- Ruff configuration.
- pytest configuration.
- `Makefile`.
- `.gitignore`.
- `.env.example`.
- Initial CI workflow.

Exit criteria:

- `make install`, `make lint`, and `make test` succeed locally.

### Phase 2 — Synthetic Data and Data Contract

Deliverables:

- Deterministic data generator.
- Reference and fixed test datasets.
- Normal, drifted, and invalid batch generation.
- Pandera schema.
- Validation reports.

Exit criteria:

- Valid datasets pass.
- Invalid datasets fail with expected errors.
- Generation is reproducible.

### Phase 3 — DVC Data Pipeline

Deliverables:

- DVC initialization.
- Data tracking.
- Curation stage.
- Profile stage.
- Drift stage.
- Reproducible local commands.

Exit criteria:

- `dvc repro` produces expected artifacts.
- Unchanged stages are reused where applicable.

### Phase 4 — Deterministic Baseline Training

Deliverables:

- Preprocessing pipeline.
- Model catalog.
- Fallback experiment plan.
- Candidate training.
- Evaluation metrics.
- Candidate selection.

Exit criteria:

- Multiple candidates train successfully.
- The best candidate is selected deterministically.
- No preprocessing leakage exists.

### Phase 5 — MLflow and DagsHub Integration

Deliverables:

- Remote tracking configuration.
- Per-candidate MLflow runs.
- Model artifact logging.
- Model registration.
- Champion alias handling.

Exit criteria:

- A run can be traced from code and data version to registered model artifact.

### Phase 6 — LLM Experiment Planner

Deliverables:

- Provider abstraction.
- GitHub Models integration.
- Structured experiment-plan schema.
- Plan validator.
- Fallback behavior.
- Planner tests.

Exit criteria:

- Valid plans execute.
- Invalid plans are rejected.
- Provider failure triggers fallback.

### Phase 7 — LangGraph Orchestration

Deliverables:

- Workflow state.
- Deterministic nodes.
- Conditional routing.
- LLM planner node.
- Final analysis node.
- Local workflow entry point.

Exit criteria:

- Normal, drifted, invalid, rejected, and promoted paths are reproducible.

### Phase 8 — Promotion Policy

Deliverables:

- Champion lookup.
- Candidate comparison.
- Configurable thresholds.
- Promotion report.
- Alias update.

Exit criteria:

- A weaker candidate cannot replace the champion.
- A qualifying candidate can be promoted.

### Phase 9 — FastAPI Serving Layer

Deliverables:

- `/health`.
- `/model-info`.
- `/predict`.
- Pydantic contracts.
- Model loader.
- Integration and contract tests.

Exit criteria:

- The API loads the complete pipeline and returns reproducible predictions.

### Phase 10 — Container and Docker Hub Publication

Deliverables:

- Minimal Dockerfile.
- Container smoke tests.
- Docker Hub publishing workflow.
- Immutable model-version and Git-SHA tags.
- Controlled `latest` tag.

Exit criteria:

- Only a promoted model produces a published image.
- The published image starts and answers health and prediction requests.

### Phase 11 — Documentation and Demonstration

Deliverables:

- README with architecture and commands.
- Example reports.
- Demonstration scenarios.
- Local execution guide.
- GitHub Actions explanation.
- DagsHub and Docker Hub links.

Exit criteria:

- A reviewer can understand and reproduce the complete lifecycle.

## 21. Project-Level Acceptance Criteria

The first version is complete when all of the following are true:

1. The repository can be installed from the lock file.
2. Linting and tests pass locally and in GitHub Actions.
3. Synthetic reference, test, normal, drifted, and invalid datasets can be generated.
4. Invalid data is rejected before curation.
5. Valid data is versioned and profiled.
6. Evidently creates separate feature and target drift results.
7. No-drift data can complete without unnecessary retraining.
8. Drifted data can trigger candidate training.
9. The LLM can propose only allowlisted experiments.
10. Invalid or unavailable LLM output uses the fallback plan.
11. Multiple scikit-learn candidates are trained without preprocessing leakage.
12. Candidate metrics and artifacts are logged to MLflow.
13. The selected candidate is registered.
14. The candidate is compared with the champion deterministically.
15. A failing candidate cannot become champion.
16. A passing candidate receives the champion alias.
17. The FastAPI service loads the complete promoted pipeline.
18. API contract and smoke tests pass.
19. A promoted model produces Docker image tags for `latest`, model version, and Git SHA.
20. A rejected model produces no deployment image.
21. The complete lifecycle can run without paid compute infrastructure.
22. Documentation explains DataOps, MLOps, and LLMOps responsibilities clearly.

## 22. Future Extensions

The following may be evaluated after version one:

- Optuna for deterministic hyperparameter optimization.
- AutoKeras as an optional model-catalog entry.
- TensorFlow models.
- Additional LLM providers.
- Local LLM inference.
- Manual approval before champion promotion.
- Scheduled drift checks.
- Delayed-label simulation.
- Prediction monitoring.
- Data-quality trend dashboards.
- Model calibration monitoring.
- Explainability reports.
- Free demonstration hosting for the API.
- Trivy container scanning.
- Software bill of materials generation.
- Kubernetes deployment as a separate learning project.

These extensions must not be added until the first complete automation cycle is working.

## 23. Open Decisions for Review

The following decisions should be confirmed before implementation begins:

1. Final repository name.
2. Python version.
3. Exact DVC remote configuration in DagsHub.
4. Exact GitHub Models model identifier.
5. Initial data ranges and category values.
6. Exact drift thresholds.
7. Primary promotion metric.
8. Absolute and relative promotion thresholds.
9. Whether every selected candidate or only the best candidate is registered.
10. Whether image publication is a separate workflow or a reusable job in the data pipeline.
11. Whether the promoted model artifact is downloaded during the Docker build or exported earlier as a workflow artifact.
12. Whether the first version requires manual approval before updating the `champion` alias.

## 24. Final Project Definition

This project combines:

- DataOps for validating, curating, profiling, and versioning data.
- MLOps for reproducible training, experiment tracking, model registration, comparison, promotion, and image publication.
- Constrained agentic experimentation for bounded LLM-generated experiment plans.
- Basic LLMOps for structured prompts, validated outputs, provider abstraction, fallback behavior, and result traceability.

The model itself is intentionally simple. The primary learning objective is the automated and auditable lifecycle from new data to a published inference image.
