# Customer Churn Agentic MLOps — Project Specification

## 1. Document Status

| Field | Value |
|---|---|
| Project | `customer-churn-agentic-mlops` |
| Document | `SPEC.md` |
| Status | Approved implementation scope |
| Version | `0.2.0` |
| Architecture | Single repository |
| Primary objective | Demonstrate a complete automated DataOps, MLOps, and constrained LLMOps lifecycle |

## 2. Executive Summary

This project will implement a small but complete machine-learning lifecycle for a synthetic customer churn use case.

The system will detect a new labeled data batch, validate it, version it, analyze drift, optionally retrain a set of candidate models, track experiments, compare the best candidate with the current champion, promote only an objectively better model, and expose the promoted pipeline through FastAPI.

An optional LLM may participate through a small provider abstraction. Its responsibility is limited to proposing bounded experiment plans and explaining verified results. LangChain and LangGraph are not required for version one because the workflow does not need their additional abstraction. The LLM will not validate data, execute arbitrary code, choose the winning model, approve promotion, or control serving quality gates.

The defining principle is:

> The LLM proposes and explains. Deterministic code validates, trains, evaluates, and promotes.

The first version will prioritize the automated DataOps and MLOps cycle over model sophistication or agent complexity. It will use synthetic data, lightweight scikit-learn models, Python and GitHub Actions for orchestration, DVC for data and pipeline versioning, Evidently for drift analysis, Pandera for data contracts, MLflow with DagsHub for experiment tracking and model registry, and FastAPI for local inference.

## 3. Project Goals

The project must:

1. Demonstrate the distinction between DataOps, MLOps, and LLMOps responsibilities.
2. Keep the complete implementation in one repository.
3. Detect and process new labeled data batches automatically.
4. Validate all incoming data before it enters the training dataset.
5. Version datasets, generated reports, pipeline stages, and relevant model artifacts.
6. Evaluate feature drift and target drift independently.
7. Optionally use an LLM to propose a constrained experiment plan.
8. Validate every LLM-generated plan before execution.
9. Provide a deterministic fallback plan when the LLM is unavailable or invalid.
10. Train and evaluate multiple lightweight candidate models.
11. Track experiments and register model versions in MLflow through DagsHub.
12. Compare candidates with the current champion through deterministic quality gates.
13. Promote only models that satisfy all configured requirements.
14. Package the full preprocessing and prediction pipeline for serving.
15. Expose the promoted model through a small FastAPI application.
16. Run without paid infrastructure requirements.
17. Be reproducible locally and in GitHub Actions.
18. Be understandable enough to serve as a portfolio and learning project.

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
- Docker Hub publication or another container-registry publication flow.
- Automated deployment of the FastAPI application.
- LangChain or LangGraph as required orchestration dependencies.

The existing Dockerfile may remain as an optional local-development convenience. Building, publishing, or deploying a container is not part of the version-one acceptance criteria.

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
- Whether API contract and integration tests pass.

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
              Optional experiment planner
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

## 8. Repository Structure

    customer-churn-agentic-mlops/
    ├── .github/
    │   └── workflows/
    │       ├── ci.yml
    │       ├── data-pipeline.yml
    │       └── promote-model.yml
    │
    ├── data/
    │   ├── reference/
    │   ├── incoming/
    │   ├── accepted/
    │   ├── rejected/
    │   ├── curated/
    │   └── test/
    │
    ├── artifacts/
    │   ├── models/
    │   ├── metrics/
    │   ├── experiment-plans/
    │   └── agent/
    │
    ├── reports/
    │   ├── data-quality/
    │   ├── data-profile/
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
    │   │   └── local_pipeline.py
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

`AGENTS.md` defines implementation conventions and instructions for coding agents. `SPEC.md` remains authoritative for product scope and required behavior.

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
| Optional LLM integration | Provider-neutral OpenAI-compatible interface |
| Initial LLM provider | Groq free tier |
| Workflow orchestration | Python application service, DVC, and GitHub Actions |
| API framework | FastAPI |
| Request and response validation | Pydantic |
| ASGI server | Uvicorn |
| Automation | GitHub Actions |
| Code quality | Ruff |
| Testing | pytest |

The LLM provider must be abstracted so it can be replaced through environment configuration without changing the domain workflow. LangChain may be introduced later only if it removes meaningful provider or structured-output complexity.

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

These two candidates are sufficient for version one because they provide one interpretable baseline and one nonlinear baseline. Additional models are optional extensions rather than lifecycle requirements.

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

The optional planner must return a Pydantic-validated `ExperimentPlan` through the provider-neutral interface.

The planner may receive only bounded policy information and structured summaries. It must never receive raw customer rows or credentials. Data-profile and drift context may be added when it materially improves experiment proposals, but version one does not require the LLM to influence model quality.

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

Candidate selection must continue to use the training/validation split. Only the selected registered candidate may then be evaluated against the immutable fixed-test dataset for promotion. The current champion must be evaluated against the same fixed-test dataset and metric implementation so the comparison remains fair.

The initial example policy is:

- Candidate ROC-AUC is at least `0.80`.
- Candidate F1 is at least `0.70`.
- Candidate recall is at least `0.65`.
- Candidate ROC-AUC exceeds champion ROC-AUC by at least `0.005`.
- The model artifact loads successfully.
- The complete artifact produces a valid prediction through the FastAPI-compatible serving contract.

All thresholds must be configurable through `params.yaml`.

When no champion exists, the first model may be promoted only if it satisfies all absolute minimum thresholds.

### FR-017 — Generate a Final Analysis

After evaluation, the system must generate a human-readable report from verified structured results. Deterministic report generation is sufficient. An LLM may optionally improve the narrative when available.

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

## 11. Workflow Orchestration Specification

Version one uses explicit Python orchestration together with DVC stages and GitHub Actions. LangGraph is not required.

The workflow must preserve an equivalent sequence:

    validate_data
        |
        +---- invalid ----> reject batch, write reports, stop
        |
        v
    version_data
        |
        v
    calculate_drift
        |
        +---- no significant drift ----> write reports, stop automatic training
        |
        v
    curate_and_profile
        |
        v
    generate_or_load_experiment_plan
        |
        v
    validate_experiment_plan
        |
        v
    train_and_evaluate_candidates
        |
        v
    log_and_register_selected_candidate
        |
        v
    compare_with_champion
        |
        +---- rejected ----> write promotion report
        |
        +---- approved ----> manual promotion may assign champion alias
                                  |
                                  v
                          generate final analysis

Only experiment planning and optional narrative analysis may require LLM inference. Every route and quality gate must remain deterministic and testable without an LLM.

The orchestrator should pass paths, identifiers, and structured summaries between operations rather than complete datasets. It may call existing subsystem functions directly; a workflow framework should be introduced only if branching, persistence, or recovery complexity later justifies it.

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

The exact division between DVC stages and Python orchestration may be adjusted during implementation, but the following must remain true:

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
- `DAGSHUB_REPOSITORY`
- `MLFLOW_TRACKING_URI`
- `MLFLOW_TRACKING_USERNAME`
- `MLFLOW_TRACKING_PASSWORD`

Optional LLM configuration may include provider, model, base URL, API key, timeout, temperature, and output-token limits. The project must support disabling LLM use and falling back to the deterministic plan. Missing LLM credentials must not prevent the DataOps and MLOps lifecycle from running.

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

Security requirements:

- External integration secrets must not be exposed to untrusted pull requests.
- CI must not register or promote models.

### 14.2 `data-pipeline.yml`

Triggers:

- Changes under `data/incoming/**`.
- Manual execution through `workflow_dispatch`.

Source or configuration changes may trigger CI or the quality portion of the data workflow, but automatic data processing requires exactly one identifiable incoming batch. The workflow must never guess which historical batch to process.

Responsibilities:

- Restore the Python environment.
- Pull required DVC artifacts.
- Validate incoming data.
- Run curation and drift analysis.
- Execute the Python workflow orchestrator.
- Generate or select an experiment plan.
- Train and evaluate candidates when required.
- Log experiments.
- Register the selected candidate.
- Compare the selected candidate with the current champion.
- Publish reports and pipeline artifacts.

The data pipeline must not move the `champion` alias automatically. Registration and promotion remain separate operations.

### 14.3 `promote-model.yml`

Trigger:

- Manual `workflow_dispatch` with an explicit registered model version.

Responsibilities:

- Resolve the requested registered model version.
- Re-run deterministic promotion gates.
- Verify the registered artifact loads.
- Require the configured API compatibility gate.
- Move the `champion` alias only when every gate passes.
- Publish the structured promotion report.

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
- Model artifact compatibility with the FastAPI serving layer.

### 15.4 End-to-End Scenarios

The project must demonstrate at least these scenarios:

1. Invalid data is rejected before training.
2. Normal valid data produces reports and skips retraining.
3. Drifted valid data triggers candidate training.
4. A weaker candidate is registered but not promoted.
5. A stronger candidate is promoted.
6. The promoted alias can be loaded and served by FastAPI.
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
- Final deterministic or LLM-assisted analysis.
- Registered model version.

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
| Registry failure | Stop before comparison or promotion |
| Candidate below threshold | Register for traceability, do not promote |
| API integration failure | Do not promote |

## 18. Security Requirements

- No credentials may be committed.
- `.env.example` may contain variable names and safe non-secret defaults, but never credentials.
- CI secrets must use GitHub encrypted secrets.
- Pull-request workflows must not receive DagsHub, MLflow, or LLM credentials.
- LLM input must contain summaries rather than raw data.
- LLM output must be treated as untrusted input.
- The plan validator must reject unknown fields when practical.
- FastAPI responses and logs must not expose tracking credentials or sensitive model URIs.

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
- OpenAI-compatible provider integration.
- Structured experiment-plan schema.
- Plan validator.
- Fallback behavior.
- Planner tests.

Exit criteria:

- Valid plans execute.
- Invalid plans are rejected.
- Provider failure triggers fallback.

### Phase 7 — Workflow Orchestration

Deliverables:

- Python application service for local orchestration.
- Conditional routing.
- Optional LLM planning step.
- Deterministic final analysis.
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

### Phase 10 — GitHub Actions Lifecycle

Deliverables:

- CI workflow without external credentials.
- Incoming-data workflow with DVC and MLflow credentials.
- Uploaded quality, drift, training, planning, and promotion artifacts.
- Separate manual promotion workflow.
- Explicit GitHub environment protection for remote operations.

Exit criteria:

- A new incoming batch can run through the remote DataOps and MLOps lifecycle.
- Promotion remains an explicit, deterministic, separately auditable operation.

### Phase 11 — Documentation and Demonstration

Deliverables:

- README with architecture and commands.
- Example reports.
- Demonstration scenarios.
- Local execution guide.
- GitHub Actions explanation.
- DagsHub tracking and registry explanation.

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
17. Promotion uses the immutable fixed-test dataset without allowing it to influence candidate selection.
18. The FastAPI service loads the complete promoted pipeline.
19. API contract and integration tests pass.
20. The complete lifecycle can run without paid compute infrastructure.
21. Documentation explains DataOps, MLOps, and optional LLMOps responsibilities clearly.

## 22. Future Extensions

The following may be evaluated after version one:

- Optuna for deterministic hyperparameter optimization.
- AutoKeras as an optional model-catalog entry.
- TensorFlow models.
- Additional LLM providers.
- Local LLM inference.
- LangChain if provider integrations or structured-output handling become materially more complex.
- LangGraph if workflow branching, persistence, retries, or recovery justify a graph framework.
- Scheduled drift checks.
- Delayed-label simulation.
- Prediction monitoring.
- Data-quality trend dashboards.
- Model calibration monitoring.
- Explainability reports.
- Free demonstration hosting for the API.
- Container image publication and a container registry.
- Trivy container scanning.
- Software bill of materials generation.
- Kubernetes deployment as a separate learning project.

These extensions must not be added until the first complete automation cycle is working.

## 23. Confirmed Decisions

Version one uses these decisions:

1. Repository name: `customer-churn-agentic-mlops`.
2. Python version: `3.12`.
3. DVC remote: DagsHub-compatible S3 remote configured with local or GitHub environment credentials.
4. Optional LLM provider: Groq free tier by default, behind an OpenAI-compatible provider interface.
5. Dataset ranges, categories, seeds, and drift thresholds are versioned in `params.yaml`.
6. Primary candidate-selection and promotion metric: ROC-AUC.
7. Absolute and relative promotion thresholds are versioned in `params.yaml`.
8. Every candidate is tracked in MLflow, but only the deterministically selected candidate is registered.
9. Registration and promotion are separate operations.
10. Updating the `champion` alias requires an explicit local command or manual GitHub Actions workflow.
11. Docker image publication and application deployment are outside version-one scope.
12. LangChain and LangGraph are optional future tools, not required dependencies.

## 24. Final Project Definition

This project combines:

- DataOps for validating, curating, profiling, and versioning data.
- MLOps for reproducible training, experiment tracking, model registration, comparison, promotion, and serving validation.
- Constrained agentic experimentation for bounded LLM-generated experiment plans.
- Basic LLMOps for structured prompts, validated outputs, provider abstraction, fallback behavior, and result traceability.

The model and agent are intentionally simple. The primary learning objective is the automated and auditable lifecycle from new data through validation, versioning, training, registration, deterministic promotion, and local inference.
