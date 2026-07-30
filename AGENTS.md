# AGENTS.md

## Purpose

This file tells coding agents how to work in the `customer-churn-agentic-mlops` repository.

Use [`SPEC.md`](./SPEC.md) as the authoritative source for project scope, functional requirements, architecture, failure behavior, implementation phases, and acceptance criteria.

Use this file for implementation conduct, engineering conventions, architectural boundaries, and task completion rules.

Do not duplicate the full specification here. When a task requires more product detail, read the relevant section of `SPEC.md` before changing code.

## Instruction Precedence

Apply instructions in this order:

1. The user's explicit instruction for the current task.
2. `SPEC.md` for required behavior and project scope.
3. `AGENTS.md` for implementation practices and agent conduct.
4. Existing tests for behavior already implemented and verified.
5. Existing source code and configuration.
6. `README.md` and supporting documentation.

When sources conflict:

- `SPEC.md` governs product behavior.
- `AGENTS.md` governs implementation conventions.
- A direct user instruction governs the current task.
- Do not silently change requirements to match existing code.
- Do not modify `SPEC.md` unless the user asks for a specification change.
- Report material conflicts and prefer the smallest reversible implementation when work can safely continue.

## Required Workflow Before Editing

Before making changes:

1. Read this file.
2. Read the relevant `SPEC.md` sections.
3. Inspect the current repository tree.
4. Inspect `pyproject.toml`, `uv.lock`, and `params.yaml` when relevant.
5. Inspect existing tests for the affected behavior.
6. Inspect GitHub Actions workflows when changing CI, training, registry, promotion, or remote pipeline behavior.

Do not implement from a task title alone when the behavior is already defined in `SPEC.md`.

## Project Overview

The repository implements a one-repository learning project that combines:

- DataOps: synthetic data generation, validation, curation, profiling, drift analysis, and data versioning.
- MLOps: reproducible training, evaluation, tracking, registration, champion comparison, promotion, and serving validation.
- Constrained agentic experimentation: an LLM proposes bounded experiment plans.
- Basic LLMOps: structured prompts, validated outputs, provider abstraction, fallback behavior, and traceability.

The business scenario is binary customer-churn prediction using synthetic labeled data.

The main objective is the automated and auditable lifecycle from a new data batch to a validated model served locally through FastAPI. Model and agent sophistication are secondary.

## Defining Principle

> The LLM proposes and explains. Deterministic code validates, trains, evaluates, and promotes.

Every implementation decision must preserve this separation.

## Non-Negotiable Invariants

Never weaken these rules without an explicit specification change:

1. Validate incoming data before curation or training.
2. Invalid data never enters accepted or curated datasets.
3. The immutable test dataset never enters training, preprocessing fitting, parameter selection, or model selection.
4. Fit preprocessing only on training data.
5. Persist and serve the complete preprocessing and prediction pipeline.
6. Evaluate feature drift and target drift separately.
7. Do not send raw customer rows to the LLM in version one.
8. Treat all LLM output as untrusted input.
9. Validate LLM plans against allowlisted models, parameters, types, ranges, and resource limits.
10. Use a deterministic fallback plan when LLM inference fails or returns an invalid plan.
11. The LLM never selects the winner, changes thresholds, or promotes a model.
12. Candidate selection and promotion are deterministic.
13. Failed quality, artifact-loading, fixed-test, or API compatibility checks block promotion.
14. A rejected candidate cannot move the `champion` alias.
15. Every promoted model is traceable to code, data, configuration, metrics, and registry metadata.
16. Never commit or log credentials, raw datasets, or sensitive environment values.
17. Pull-request workflows never receive registry, DagsHub, MLflow, or LLM secrets.
18. Version one must not require paid compute infrastructure.

## Scope Control

Implement only the current task and the relevant phase of `SPEC.md`.

Do not introduce these technologies or capabilities in version one unless the user explicitly changes the specification:

- Kubernetes.
- Airflow, Prefect, or Dagster.
- TensorFlow or AutoKeras.
- GPU training.
- Deep Agents.
- Autonomous code generation.
- LLM-generated shell execution.
- Arbitrary model architectures.
- Feature stores.
- Streaming ingestion.
- Online learning.
- Real-time model monitoring.
- Cloud databases.
- Permanent cloud API hosting.
- Multi-environment deployment frameworks.

Prefer the smallest coherent vertical slice. Do not add abstractions, dependencies, or services only because they may be useful later.

## Repository Boundaries

### `src/data/`

Owns deterministic data operations:

- Synthetic generation.
- Pandera schemas.
- Validation.
- Profiling.
- Evidently drift analysis.
- Curation.

It must not call the LLM, serve HTTP requests, train models, or promote models.

### `src/agent/`

Owns constrained LLM behavior:

- Provider abstraction.
- Pydantic plan schemas.
- Experiment planning.
- Deterministic plan validation.
- Human-readable final analysis.

It must not train models, execute generated code, or update registry aliases.

### `src/training/`

Owns deterministic ML behavior:

- Model catalog.
- Preprocessing.
- Candidate training.
- Metrics.
- Candidate selection.
- Champion comparison.
- MLflow logging.
- Model registration.

Core training logic must not depend on an LLM provider or workflow framework.

### `src/workflow/`

Owns orchestration:

- Explicit Python application services.
- Conditional routing.
- Local workflow entry point.

State should contain paths, identifiers, and structured summaries rather than complete datasets.

Only experiment planning and final analysis may require LLM inference.

### `src/api/`

Owns model serving:

- Application startup.
- Model loading.
- Pydantic contracts.
- `/health`.
- `/model-info`.
- `/predict`.

It must not train models, invoke the LLM, run DVC, or perform drift analysis.

## Standard Task Procedure

For every coding task:

1. Locate the relevant requirement and exit criteria in `SPEC.md`.
2. Inspect the current code and tests before editing.
3. Choose the smallest change that satisfies the requirement.
4. Preserve architectural boundaries and invariants.
5. Add or update tests with the implementation.
6. Run focused tests first.
7. Run the complete relevant quality checks before finishing when practical.
8. Review the diff for unrelated changes, secrets, generated artifacts, and formatting churn.
9. Update documentation when commands, configuration, contracts, or behavior changed.
10. Report changes, checks run, assumptions, and unresolved specification decisions.

Do not implement future phases unless a minimal interface is required for the current phase.

## Handling Missing Detail

When implementation detail is unclear:

1. Search `SPEC.md`, including functional requirements, failure behavior, implementation phases, acceptance criteria, and open decisions.
2. Inspect `params.yaml`, tests, and established interfaces.
3. Prefer a conservative, deterministic, reversible default.
4. Keep values configurable when they represent policy.
5. Do not invent behavior that expands scope.
6. Do not silently resolve an explicitly open decision that materially affects contracts, security, promotion, or external integrations.
7. Record temporary assumptions in the final task summary.

## Python Conventions

Use the Python version declared by the repository. Do not change it independently.

Code should:

- Use type annotations for public functions, methods, and structured data.
- Prefer `pathlib.Path` for filesystem paths.
- Prefer small functions with explicit inputs and outputs.
- Prefer pure functions for validation, metrics, profiling, and policies.
- Use Pydantic at API, configuration, artifact, and LLM boundaries where appropriate.
- Use `Enum` or `Literal` for constrained identifiers where useful.
- Avoid broad `Any` types unless an integration boundary requires them.
- Avoid hidden mutable global state.
- Avoid import-time network calls, downloads, or workflow execution.
- Separate configuration loading from business logic.
- Raise actionable domain-specific exceptions.
- Preserve exception context when wrapping failures.
- Use structured logging.
- Never log secrets or full raw datasets.

Do not broadly suppress Ruff findings when a narrow correction is possible.

## Dependencies

Use `uv` and the committed lock file.

- Add dependencies only when the standard library or existing dependencies cannot reasonably solve the task.
- Separate serving dependencies from training and development dependencies where supported.
- Never edit `uv.lock` manually.
- Update `pyproject.toml` and `uv.lock` together through the established `uv` command.
- Do not upgrade unrelated packages during a focused task.

## Configuration and Secrets

Versioned, non-secret behavior belongs in `params.yaml` or another committed configuration file.

Examples:

- Random seeds.
- Data-generation settings.
- Validation and drift thresholds.
- Model limits.
- Primary metric.
- Promotion thresholds.
- Registered model name.
- API metadata defaults.

Secrets belong only in environment variables or GitHub encrypted secrets.

Never place secrets in:

- Source code.
- `params.yaml`.
- `.env.example`.
- Notebooks.
- Prompts.
- Tests or fixtures.
- DVC metadata.
- Workflow defaults.
- Logs or reports.

Do not duplicate configurable policy values across modules, and never allow an LLM response to override them.

## DataOps Rules

### Synthetic data

- Generation must be deterministic for the same seed and parameters.
- Reference, fixed-test, normal, drifted, and invalid scenarios must remain reproducible.
- Invalid scenarios must intentionally violate documented rules.

### Validation

- Validate before curation.
- Produce machine-readable details and a human-readable summary.
- Keep rejected data outside accepted and curated outputs.
- Test successful and failing paths.

### Curation

- Curate only accepted labeled batches.
- Keep merge and deduplication behavior deterministic.
- Preserve the documented feature schema and target.
- Never merge the fixed test dataset into training data.

### Drift

- Separate feature drift from target drift.
- Do not count the target as an ordinary feature.
- Drift may trigger candidate training, but it does not justify promotion.
- Persist structured drift results as well as visual reports.

### DVC

- Declare stage dependencies, parameters, and outputs explicitly.
- Avoid hidden inputs.
- Keep stages runnable locally and in CI.
- Avoid unnecessary rewrites that defeat caching.
- Do not commit large DVC-managed artifacts directly to Git.

## MLOps Rules

### Preprocessing and training

- Split before fitting transformations.
- Fit transformations only on training data.
- Use a scikit-learn `Pipeline` or equivalent complete artifact.
- Train only allowlisted candidates.
- Enforce experiment and parameter bounds deterministically.
- Use fixed random seeds where supported.
- Do not use the fixed test dataset for model or parameter selection.

### Evaluation

Support the metrics required by `SPEC.md`, including:

- ROC-AUC.
- PR-AUC.
- F1.
- Precision.
- Recall.
- Confusion matrix.

Accuracy may be logged but cannot be the sole decision metric.

Candidate selection and tie-breaking must be explicit and deterministic.

### Tracking and registry

Candidate runs must retain traceability to the required code, data, configuration, plan, metric, and artifact metadata defined in `SPEC.md`.

- Do not promote when required MLflow logging fails.
- Registering and promoting are separate operations.
- Change the `champion` alias only after every deterministic gate passes.

### Promotion

- Read policy thresholds from versioned configuration.
- Produce structured pass/fail reasons for every gate.
- Handle the no-champion case explicitly.
- Treat incomplete or failed comparisons as no promotion.
- Include artifact loading and API integration checks in the gate.

## LLM and Workflow Rules

The LLM integration must be optional and replaceable.

- Hide provider-specific behavior behind a small interface.
- Use Pydantic structured output.
- Bound timeouts and retries.
- Send only the minimum structured context.
- Exclude secrets and raw customer records.
- Version prompts under `prompts/`.
- Persist the proposed plan and approved plan when they differ.
- Record when fallback was used and why.
- Verify model names, parameters, counts, and metrics independently.
- Build final analysis only from verified pipeline outputs.

The LLM must not generate content for automatic execution, including Python code, shell commands, package installations, arbitrary paths, or registry actions.

Workflow operations must be small, typed, and testable. Routing logic belongs in deterministic Python conditions, not prompts. LangChain and LangGraph are optional future tools and must not be introduced unless they remove demonstrated complexity.

Required paths must remain possible and testable:

- Invalid data stops before curation.
- No significant drift can stop without training.
- LLM failure reaches fallback training.
- Rejected candidates cannot become champion.
- Final analysis cannot modify the promotion result.

## FastAPI Rules

Follow the endpoint contracts in `SPEC.md`.

- Use Pydantic request and response models.
- Load the complete model pipeline during startup or application lifespan.
- Fail clearly when the serving artifact cannot be loaded.
- Return non-sensitive model metadata.
- Validate all required input features.
- Return class, probability when supported, and served model version.
- Keep business logic outside endpoint functions.
- Never call the LLM during prediction.

## Optional Local Docker

The existing Dockerfile is a local-development convenience, not a version-one delivery artifact.

- Never bake credentials into image layers.
- Keep raw data, reports, tests, local environments, and secrets out of the build context.
- Do not add image publication or deployment workflows unless `SPEC.md` changes explicitly.

## GitHub Actions Rules

Use least privilege and explicit permissions.

### CI

- Run linting, formatting verification, tests, and contract checks.
- Do not register or promote models.
- Do not expose external integration secrets to pull requests.

### Data pipeline

- Run the validated data and training lifecycle.
- Support manual execution.
- Publish reports and traceability artifacts.
- Use fallback when the LLM is unavailable or invalid.

### Promotion

- Keep promotion in a separate manually triggered workflow.
- Resolve the exact registered model version.
- Re-run fixed-test, artifact-loading, policy, and API compatibility gates.
- Move `champion` only after every gate passes.

General workflow rules:

- Pin action versions according to repository policy.
- Avoid printing secrets or full datasets.
- Use concurrency controls where duplicate remote operations are possible.
- Prefer Makefile targets or project scripts over duplicated shell logic.

## Testing Requirements

Every behavior change needs relevant tests.

### Unit tests

Use unit tests for deterministic logic such as:

- Synthetic generation.
- Pandera rules.
- Profiles and drift parsing.
- Model-catalog constraints.
- Plan validation and fallback.
- Metrics and tie-breaking.
- Promotion policy.
- API schemas and metadata formatting.

Unit tests must not require network access. Mock or fake LLM providers, remote MLflow/DagsHub, and GitHub APIs.

### Integration tests

Cover important boundaries such as:

- Valid and invalid data paths.
- Curation.
- Drift-triggered training.
- LLM failure followed by fallback.
- Candidate training and artifact loading.
- MLflow behavior with a local or test backend where practical.
- Champion comparison.
- FastAPI prediction using a real generated model artifact.

### Contract tests

Verify:

- Input feature names and types.
- Prediction response structure.
- `/health`.
- `/model-info`.
- Compatibility between the trained artifact and the serving application.

### Test quality

- Use fixed seeds.
- Avoid timing-dependent and order-dependent assertions.
- Keep fixtures small.
- Assert failure reasons, not only boolean outcomes.
- Test safeguards and negative paths around validation, fallback, registration, and promotion.

## Commands and Tooling

Prefer commands exposed through the `Makefile`.

Expected command concepts include:

- `make install`
- `make lint`
- `make test`
- `dvc repro`

Inspect the current `Makefile`, `pyproject.toml`, and workflows before running or documenting a command. Do not assume a target exists merely because it appears in the specification.

When no target exists, use the repository's established `uv run ...` command. Add a Makefile target only when it improves repeatability for local development and CI.

Use Ubuntu-compatible shell commands. Do not add PowerShell-specific instructions.

## Documentation Responsibilities

- `SPEC.md`: requirements, scope, behavior, architecture, and acceptance criteria.
- `AGENTS.md`: coding-agent conduct and implementation conventions.
- `README.md`: human installation, execution, architecture overview, and demonstrations.
- `prompts/`: versioned LLM instructions.
- Code comments: non-obvious reasoning, constraints, and safety decisions.

Update documentation when a change affects installation, commands, configuration, architecture, data contracts, workflows, promotion policy, API contracts, secrets, or demonstration scenarios.

Do not use documentation as a substitute for tests or configuration.

## Change Discipline

Keep every change focused.

Do not:

- Reformat unrelated files.
- Rename unrelated modules.
- Upgrade unrelated dependencies.
- Change public contracts without updating tests and documentation.
- Remove safeguards to make tests pass.
- Weaken validation, promotion, or security behavior.
- Commit generated datasets, models, reports, caches, credentials, or local environments accidentally.

When refactoring, preserve behavior through tests and keep migrations explicit when formats or interfaces change.

## Definition of Done

A task is complete only when:

1. The relevant `SPEC.md` requirement is satisfied.
2. Architectural boundaries and invariants remain intact.
3. Relevant tests were added or updated.
4. Relevant tests pass.
5. Lint and formatting checks pass for changed code.
6. Configuration and documentation are updated when required.
7. No secret, raw dataset, or unintended generated artifact was introduced.
8. Failure behavior remains explicit and safe.
9. The diff contains no unrelated changes.
10. The final task summary states:
    - What changed.
    - Which files changed.
    - Which checks were run.
    - Any assumption or unresolved specification decision.

## Phase-Based Work

Follow the implementation phases in `SPEC.md`.

When assigned a phase:

- Complete its deliverables.
- Verify its exit criteria.
- Avoid implementing later phases unless a minimal interface is required.
- Leave working, testable software at the end of the phase.
- Keep future integrations behind small interfaces instead of adding premature infrastructure.

When a task spans phases, preserve the dependency order defined in the specification.

## Final Agent Checklist

Before finishing, confirm:

- I read the relevant `SPEC.md` sections.
- I preserved deterministic quality gates.
- I kept LLM responsibilities constrained.
- I prevented preprocessing and test-data leakage.
- I protected secrets and raw data.
- I added or updated the correct tests.
- I ran the relevant checks.
- I avoided unrelated scope expansion.
- I documented meaningful assumptions.

If any answer is no, the task is not complete.
