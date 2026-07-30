# Agent And LLMOps Guide

This directory owns the constrained LLM behavior for the project.

The agent does not train models, validate data, promote models, execute code, or
run shell commands. Its role is limited to proposing bounded experiment plans and
generating human-readable analysis from verified artifacts.

The project rule is:

```text
The LLM proposes and explains.
Deterministic code validates, trains, selects, and promotes.
```

## Current Scope

The current implementation covers the first planning boundary:

- `schemas.py`: strict Pydantic models for experiment plans.
- `plan_validator.py`: deterministic validation against metric and model-catalog
  policy.
- `planner.py`: fallback planning plus optional provider-based planning.
- `llm.py`: replaceable provider protocol.
- `analyst.py`: deterministic audit/report generation from verified artifacts.
- `prompts/experiment-planner.prompt.md`: bounded prompt contract.

The network provider uses an OpenAI-compatible chat-completions interface. Groq is
the current free default, but the same configuration shape can point to OpenAI or
another compatible provider later.

## Why The Provider Boundary Exists

The code does not import a provider-specific SDK directly inside training.
Instead, `LlmProvider` defines one method:

```python
generate_experiment_plan(prompt: str) -> str
```

That method returns raw text. The raw text is intentionally untrusted. It must pass
two gates before training can use it:

1. Pydantic schema validation through `ExperimentPlan`.
2. Deterministic project policy validation through `validate_experiment_plan`.

If either gate fails, the fallback plan is used.

## Fallback Behavior

The fallback plan is deterministic and versioned. It includes:

- logistic regression as an interpretable baseline
- random forest as a bounded nonlinear baseline

The fallback is used when:

- `LLM_ENABLED=false`
- the provider is unavailable
- the provider returns invalid JSON
- the provider returns a schema-invalid plan
- the provider proposes a model, metric, or parameter outside project policy

This is important because the data/training pipeline should not fail just because
an optional LLM is unavailable.

## Running The Planner

Create the current approved experiment plan:

```bash
make plan-experiments
```

By default, this writes:

```text
artifacts/experiment-plans/fallback.json
artifacts/agent/planner-trace.json
```

With the current default `.env` setting:

```text
LLM_ENABLED=false
```

the output plan is deterministic fallback.

Generate a standalone human-readable analysis report after training metrics exist:

```bash
make agent-analysis
```

By default, this writes:

```text
artifacts/agent/agent-analysis.md
```

The Markdown report is for human inspection. It summarizes the plan source,
fallback decision, proposed experiments, validation status, selected model,
candidate metrics, and promotion result when one is explicitly supplied:

```bash
make agent-analysis PROMOTION=artifacts/metrics/promotion.json
```

The optional argument is deliberate. A standalone report must not silently read a
stale promotion file from an older model version.

## How This Connects To Training

Training reads an approved plan from:

```text
artifacts/experiment-plans/fallback.json
```

Then `src.training.train` builds only the candidates listed in that plan. The LLM
never creates estimators directly. It can only propose structured values that the
catalog already allows.

The training step validates the plan again before fitting models. This repeated
validation is intentional because artifacts are treated as untrusted inputs when
they cross subsystem boundaries.

## DVC Integration

The DVC stage is named `fallback_plan` for compatibility with the current
deterministic pipeline. It depends on:

- the planner code
- the plan schema
- the deterministic validator
- the model catalog
- the experiment-planner prompt
- relevant `params.yaml` sections

That means changing the prompt, schema, catalog, or experiment policy causes DVC
to know the plan artifact may need to be regenerated.

The final analysis is not a DVC stage. Registration and comparison are remote,
run-specific operations, so placing the final report in the static DVC graph
would either omit the current promotion result or leave a DVC output modified
after reproduction.

The operational `make pipeline` path has stronger ordering:

```text
train -> register exact selected version -> deterministic comparison
      -> agent analysis using that comparison artifact
```

This guarantees that the final report describes the current registered candidate,
not an earlier local report. The analysis remains read-only and cannot influence
selection, registration, or promotion.

## Secret Handling

Local LLM settings live in `.env`:

```text
LLM_ENABLED=false
LLM_PROVIDER=groq
LLM_MODEL=openai/gpt-oss-20b
LLM_API_KEY=
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_TIMEOUT_SECONDS=30
LLM_TEMPERATURE=0
LLM_MAX_TOKENS=1200
```

`LLM_API_KEY` is sensitive and must not be committed. `.env.example` only stores
variable names and safe defaults.
In GitHub Actions, configure the same `LLM_*` names as encrypted environment
secrets instead of committing a `.env` file.

The prompt and planner must never include raw customer rows, credentials, shell
commands, package installation instructions, or arbitrary file paths.

## Provider Switching

For the current free Groq setup:

```text
LLM_PROVIDER=groq
LLM_MODEL=openai/gpt-oss-20b
LLM_BASE_URL=https://api.groq.com/openai/v1
```

For a future OpenAI setup, the code should not need to change if the same
OpenAI-compatible endpoint is used:

```text
LLM_PROVIDER=openai
LLM_MODEL=gpt-4.1-mini
LLM_BASE_URL=https://api.openai.com/v1
```

Only the environment variables change. The planner still treats provider output
as untrusted and applies the same validation gates.
