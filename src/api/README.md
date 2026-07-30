# FastAPI Serving Guide

This directory contains the local serving layer for the promoted customer-churn
model.

The API has one responsibility: load the complete trained MLflow model pipeline
and expose predictions through HTTP. It does not train models, run DVC, detect
drift, call the LLM, or promote model versions.

## Environment

The API reads serving configuration from environment variables. Locally, these
variables normally come from `.env`. In production or CI, the same variables can
come from the operating-system environment or GitHub Secrets.

```text
MLFLOW_TRACKING_URI=<dagshub-mlflow-uri>
MLFLOW_TRACKING_USERNAME=<dagshub-username>
MLFLOW_TRACKING_PASSWORD=<dagshub-token>
MODEL_NAME=customer-churn
MODEL_ALIAS=champion
```

`MODEL_ALIAS` selects which registered model alias is served. For the normal
MLOps flow, keep it as `champion`. If you want to test another alias later, change
only this value without changing application code.

The resolved model URI is:

```text
models:/<MODEL_NAME>@<MODEL_ALIAS>
```

For example:

```text
models:/customer-churn@champion
```

`MODEL_URI` is also supported as an advanced override. When it is set, the API
uses that exact URI instead of building one from `MODEL_NAME` and `MODEL_ALIAS`.

## Secret Handling

Sensitive values must come only from local environment variables, `.env`, GitHub
Secrets, or a real deployment secret manager. They must not be committed to Git
or written into versioned configuration files.

For local serving, the sensitive values are:

```text
MLFLOW_TRACKING_URI
MLFLOW_TRACKING_USERNAME
MLFLOW_TRACKING_PASSWORD
```

The API does not return these values from `/model-info`, and startup errors avoid
printing the full model URI. This matters because `MODEL_URI` is configurable and
could accidentally contain sensitive connection details in a real project.

`.env.example` is safe to commit because it contains variable names and defaults,
not real credentials. The real `.env` file must stay local and ignored by Git.

## Run Locally

Start the API from the repository root:

```bash
make run-api
```

The service listens on:

```text
http://localhost:8000
```

During startup, FastAPI's lifespan hook creates the `ChampionModelService`, loads
the MLflow model once, and stores the service in `app.state`. The service uses a
Lazy Singleton metaclass, so repeated `ChampionModelService(...)` construction in
the same process returns the same object. After startup, all requests reuse the
same in-memory pipeline through dependency injection.

## Endpoints

Health:

```bash
curl http://localhost:8000/health
```

Model metadata:

```bash
curl http://localhost:8000/model-info
```

Prediction:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "tenure_months": 12,
    "monthly_spend": 120.5,
    "support_tickets_90d": 3,
    "late_payments_12m": 1,
    "usage_hours_monthly": 45.0,
    "plan_type": "basic",
    "region": "north"
  }'
```

The request includes only model features. It intentionally excludes
`customer_id` and `churned`, because `customer_id` is not a predictive feature and
`churned` is the training target.

The response includes:

- `predicted_class`: `0` for retained or `1` for churn.
- `churn_probability`: probability for the churn class when the model supports
  probabilities.
- `model_name`, `model_version`, and `model_alias`: metadata showing what was
  served.

## Docker

Build the local image:

```bash
make docker-build
```

Run it locally on port `8000`:

```bash
make docker-run
```

The Docker command passes `.env` into the container at runtime. Credentials are
not copied into the image layers.

The image is intentionally local-only for now. Publishing to Docker Hub becomes
useful when we add the later publication workflow, but it is not needed to test
the serving layer on your machine.
