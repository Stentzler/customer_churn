.PHONY: install lint test generate-data process-batch curate-data profile-data drift-data plan-experiments agent-analysis data-pipeline train-models track-models compare-model promote-model run-api docker-build docker-run pipeline create-incoming-batch acceptance-local acceptance-remote

install:
	uv sync --locked

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest

generate-data:
	uv run python -m src.data.generate --scenario all

process-batch:
	uv run python -m src.data.ingest --input "$(INPUT)"

curate-data:
	uv run python -m src.data.curate

profile-data:
	uv run python -m src.data.profile

drift-data:
	uv run python -m src.data.drift --current "$(CURRENT)"

plan-experiments:
	uv run python -m src.agent.planner

agent-analysis:
	uv run python -m src.agent.analyst $(if $(PROMOTION),--promotion "$(PROMOTION)",)

data-pipeline:
	uv run dvc repro

train-models:
	uv run python -m src.training.train

track-models:
	uv run python -m src.training.registry

compare-model:
	uv run python -m src.training.compare $(if $(VERSION),--candidate-version "$(VERSION)",)

promote-model:
	uv run python -m src.training.compare --promote $(if $(VERSION),--candidate-version "$(VERSION)",)

run-api:
	uv run uvicorn src.api.main:create_app --factory --host 0.0.0.0 --port 8000

docker-build:
	docker build -t customer-churn-api:local .

docker-run:
	docker run --rm --env-file .env -p 8000:8000 customer-churn-api:local

pipeline:
	uv run python -m src.workflow.local_pipeline --input "$(INPUT)" $(if $(filter 1 true yes,$(FORCE_RETRAIN)),--force-retrain,)

create-incoming-batch:
	uv run python -m scripts.local_acceptance --create-only $(ARGS)

acceptance-local:
	uv run python -m scripts.local_acceptance

acceptance-remote:
	uv run python -m scripts.local_acceptance --remote --push-dvc
