.PHONY: install lint test generate-data process-batch curate-data profile-data drift-data data-pipeline train-models track-models pipeline acceptance-local acceptance-remote

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

data-pipeline:
	uv run dvc repro

train-models:
	uv run python -m src.training.train

track-models:
	uv run python -m src.training.registry

pipeline:
	uv run python -m src.workflow.local_pipeline --input "$(INPUT)"

acceptance-local:
	uv run python scripts/local_acceptance.py

acceptance-remote:
	uv run python scripts/local_acceptance.py --remote --push-dvc
