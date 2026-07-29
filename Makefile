.PHONY: install lint test generate-data process-batch curate-data

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
