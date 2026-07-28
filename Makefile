.PHONY: install lint test

install:
	uv sync --locked

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest
