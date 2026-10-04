UV ?= uv
export PYTHONPATH := src

.PHONY: setup up down dev migrate revision test test-integration lint format typecheck seed check

setup:
	$(UV) sync --all-groups

up:
	docker compose up -d db

down:
	docker compose down

migrate:
	$(UV) run alembic upgrade head

revision:
	$(UV) run alembic revision --autogenerate -m "$(m)"

dev:
	$(UV) run uvicorn evidencedesk.main:app --reload --port 8000

test:
	$(UV) run pytest

test-integration:
	TEST_DATABASE_URL=$${TEST_DATABASE_URL:-postgresql+psycopg://evidencedesk:evidencedesk@localhost:5432/evidencedesk} $(UV) run pytest -m integration

lint:
	$(UV) run ruff check .

format:
	$(UV) run ruff format .

typecheck:
	$(UV) run mypy src

seed:
	$(UV) run python scripts/seed.py

check: lint typecheck test
