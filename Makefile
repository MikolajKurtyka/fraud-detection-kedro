.PHONY: help setup data ingest run viz mlflow api test lint format docker-up docker-down clean

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

setup:  ## Install environment, git hooks, check Kaggle access
	bash scripts/setup.sh

data:  ## Download competition data into data/01_raw
	uv run python scripts/download_data.py

ingest:  ## Run the data_ingestion pipeline
	uv run kedro run --pipelines data_ingestion

run:  ## Run all pipelines
	uv run kedro run

viz:  ## Open the pipeline graph (kedro-viz)
	uv run kedro viz run

mlflow:  ## Local MLflow UI (sqlite:///mlflow.db)
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

api:  ## Scoring API with auto-reload
	uv run uvicorn fraud_detection.api.main:app --reload

test:  ## Run tests with coverage
	uv run pytest --cov=fraud_detection --cov-report=term-missing

lint:  ## Lint and check formatting
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts

format:  ## Auto-fix lint and formatting
	uv run ruff check --fix src tests scripts
	uv run ruff format src tests scripts

docker-up:  ## Start MLflow + API (docker compose)
	docker compose up -d --build

docker-down:  ## Stop the stack
	docker compose down

clean:  ## Remove generated data (keeps 01_raw)
	find data -mindepth 2 -not -path 'data/01_raw/*' -not -name .gitkeep -type f -delete
