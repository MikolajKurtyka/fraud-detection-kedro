URL ?= http://127.0.0.1:8000
REQUESTS ?= 1 10 1000 10000
BATCHES ?= 1 10 100 1000
DURATION ?= 60
CLIENTS ?= 1
WORKERS ?= 4
MODEL ?= lightgbm

.PHONY: help setup data ingest validate features train train-fast baselines evaluate production compare feature-study serving-latency run viz mlflow api api-prod loadtest test lint format docker-up docker-down clean

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

setup:  ## Install environment, git hooks, check Kaggle access
	bash scripts/setup.sh

data:  ## Download competition data into data/01_raw
	uv run python scripts/download_data.py

ingest:  ## Run the data_ingestion pipeline
	uv run kedro run --pipelines data_ingestion

validate:  ## pandera checks on the ingested tables -> data/08_reporting/data_validation.json
	uv run kedro run --pipelines data_validation

features:  ## Run cleaning + feature engineering (02 -> 05)
	uv run kedro run --pipelines data_cleaning,feature_engineering

train:  ## Research: tune + train all models on all features, evaluate, write their submissions
	uv run kedro run --pipelines training,evaluation

train-fast:  ## Same, but train once with base params (no tuning)
	uv run kedro run --pipelines training,evaluation --params "lightgbm.model_options.n_trials=0,xgboost.model_options.n_trials=0,catboost.model_options.n_trials=0"

baselines:  ## Train + evaluate only the baselines (logreg, random_forest, hist_gb)
	uv run kedro run --pipelines training,evaluation --namespaces logreg,random_forest,hist_gb

evaluate:  ## Metrics, curves and Kaggle submissions for the saved models
	uv run kedro run --pipelines evaluation

production:  ## Production model (MODEL, top-N features, holdout) -> submission_production.csv + MLflow @champion
	uv run kedro run --pipelines production_$(MODEL)

compare:  ## History of all trained models from MLflow as a table
	uv run python scripts/compare_runs.py

feature-study:  ## Accuracy, model size and serving latency vs number of features (top-N)
	uv run python scripts/feature_size_study.py

serving-latency:  ## Single-row latency of every saved model: pandas vs NumPy path, threads vs 1 thread
	uv run python scripts/serving_latency.py

run:  ## Run all pipelines
	uv run kedro run

viz:  ## Open the pipeline graph (kedro-viz)
	uv run kedro viz run

mlflow:  ## Local MLflow UI (sqlite:///mlflow.db)
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

api:  ## Scoring API with auto-reload (one worker, for development)
	uv run uvicorn fraud_detection.api.main:app --reload

api-prod:  ## Scoring API with WORKERS processes, as in Docker (MICROBATCH_WAIT_MS=2 turns on micro-batching)
	rm -rf .prometheus && mkdir -p .prometheus
	PROMETHEUS_MULTIPROC_DIR=.prometheus OMP_NUM_THREADS=1 uv run uvicorn fraud_detection.api.main:app \
		--host 127.0.0.1 --port 8000 --workers $(WORKERS) --no-access-log

loadtest:  ## Open-loop load test of a running API (URL, REQUESTS per minute, BATCHES sizes, DURATION s, CLIENTS processes)
	uv run python scripts/loadtest.py --url $(URL) --requests $(REQUESTS) --batch-sizes $(BATCHES) \
		--duration $(DURATION) --clients $(CLIENTS)

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
