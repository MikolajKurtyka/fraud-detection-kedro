# Fraud Detection: Real-Time Transaction Scoring with Kedro, MLflow and FastAPI

An end-to-end MLOps project that detects fraudulent card transactions on the
[IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) dataset.
It shows how to go from raw data to a model behind an API: reproducible Kedro pipelines,
experiment tracking and a model registry in MLflow, a fraud threshold chosen by business cost,
and a containerised FastAPI scoring service.

> **What this project demonstrates**
> - **Pipeline engineering with Kedro**: data catalog, modular pipelines, parameters, and a lineage graph in `kedro-viz`
> - **No training/serving skew**: the Kedro inference pipeline (features + model) is logged as **one MLflow model**, and the API serves exactly that
> - **Cost-sensitive decisions**: the fraud threshold minimises the cost of missed fraud plus the cost of false alarms, instead of using 0.5
> - **Model lifecycle**: experiments, versions, and a champion/challenger alias in the MLflow registry
> - **Serving and infrastructure**: FastAPI + Docker + docker compose (MLflow with a Postgres backend), CI on GitHub Actions
> - **Reproducible data**: a one-command Kaggle download with a checksum manifest

## Project Status

| Area | Status |
|---|---|
| Repo structure, setup scripts, Kaggle download with manifest | Done |
| `data_ingestion` pipeline (join, schema fix, memory reduction) | Done |
| FastAPI service (`/health`, `/predict`), Dockerfile, docker compose | Done (waits for a registered model) |
| CI (lint, tests, Kedro load check, Docker build) | Done |
| `data_validation` (pandera schemas) | Planned |
| `feature_engineering` (time features, client UID, aggregations) | Planned |
| `training` (LightGBM + Optuna, time-based split) | Planned |
| `evaluation` (PR-AUC, cost-based threshold, SHAP, promotion) | Planned |

## Project Overview

Card fraud is a **rare-event and time-sensitive** problem: about 3.5% of transactions are fraudulent,
fraud patterns change over time, and a decision is needed within milliseconds of a card swipe.
This project builds the full path:

- **Data**: two tables (`transaction`, `identity`) joined on `TransactionID`, ~590k training rows and ~430 columns
- **Features**: time features from `TransactionDT`, a client identifier approximated from card and address fields, and per-client aggregations
- **Model**: LightGBM tuned with Optuna on a **time-based** validation split, optimising PR-AUC
- **Decision**: a threshold derived from a cost matrix (missed fraud costs the transaction amount, a false alarm costs a manual review)
- **Serving**: a REST endpoint that loads the `champion` model from the MLflow registry

## Architecture

```
┌──────────────────────────── Kedro project ─────────────────────────────┐
│                                                                        │
│  01_raw          02_intermediate     04_feature        06_models       │
│ ┌──────────┐    ┌──────────────┐   ┌─────────────┐   ┌─────────────┐   │
│ │ Kaggle   │    │ train / test │   │ engineered  │   │ LightGBM +  │   │
│ │ CSVs     │───►│ joined,      │──►│ features    │──►│ Optuna      │   │
│ │ (+ hash  │    │ typed,       │   │ (time, UID, │   │ (time-based │   │
│ │ manifest)│    │ parquet      │   │ aggregates) │   │ CV, PR-AUC) │   │
│ └──────────┘    └──────┬───────┘   └─────────────┘   └──────┬──────┘   │
│  data_ingestion        │ data_validation                    │          │
│                        ▼ (pandera)                          ▼          │
│                 ┌──────────────┐                     ┌─────────────┐   │
│                 │ validation   │      08_reporting   │ evaluation  │   │
│                 │ report       │    ◄────────────────│ cost-based  │   │
│                 └──────────────┘                     │ threshold,  │   │
│                                                      │ SHAP        │   │
└──────────────────────────────────────────────────────┴──────┬──────┴───┘
                                                              │ kedro-mlflow:
                                                              │ inference pipeline
                                                              ▼ = one MLflow model
┌───────────────── docker compose ─────────────────────────────────────────┐
│                                                                          │
│  ┌────────────┐      ┌───────────────────────┐      ┌─────────────────┐  │
│  │ Postgres   │◄─────│ MLflow server         │◄─────│ FastAPI         │  │
│  │ (metadata) │      │ tracking + registry   │ load │ /predict        │  │
│  └────────────┘      │ alias: @champion      │      │ /health         │  │
│                      └───────────────────────┘      └────────┬────────┘  │
└──────────────────────────────────────────────────────────────┼───────────┘
                                                               │
                                     client ── POST /predict ──┘ → fraud probability + decision

GitHub Actions: lint → tests → Kedro load check → Docker image build
```

## Key Components

**Kedro pipelines** (`src/fraud_detection/pipelines/`)

| Pipeline | Input → Output | What it does |
|---|---|---|
| `data_ingestion` | `01_raw` → `02_intermediate` | Fixes test column names (`id-01` → `id_01`), left-joins identity onto transactions, adds `has_identity`, downcasts numeric types |
| `data_validation` | `02_intermediate` → `08_reporting` | Checks schema, ranges and null rates with pandera; checks train/test column parity |
| `feature_engineering` | `02` → `04_feature` | Time features, client UID proxy, frequency and aggregation features |
| `training` | `04` → `06_models` | Time-based split, LightGBM + Optuna, MLflow logging |
| `evaluation` | `06` → `08_reporting` | PR-AUC, recall at a fixed false-positive rate, cost-based threshold, SHAP, champion promotion |

**MLflow** (through `kedro-mlflow`) records parameters, metrics and artifacts for every `kedro run`
and keeps models in the registry. The model that serves traffic carries the `champion` alias.

**Scoring API** (`src/fraud_detection/api/`) is a FastAPI app that loads `models:/fraud-detection@champion`
at startup and exposes `/predict` and `/health`. It still starts without a model and `/health` reports that,
so a deployment never fails silently.

## Project Structure

```
fraud-detection-kedro/
├── conf/
│   ├── base/                      # committed config
│   │   ├── catalog.yml            # every dataset, with dataset factories
│   │   ├── parameters*.yml        # per-pipeline parameters
│   │   └── mlflow.yml             # kedro-mlflow tracking config
│   └── local/                     # secrets and machine overrides (git-ignored)
├── data/                          # git-ignored, recreated by scripts/download_data.py
│   ├── 01_raw/                    # Kaggle CSVs + _manifest.json (checksums)
│   ├── 02_intermediate/           # joined, typed parquet
│   ├── 03_primary/ … 07_model_output/
│   └── 08_reporting/              # metrics, plots, validation reports
├── docker/mlflow.Dockerfile       # MLflow server image (Postgres driver)
├── docs/model_card.md
├── notebooks/                     # exploration only, no pipeline logic
├── scripts/
│   ├── download_data.py           # Kaggle download, idempotent, writes manifest
│   ├── setup.ps1                  # one-command setup (Windows)
│   └── setup.sh                   # one-command setup (Linux/macOS)
├── src/fraud_detection/
│   ├── pipelines/                 # one folder per Kedro pipeline
│   ├── api/                       # FastAPI scoring service
│   ├── pipeline_registry.py
│   └── settings.py
├── tests/
├── .github/workflows/ci.yml
├── Dockerfile                     # API image (multi-stage, uv)
├── docker-compose.yml             # Postgres + MLflow + API
├── Makefile
└── pyproject.toml                 # dependencies (uv), ruff, pytest, Kedro metadata
```

## Setup

### Prerequisites

- Python 3.11–3.13 (the project pins 3.12 in `.python-version`; uv downloads it if needed)
- [uv](https://docs.astral.sh/uv/) package manager (the setup script installs it)
- A Kaggle account and API token
- Docker Desktop (only for the serving stack)
- ~4 GB of free disk space

### 1. Kaggle access (one-time)

1. Accept the competition rules: open
   [ieee-fraud-detection/rules](https://www.kaggle.com/competitions/ieee-fraud-detection/rules)
   and click **"I Understand and Accept"**. **Without this the download fails with a 403 error.**
2. Create an API token: [kaggle.com/settings](https://www.kaggle.com/settings) → **API** → **Create New Token**.
3. Save the token where the Kaggle CLI looks for it:

   ```bash
   # Linux / macOS / Git Bash
   mkdir -p ~/.kaggle && echo <YOUR_TOKEN> > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
   ```

   ```powershell
   # Windows PowerShell
   New-Item -ItemType Directory -Force $HOME\.kaggle | Out-Null
   Set-Content $HOME\.kaggle\access_token "<YOUR_TOKEN>"
   ```

   Alternatives: set the `KAGGLE_API_TOKEN` environment variable (used in CI), or use a legacy `~/.kaggle/kaggle.json`.

### 2. One-command setup

```powershell
git clone <repo-url> fraud-detection-kedro
cd fraud-detection-kedro
.\scripts\setup.ps1 -Download        # Windows
./scripts/setup.sh --download        # Linux / macOS
```

The script installs uv if it's missing, creates the virtual environment (`uv sync`), installs
pre-commit hooks, creates `.env` from `.env.example`, checks Kaggle access, and downloads the data.

### 3. Data download in detail

`scripts/download_data.py` is the only supported way to get data into `data/01_raw/`:

```bash
uv run python scripts/download_data.py --check   # credentials + rules accepted?
uv run python scripts/download_data.py           # download (skips files already present)
uv run python scripts/download_data.py --force   # download everything again
```

- Downloads each competition file, unzips it if Kaggle compressed it, and checks the file size
- Writes `data/01_raw/_manifest.json` with the size and **SHA256** of every file, so any model can be traced to the exact data it was trained on
- Gives clear errors for missing credentials (401) and rules not accepted (403)

**Manual alternative:** download the zip from the
[competition data page](https://www.kaggle.com/competitions/ieee-fraud-detection/data) and unzip
the five CSVs directly into `data/01_raw/`:

```
data/01_raw/
├── train_transaction.csv   (683 MB)
├── train_identity.csv      (27 MB)
├── test_transaction.csv    (613 MB)
├── test_identity.csv       (26 MB)
└── sample_submission.csv
```

Data is never committed. `.gitignore` excludes everything under `data/` except the `.gitkeep` files that keep the folder structure.

## Usage

All commands have a `make` shortcut (`make help`); on Windows without `make`, run the `uv run ...` command directly.

### Pipelines

```bash
uv run kedro run --pipelines data_ingestion   # make ingest
uv run kedro run                             # make run (all pipelines)
uv run kedro registry list                   # list pipelines
uv run kedro viz run                         # make viz: interactive pipeline graph
```

### Experiment tracking

By default runs are logged to a local SQLite store (`mlflow.db`, artifacts in `./mlruns`):

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000   # make mlflow
```

To log to the docker compose server instead, set `MLFLOW_TRACKING_URI=http://localhost:5000` before `kedro run`.

### Serving stack (Docker)

```bash
docker compose up -d --build    # make docker-up
```

| Service | URL | Purpose |
|---|---|---|
| MLflow | http://localhost:5000 | Tracking UI and model registry (Postgres backend) |
| API | http://localhost:8000/docs | Swagger UI for `/predict` and `/health` |

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"TransactionID": 3663549, "features": {"TransactionDT": 18403224, "TransactionAmt": 31.95, "ProductCD": "W", "card1": 10409}}'
```

```json
{"TransactionID": 3663549, "fraud_probability": 0.031, "is_fraud": false, "threshold": 0.42, "model_uri": "models:/fraud-detection@champion"}
```

Until a model is registered and given the `champion` alias, `/health` returns `"model_loaded": false` and `/predict` returns `503`.

### Tests and code quality

```bash
uv run pytest            # make test
uv run ruff check .      # make lint
```

## CI/CD

`.github/workflows/ci.yml` runs on every push to `main` and on every pull request:

1. **lint-test**: `uv sync --frozen` → ruff lint and format check → pytest with coverage → `kedro registry list` (checks the project loads)
2. **docker**: builds the API image with layer caching (needs lint-test to pass)

Planned: a scheduled retraining workflow that downloads data with the `KAGGLE_API_TOKEN` secret, runs the
training and evaluation pipelines against a remote MLflow server, and moves the `champion` alias only if the
challenger wins on the cost metric.

## Model Performance

*Filled in once the training and evaluation pipelines are implemented.*

| Metric | Validation (time-based) | Notes |
|---|---|---|
| PR-AUC | – | main metric for a ~3.5% positive class |
| ROC-AUC | – | the Kaggle leaderboard metric |
| Recall @ 1% FPR | – | fraud caught when 1 in 100 good transactions is flagged |
| Expected cost / 1k transactions | – | at the cost-optimal threshold |

## Model Card

See [docs/model_card.md](docs/model_card.md) for intended use, data, evaluation, limitations and fairness considerations.

## Contributing

1. Fork the repository and create a feature branch
2. Run `make lint test` before committing (pre-commit runs ruff automatically)
3. Open a pull request; CI must pass

## License

MIT, see [LICENSE](LICENSE).

## Contact

Mikolaj Kurtyka

## Acknowledgments

- [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) dataset by Vesta Corporation and the IEEE Computational Intelligence Society
- [Kedro](https://kedro.org), [kedro-mlflow](https://github.com/Galileo-Galilei/kedro-mlflow), [MLflow](https://mlflow.org)
