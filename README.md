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
| `data_ingestion` pipeline (typing per table, join, schema fix) | Done |
| FastAPI service (`/predict`, `/predict/batch`, `/explain`, `/health/live`, `/health/ready`, Prometheus `/metrics`), Dockerfile, docker compose | Done |
| CI (lint, tests, Kedro load check, Docker build) | Done |
| `data_cleaning` (time split, column filter fitted on train) | Done |
| `feature_engineering` (time features, client UID, aggregations, encodings) | Done |
| `training` (LightGBM / XGBoost / CatBoost + Optuna, 3 scikit-learn baselines, optional time-series CV, refit on train + valid, MLflow child runs) | Code done. Baseline trained; **Optuna tuning not run yet** |
| `evaluation`: precision / recall / F1 at 4 thresholds, PR + ROC + cost curves, logged into MLflow | Done |
| `evaluation`: Kaggle submissions per model + rank-average blend | Done (files written, not uploaded yet) |
| Run history table from MLflow (`scripts/compare_runs.py`) | Done |
| `evaluation`: cost-based threshold (used by the API), global SHAP importance | Done |
| Inference wrapper for the API (`InferenceBundle` + `FraudPyfunc` in `src/fraud_detection/inference.py`) | Done |
| NumPy fast path for serving (`fast_inference.py`, ~2 ms per row, bit-identical to the pandas path) | Done |
| Open-loop async load test, 1–10000 requests/min × 1–1000 rows (`make loadtest`) | Done |
| Feature-count study (accuracy / size / latency vs top-N features, all 6 models) and per-model serving latency (`make feature-study`, `make serving-latency`) | Done |
| `production`: a few candidates (model type + top-N features) on an honest holdout; one is chosen by latency, PR-AUC and feature count and becomes the Kaggle submission and the API model; `@champion` moves only if holdout PR-AUC improves (`make production`) | Done |
| Every trained model (train-only and refit) stored in its MLflow run, not only on disk | Done |
| `data_validation` (pandera schemas, train/test parity) | Done |
| Per-prediction SHAP explanations (`POST /explain`) | Done |
| Scheduled retraining (`.github/workflows/retrain.yml`) | Done; needs a hosted MLflow server and two secrets |

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

```mermaid
flowchart LR
    subgraph kedro["Kedro project (kedro run)"]
        direction LR
        RAW[("01_raw<br/>Kaggle CSVs<br/>+ SHA256 manifest")]
        ING["data_ingestion<br/>type + join"]
        VAL["data_validation<br/>pandera checks"]
        CLN["data_cleaning<br/>time split +<br/>column filter"]
        FE["feature_engineering<br/>D-norm, UID,<br/>aggregates, encodings"]
        TR["training<br/>3 boosting + 3 baselines<br/>Optuna, refit"]
        EV["evaluation<br/>4 thresholds incl. min cost,<br/>curves, SHAP, submissions"]
        SV["production<br/>candidates on top-N features,<br/>holdout, choose one →<br/>submission + MLflow pyfunc"]
        RAW --> ING --> VAL
        ING --> CLN --> FE --> TR --> EV
        TR -->|"best params,<br/>feature ranking"| SV
    end

    subgraph compose["docker compose"]
        PG[(Postgres<br/>metadata)]
        MF["MLflow server<br/>tracking + registry<br/>alias @champion"]
        API["FastAPI<br/>/predict · /predict/batch<br/>/explain · /health/* · /metrics"]
        MF --> PG
        API -->|"load pinned version<br/>at startup"| MF
    end

    TR & EV -.->|"runs, metrics,<br/>artifacts"| MF
    SV -->|"register; move @champion<br/>only if holdout PR-AUC improves"| MF
    Client([Client]) -->|"POST /predict"| API
    API -->|"probability + decision<br/>(cost-optimal threshold)"| Client
```

GitHub Actions: `ci.yml` runs lint → tests → Kedro load check → Docker image build on every push and
PR; `retrain.yml` retrains the boosting models and the production candidates every Monday.

## Key Components

**Kedro pipelines** (`src/fraud_detection/pipelines/`)

| Pipeline | Input → Output | What it does |
|---|---|---|
| `data_ingestion` | `01_raw` → `02_intermediate` | Types each table against the schema in parameters (categorical / numerical, ID, optional target), fixes test column names (`id-01` → `id_01`), left-joins identity onto transactions, adds `has_identity` |
| `data_validation` | `02_intermediate` → `08_reporting` | pandera checks: dtypes, unique IDs, value ranges, required columns, time order, fraud rate; train/test parity (columns, dtypes, test after train, null rates). Stops the run if a check fails |
| `data_cleaning` | `02` → `03_primary` | Time split (earlier 80% of days = train, rest = valid); drops near-empty, near-constant and correlated V columns with a filter fitted on train |
| `feature_engineering` | `03` → `04_feature` → `05_model_input` | Time features, normalized D columns, email parts, client UID proxy, group-ratio / frequency / UID aggregates and category encodings fitted on train |
| `training` | `05` → `06_models` | LightGBM, XGBoost (GPU), CatBoost (GPU) with Optuna tuning, plus logistic regression, random forest and HistGradientBoosting baselines; early stopping on valid, optional time-series CV, refit on train + valid for the submission; one nested MLflow run per model |
| `evaluation` | `06` → `08_reporting` | Precision / recall / F1 at 4 thresholds incl. the cost-optimal one, PR + ROC + cost curves, global SHAP importance (boosting models), Kaggle submissions per model + blend |
| `production` | research `06` / `08` + `05` → `07` + MLflow registry | A few candidates, each a model type with its tuned params on its top-N features. Splits valid into valid + holdout, reports each candidate on the holdout and refits it for its own submission. Chooses one (latency budget, best holdout PR-AUC, fewest features within a tolerance), copies its file to `submission_production.csv`, wraps it with its column filter and encoders as an MLflow pyfunc, moves `@champion` only if holdout PR-AUC improves |

**MLflow** (through `kedro-mlflow`) records parameters, metrics and artifacts for every `kedro run`.
Every trained model is stored in its own run (`model/model.pkl`, `model/final_model.pkl`), so
retraining never loses an old model; `data/06_models/` only holds the latest copies. The registry
holds the production models; the one that serves traffic carries the `champion` alias.

**Scoring API** (`src/fraud_detection/api/`) is a FastAPI app that resolves `models:/fraud-detection@champion`
to a concrete version at startup and exposes `/predict`, `/predict/batch`, `/explain`, `/health/live`,
`/health/ready` and Prometheus `/metrics`. It still starts without a model; `/health/ready` then returns
503 with the reason, so a deployment never fails silently. A new champion is picked up on restart.

> **Not production-ready yet.** These gaps are deliberately out of scope here and must be handled
> before the API takes real traffic:
>
> - **No authentication or rate limiting.** Anyone who can reach the port can score and call
>   `/explain`. Put an API gateway or reverse proxy in front (API keys or mTLS, per-client limits).
> - **No hot reload of the model.** A new `@champion` is only served after the API restarts. Trigger
>   a rolling restart on registry changes, or check the alias periodically.
> - **No drift monitoring.** Predictions are logged and the score histogram is on `/metrics`, but
>   nothing compares live feature or score distributions against the training data.
>
> Details and typical fixes: [docs/07_serving_api.md](docs/07_serving_api.md#before-real-production-traffic).

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
├── docs/                          # numbered docs, start at 01_introduction.md
├── notebooks/                     # exploration only, no pipeline logic
├── scripts/
│   ├── download_data.py           # Kaggle download, idempotent, writes manifest
│   ├── compare_runs.py            # history of trained models from MLflow (make compare)
│   ├── feature_size_study.py      # accuracy / size / latency vs number of features (make feature-study)
│   ├── serving_latency.py         # single-row latency per model: pandas vs NumPy, threads (make serving-latency)
│   ├── loadtest.py                # open-loop load test of a running API (make loadtest)
│   ├── setup.ps1                  # one-command setup (Windows)
│   └── setup.sh                   # one-command setup (Linux/macOS)
├── src/fraud_detection/
│   ├── pipelines/                 # one folder per Kedro pipeline
│   ├── api/                       # FastAPI scoring service (+ optional micro-batching)
│   ├── inference.py               # InferenceBundle (raw rows → probability) + MLflow pyfunc
│   ├── fast_inference.py          # NumPy fast path the API uses (same scores)
│   ├── keys.py                    # stable string keys for categories
│   ├── pipeline_registry.py
│   └── settings.py
├── tests/
├── .github/workflows/          # ci.yml (every push / PR), retrain.yml (weekly)
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
uv run kedro run --pipelines data_ingestion                      # make ingest
uv run kedro run --pipelines data_validation                     # make validate
uv run kedro run --pipelines data_cleaning,feature_engineering   # make features
uv run kedro run --pipelines training                            # 3 boosting models + Optuna, 3 baselines
uv run kedro run --pipelines production                          # make production: candidates, choose the Kaggle + API model
uv run kedro run                                                 # make run (all pipelines)
uv run kedro registry list                                       # list pipelines
uv run kedro viz run                                             # make viz: interactive pipeline graph
```

### Full run, step by step

Run from the project folder. Each step reads what the previous one saved, so you can stop and
resume anywhere:

```bash
uv run python scripts/download_data.py                           # 1. raw CSVs (once)
uv run kedro run --pipelines data_ingestion                      # 2. typed + joined   (~30 s)
uv run kedro run --pipelines data_validation                     #    pandera checks    (~5 s)
uv run kedro run --pipelines data_cleaning,feature_engineering   # 3. split, filter, features (~2 min)
uv run kedro run --pipelines training,evaluation                 # 4. research: tune + train all models, evaluate, submissions (up to ~3 h)
uv run python scripts/compare_runs.py                            # 5. results table of all runs
uv run kedro run --pipelines production                          # 6. production: candidates, holdout reports, choice, registry
```

Step 4 without tuning (~8 min): add
`--params "lightgbm.model_options.n_trials=0,xgboost.model_options.n_trials=0,catboost.model_options.n_trials=0"`.

### Training

`training` runs LightGBM (CPU), XGBoost (GPU) and CatBoost (GPU), then the three baselines
(logistic regression, random forest, HistGradientBoosting), one after another. Each model is tuned
with Optuna (`n_trials`, `timeout_minutes`) and then fitted once more with the best parameters,
with early stopping on the valid split, and finally refitted on train + valid for the Kaggle
submission. All settings are in `conf/base/parameters_training.yml`. Parameter optimisation only
happens when `n_trials > 0` (default 25 per boosting model, 0 for the baselines).

```bash
# One model only
uv run kedro run --pipelines training_lightgbm      # or training_xgboost / _catboost / _logreg / _random_forest / _hist_gb

# Quick baseline: skip tuning, train once with the base params (~8 min for all three)
uv run kedro run --pipelines training --params "lightgbm.model_options.n_trials=0,xgboost.model_options.n_trials=0,catboost.model_options.n_trials=0"   # make train-fast

# Change the tuning budget for one model
uv run kedro run --pipelines training_xgboost --params "xgboost.model_options.n_trials=50,xgboost.model_options.timeout_minutes=90"
```

No NVIDIA GPU? Set `device: cpu` for XGBoost and `device: CPU` for CatBoost in `parameters_training.yml`.

Outputs: `data/06_models/<model>.pkl` (train only, the validated one),
`data/06_models/<model>_final.pkl` (refitted on train + valid), `data/06_models/<model>_best_params.json`,
`data/08_reporting/<model>_metrics.json`, `<model>_cv_metrics.json` and `<model>_feature_importance.csv`.

### Evaluation: precision, recall, curves

```bash
uv run kedro run --pipelines evaluation   # make evaluate (~30 s, uses the saved models)
```

On the valid split, per model:

| Metric group | Threshold | Question it answers |
|---|---|---|
| `fixed_*` | 0.5 | What happens with the default cut-off? |
| `best_f1_*` | the one that maximises F1 on valid | Best balance of precision and recall (slightly optimistic: picked on valid) |
| `at_fpr_*` | where 1% of legitimate transactions get flagged | How much fraud is caught at a tolerable false-alarm rate? |
| `min_cost_*` | the one with the lowest expected cost on valid | Which cut-off loses the least money? (see [Cost of a threshold](#cost-of-a-threshold)) |

Each group has precision, recall, F1, FPR, the confusion-matrix counts (tp / fp / fn / tn) and
`*_cost_per_1k`, the expected cost per 1,000 transactions (`no_model_cost_per_1k` is the cost of
approving everything). They are saved to `data/08_reporting/<model>_evaluation.json`, the PR, ROC
and cost curves to `data/08_reporting/<model>_curves.png`, and both are added to the model's
MLflow run (metrics `eval_*`, artifact `evaluation_curves.png`). The production pipeline stores the
`min_cost` threshold with the registered model, so the API decides by cost.

For LightGBM, XGBoost and CatBoost the evaluation also computes **SHAP** values on 5,000 valid rows
(exact TreeSHAP from each library) and saves the mean |SHAP| per feature to
`data/08_reporting/<model>_shap_importance.csv` and `<model>_shap.png` (also in the MLflow run).

What these metrics mean in business terms, with blocks and losses per 10,000 transactions:
[Metrics Explained](#metrics-explained-what-they-mean-for-the-business).

### Kaggle submission

The file to upload is `data/07_model_output/submission_production.csv`, from the production
model ([below](#research-vs-production)). The evaluation pipeline also scores the Kaggle test set
(506,691 rows) with every research model, for comparison:

| File | Model |
|---|---|
| `data/07_model_output/submission_production.csv` | **Production model** (the chosen candidate) |
| `data/07_model_output/submission_production_<candidate>.csv` | Each production candidate, e.g. `submission_production_xgboost_200.csv` |
| `data/07_model_output/submission_lightgbm.csv` | LightGBM |
| `data/07_model_output/submission_xgboost.csv` | XGBoost |
| `data/07_model_output/submission_catboost.csv` | CatBoost |
| `data/07_model_output/submission_{logreg,random_forest,hist_gb}.csv` | Baselines (for comparison) |
| `data/07_model_output/submission_blend.csv` | Rank average of the three boosting models |

Upload (needs your Kaggle token and the competition rules accepted; Kaggle allows a limited number
of submissions per day):

```bash
uv run kaggle competitions submit -c ieee-fraud-detection -f data/07_model_output/submission_production.csv -m "production"
uv run kaggle competitions submissions -c ieee-fraud-detection   # see the scores
```

Or upload the CSV on the competition page → **Submit Prediction**.

Submissions come from `<model>_final.pkl`: the model refitted on all labelled data (train + valid,
`refit: true`) with `best_iteration × 1.2` trees, because the valid days are the closest to test.
All valid metrics still come from the train-only `<model>.pkl`.

### Research vs production

The project has two tracks:

| | Research (`training`, `evaluation`, scripts) | Production (`production`) |
|---|---|---|
| Purpose | Show what was tried: 6 models, all 395 features, Optuna, feature-size study | One model for both the Kaggle submission and the API |
| Models | Every model type | A short list of **candidates**, e.g. `lightgbm_100`, `xgboost_200` |
| Features | All | Each candidate's top `n_features` of its model's research importance |
| Params | Optuna search | The research `best_params`, plus optional `param_overrides` (e.g. smaller trees) |
| Scored on | Valid (also used for early stopping and tuning, so optimistic) | A **holdout**: the later half of valid, never fitted, early-stopped or picked on |
| Output | `submission_<model>.csv`, reports | One submission per candidate; the chosen one as `submission_production.csv` and MLflow `@champion` |

The candidates are listed in `conf/base/parameters_production.yml`; pick them from the
[feature-size study](#fewer-features-smaller-models-faster-serving) rather than trying every model
at every size (that would take many hours, and choosing among many on the holdout makes it
optimistic too). Each candidate reads the research outputs of its model type live
(`<model>_best_params.json`, `<model>_feature_importance.csv`), so train that model first.

```yaml
production:
  candidates:
    lightgbm_100: {model: lightgbm, n_features: 100}
    lightgbm_50: {model: lightgbm, n_features: 50}
    xgboost_200: {model: xgboost, n_features: 200}
    catboost_100: {model: catboost, n_features: 100}
  selection:
    max_latency_p99_ms: 20    # one-row model latency budget
    pr_auc_tolerance: 0.005   # this close to the best counts as equally good
```

The choice, from the candidates' holdout reports:

1. drop the candidates whose one-row p99 latency is above `max_latency_p99_ms`;
2. take the best holdout PR-AUC among the rest;
3. of the candidates within `pr_auc_tolerance` of it, take the one with the fewest features.

So a smaller model wins when it is as good as the best one, and a slow model never reaches the API.
The chosen candidate's train-only model is registered (the API serves it, because its threshold was
picked for it) and `@champion` moves to it only if its `holdout_pr_auc` beats the current champion.
Its refit on all labelled data becomes `submission_production.csv`.

```bash
uv run kedro run --pipelines production                   # make production: every candidate + the choice
uv run kedro run --pipelines production_xgboost_200       # make candidate CANDIDATE=xgboost_200 (registers nothing)
uv run kedro run --pipelines production_choose            # make choose: choose again from the saved results
```

| Output | What it holds |
|---|---|
| `data/08_reporting/production_candidates.md` | Every candidate side by side: holdout PR-AUC / ROC-AUC, cost per 1k, features, latency, which one was chosen |
| `data/08_reporting/production_choice.json` | The chosen candidate and the rule's inputs |
| `data/08_reporting/production_<candidate>_evaluation.json` | One candidate's valid metrics (where the threshold is picked) and `holdout_*` metrics at that threshold, `n_features`, `latency_p50_ms` / `latency_p99_ms` |
| `data/06_models/production_<candidate>.pkl`, `_final.pkl`, `_features.json` | Its train-only model, its refit, its feature list |
| `data/07_model_output/submission_production_<candidate>.csv` | Its Kaggle file |

The holdout is mostly clean, not perfectly clean: the research params and the feature ranking were
chosen on the full valid, which includes the holdout days, and choosing among a few candidates on
it adds a little optimism to the winner's number.

### Watching a run

Open each command in its own terminal, from the project folder:

```powershell
# 1. Live log: tuning trials, AUC every 500 rounds, final metrics (Ctrl+C stops following, not the run)
Get-Content logs\info.log -Wait -Tail 30     # PowerShell
tail -f logs/info.log                        # bash / macOS / Linux

# 2. GPU usage, refreshed every 2 s (rises for XGBoost / CatBoost, not for LightGBM)
nvidia-smi -l 2
```

### Experiment tracking (MLflow)

By default runs are logged to a local SQLite store (`mlflow.db`, artifacts in `./mlruns`).
Start the UI and keep that terminal open while you use it:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000   # make mlflow
```

Then open http://localhost:5000. `ERR_CONNECTION_REFUSED` means the UI server is not running (or
its terminal was closed); start it with the command above. Run it **from the project folder**,
otherwise `sqlite:///mlflow.db` points to a new, empty database.

Direct links (open them while the UI is running):

| View | Link |
|---|---|
| The model runs only (one per model type), best PR-AUC first | http://localhost:5000/#/experiments/1/runs?searchFilter=tags.stage%20%3D%20%27final%27&orderByKey=metrics.%60valid_pr_auc%60&orderByAsc=false |
| Same runs as bar charts, one chart per metric | http://localhost:5000/#/experiments/1/runs?searchFilter=tags.stage%20%3D%20%27final%27&orderByKey=metrics.%60valid_pr_auc%60&orderByAsc=false&compareRunsMode=CHART |
| All runs (pipelines, tuning, models) | http://localhost:5000/#/experiments/1/runs |

Why the default page looks empty:

- The switch at the top left defaults to **GenAI** (traces, prompts, judges: for LLM apps). Click
  **Model training**.
- The model runs are nested under their `training` parent run: click **+** next to it, or use the
  filter `tags.stage = 'final'` as in the links above.
- The table shows no metric columns by default: **Columns** → tick `valid_pr_auc`,
  `valid_roc_auc`, `eval_best_f1_precision`, `eval_best_f1_recall`, … Or click a run name: its
  page lists every metric, parameter and the `evaluation_curves.png` artifact.

Details: [docs/02_infrastructure.md](docs/02_infrastructure.md#experiment-tracking-mlflow).

History of all trained models as one table (also saved to `data/08_reporting/model_comparison.csv` / `.md`):

```bash
uv run python scripts/compare_runs.py            # all runs, newest first
uv run python scripts/compare_runs.py --last 1   # only the latest kedro run
```

What every hyperparameter means, with base values and Optuna ranges:
[docs/06_modeling_approach.md](docs/06_modeling_approach.md#5-hyperparameters-explained).

How runs are organised:

- Experiment **fraud-detection** holds one parent run per `kedro run`.
- Expand a parent run (**+**) to see one child run per model: **lightgbm**, **xgboost**, **catboost**,
  **logreg**, **random_forest**, **hist_gb**.
  They log the same metric names (`valid_roc_auc`, `valid_pr_auc`, `train_roc_auc`, `roc_auc_gap`,
  `best_iteration`, `fit_seconds`), so add those columns and sort to compare models.
- Each model child run stores `feature_importance.csv` under **Artifacts**.
- With tuning on, `<model>-tuning` runs hold one nested run per Optuna trial; select them and click
  **Compare** to see parameters against `valid_pr_auc`.

To log to the docker compose server instead, set `MLFLOW_TRACKING_URI=http://localhost:5000` before `kedro run`.

### Serving stack (Docker)

```bash
docker compose up -d --build    # make docker-up
```

| Service | URL | Purpose |
|---|---|---|
| MLflow | http://localhost:5000 | Tracking UI and model registry (Postgres backend) |
| API | http://localhost:8000/docs | Swagger UI for `/predict`, `/predict/batch`, `/explain` and `/health/*` |

The API container waits until the MLflow server answers its healthcheck, because it loads the
`@champion` model once at startup. Register a model first (`make production` with
`MLFLOW_TRACKING_URI=http://localhost:5000`), then restart the API (`docker compose restart api`).

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"TransactionID": 3663549, "features": {"TransactionDT": 18403224, "TransactionAmt": 31.95, "ProductCD": "W", "card1": 10409}}'
```

```json
{"TransactionID": 3663549, "fraud_probability": 0.031, "is_fraud": false, "threshold": 0.077,
 "model_version": "2", "model_uri": "models:/fraud-detection/2"}
```

`model_uri` is the pinned version the alias pointed to at startup, so every response says exactly
which model scored it.

`POST /explain?top_k=5` takes the same body and adds the features that pushed this score up or
down most (SHAP values in log-odds, ~110 ms per row, so it is kept off `/predict`):

```json
{"TransactionID": 2988500, "fraud_probability": 0.0026, "is_fraud": false, "threshold": 0.077,
 "contributions": [{"feature": "V30", "value": 2.0, "shap": -0.358},
                   {"feature": "card1_freq", "value": 0.0019, "shap": 0.223},
                   {"feature": "card5", "value": "166", "shap": -0.218}, ...]}
```

Until a model is registered and given the `champion` alias, `/health/ready` returns `503` with
`"model_loaded": false` and the load error, and `/predict` returns `503`.

### Tests and code quality

```bash
uv run pytest            # make test
uv run ruff check .      # make lint
```

## CI/CD

`.github/workflows/ci.yml` runs on every push to `main` and on every pull request:

1. **lint-test**: `uv sync --frozen` → ruff lint and format check → pytest with coverage → `kedro registry list` (checks the project loads)
2. **docker**: builds the API image with layer caching (needs lint-test to pass)

`.github/workflows/retrain.yml` retrains every Monday (and on demand from the Actions tab): it downloads
the data, runs ingestion, validation and features, tunes and trains the research LightGBM, XGBoost and
CatBoost on the CPU (5 trials each by default), then builds the production candidates, chooses one and
registers it on a remote MLflow server, moving `@champion` only if it beats the current one on holdout
PR-AUC. It needs the `KAGGLE_API_TOKEN` and `MLFLOW_TRACKING_URI` secrets (plus credentials if the
server needs them) and skips with a notice without them. The MLflow server must serve artifacts, so the
API can load what CI registered.

## Model Performance

Baseline (base params, no tuning). Train = days 1–140, valid = days 141–182, 395 features:

| Model | Valid ROC-AUC | Valid PR-AUC | Train ROC-AUC | Gap | Fit time |
|---|---|---|---|---|---|
| LightGBM (CPU) | 0.9315 | 0.6137 | 0.9991 | 0.068 | 92 s |
| XGBoost (GPU) | 0.9292 | 0.6117 | 0.9962 | 0.067 | 59 s |
| CatBoost (GPU) | 0.9056 | 0.5449 | 0.9709 | 0.065 | 291 s |

Precision and recall on valid (fraud rate 3.4%):

| Model | Precision @0.5 | Recall @0.5 | Best F1 (threshold) | Precision / recall @best F1 | Recall @1% FPR |
|---|---|---|---|---|---|
| LightGBM | 0.809 | 0.448 | 0.596 (0.358) | 0.731 / 0.503 | 0.537 |
| XGBoost | 0.836 | 0.421 | 0.589 (0.259) | 0.678 / 0.520 | 0.531 |
| CatBoost | 0.802 | 0.392 | 0.548 (0.373) | 0.729 / 0.439 | 0.467 |

At 1% false alarms, LightGBM catches about 54% of fraud.

The ~0.07 train–valid gap points to overfitting and to the client-UID aggregates, which are fitted on
train and include each training row itself. Valid scores are slightly optimistic: the same split is used
for early stopping and tuning.

Summary for LightGBM (baseline):

| Metric | Validation (time-based) | Notes |
|---|---|---|
| PR-AUC | 0.614 | main metric for a ~3.5% positive class (random model: 0.034) |
| ROC-AUC | 0.931 | the Kaggle leaderboard metric |
| Recall @ 1% FPR | 0.537 | fraud caught when 1 in 100 good transactions is flagged |
| Expected cost / 10k transactions | $27.8k (vs $51.6k with no model) | at the cost-optimal threshold, assumptions in the next section |

### Fewer features, smaller models, faster serving

`make feature-study` retrains each model on its top-N features, with the tuned params and with
small "compact" trees. Valid PR-AUC, tuned params:

| Model | 395 features | 200 | 100 | 50 | 25 |
|---|---|---|---|---|---|
| XGBoost | 0.616 | **0.623** | 0.614 | 0.601 | 0.561 |
| LightGBM | 0.613 | 0.616 | 0.611 | 0.601 | 0.559 |
| HistGB | 0.591 | 0.613 | 0.597 | 0.593 | 0.549 |
| CatBoost | 0.567 | 0.573 | 0.551 | 0.554 | 0.532 |

- The top 100–150 features keep the full model's accuracy, and ~50 lose only ~0.01.
- Model size comes from the tree settings, not the feature count. Compact trees are 3–10× smaller
  (1.5–5 MB) for 0.01–0.04 less PR-AUC.
- Serving speed is decided by the code path, not the feature count. On the API's NumPy fast path,
  LightGBM scores a raw row in **~2 ms (0.8 ms single-threaded)**. The other model types need a
  pandas DataFrame and take ~20–65 ms (~20–40 ms single-threaded; `make serving-latency`). See
  [Speed analysis](#speed-analysis-data-handling-model-features).

The study ranks features by a model early-stopped on valid and scores the top-N runs on the same
valid, so its numbers lean slightly optimistic. That is accepted: the research track keeps every
labelled day for training, which is what the Kaggle score needs. The production
[holdout](#research-vs-production) gives the honest number for each candidate you pick from it.

Details: [Modeling approach](docs/06_modeling_approach.md#how-many-features-are-needed) and
[Serving API](docs/07_serving_api.md#latency-per-model-type).

### Speed analysis: data handling, model, features

Scoring one raw transaction has two parts: **building the features** from the raw row (types,
column filter, time features, encoders, UID aggregates) and **running the model**. Three choices
set the time, in this order of impact:

| Lever | Options | One-row effect |
|---|---|---|
| 1. Data handling | pandas DataFrame vs NumPy arrays | **~110–185 ms → ~2–65 ms** |
| 2. Model type | LightGBM vs XGBoost / CatBoost / HistGB / RF / LogReg | **~2 ms vs ~20–65 ms** on the NumPy path |
| 3. Number of features | 395 → 100 → 50 | model time roughly **÷2** |

**1 + 2. Data handling and model type** (`make serving-latency`, all 395 features, one row,
p50 in ms, `data/08_reporting/serving_latency.csv`):

| Model | pandas path | NumPy fast path | NumPy, 1 thread | Model only, 1 thread |
|---|---|---|---|---|
| **LightGBM** | 166 | 2.0 | **0.8** | 7.6 |
| CatBoost | 110 | 19 | 18 | 12 |
| XGBoost | 172 | 51 | 22 | 14 |
| HistGB | 185 | 66 | 24 | 15 |
| Random forest | 138 | 61 | 26 | 20 |
| Logistic regression | 114 | 40 | 38 | 33 |

- **pandas is the bottleneck, not the model.** The pandas path (`InferenceBundle.predict_proba`,
  the same node functions the batch pipelines use) makes thousands of small pandas calls, each with
  ~0.1–1 ms of fixed overhead, so one row costs ~110–185 ms whatever the model. The NumPy fast path
  (`fast_inference.py`, used by the API) does the same steps on arrays and dict lookups and gives
  **bit-identical scores** (tested in `tests/test_inference.py`).
- **LightGBM is the only model that stays NumPy end to end**: it takes the float matrix directly.
  The others need a DataFrame with category dtypes rebuilt from it, which adds 10–35 ms.
- **Threads hurt single rows.** Default `n_jobs=-1` / OpenMP starts a thread pool per call; one
  thread is 2–3× faster for XGBoost, HistGB and random forest.

**3. Number of features** (`make feature-study`, tuned params, model time on one row, p50 in ms):

| Model | 395 features | 200 | 100 | 50 | 25 |
|---|---|---|---|---|---|
| LightGBM | 21.8 | 16.1 | 11.1 | 9.5 | 5.2 |
| XGBoost | 45.4 | 21.0 | 16.6 | 12.1 | 8.2 |
| CatBoost | 14.6 | 8.8 | 4.8 | 3.1 | 2.1 |
| HistGB | 50.6 | 55.1 | 40.8 | 31.9 | 31.1 |
| Random forest | 57.2 | 57.6 | 47.3 | 45.0 | 44.0 |
| Logistic regression | 35.0 | 24.9 | 11.0 | 6.3 | 3.9 |

- Going from 395 to 100 features roughly halves the model time and costs ≤0.002 PR-AUC for LightGBM and
  XGBoost (0.016 for CatBoost)
  ([above](#fewer-features-smaller-models-faster-serving)), but it does not touch the 100+ ms
  pandas overhead: feature count is a fine-tuning lever, data handling is the big one.
- These model times are measured on a DataFrame with default threads, so they are higher than
  the NumPy / single-thread numbers in the previous table; compare within one table.

Bottom line: the API serves **LightGBM on the NumPy fast path, ~2 ms per row** (~0.8 ms with one
thread), about **80× faster** than the same model and features through pandas. Measured on one
Windows laptop (32 threads); single runs vary by ±30%.

## Metrics Explained: What They Mean for the Business

The model does not say "fraud" or "not fraud". It gives every transaction a **fraud score
between 0 and 1**. The business chooses a **threshold**: transactions above it are blocked
(or sent for review). The metrics below answer two different questions:

- **How well does the model rank transactions?** (ROC-AUC, PR-AUC). This does not depend on the threshold, so these metrics are used to compare models.
- **What happens at our chosen threshold?** (precision, recall, false positive rate, cost). These metrics are used to make the decision.

### The four outcomes

| | Really fraud | Really legitimate |
|---|---|---|
| **Blocked** | **True positive (TP)**: fraud stopped | **False positive (FP)**: good customer blocked for no reason |
| **Approved** | **False negative (FN)**: fraud gets through, we pay the chargeback | **True negative (TN)**: normal purchase |

Every metric below is a ratio of these four counts.

### Precision, recall, false positive rate

| Metric | Formula | Plain question | LightGBM @ 0.5 |
|---|---|---|---|
| **Precision** | TP / (TP + FP) | Of the transactions we blocked, how many were really fraud? | 0.81: about 4 in 5 blocks are correct |
| **Recall** (true positive rate) | TP / (TP + FN) | Of all fraud, how much did we catch? | 0.45: we catch less than half of the fraud |
| **False positive rate (FPR)** | FP / (FP + TN) | Of all good customers, how many did we block? | 0.4%: about 1 in 265 good customers |
| **F1** | harmonic mean of precision and recall | One number that balances both | 0.58 |

Precision and recall pull in opposite directions. A lower threshold catches more fraud
(higher recall) but blocks more good customers (lower precision). There is no setting that wins on both.

**Why accuracy is not used:** only 3.4% of transactions are fraud, so a "model" that approves
everything is 96.6% accurate and catches nothing.

**Why FPR can look better than it is:** good customers outnumber fraud about 28 to 1. An FPR of
"only 1%" still means **96 blocked good customers per 10,000 transactions**, compared with
185 frauds caught. FPR is always worth reading next to the absolute counts.

### ROC-AUC and PR-AUC

- **ROC-AUC (0.931)**: take one random fraud and one random legitimate transaction. ROC-AUC is the probability that the
  model gives the fraud the higher score. 0.5 means random, 1.0 means perfect. It is the Kaggle metric, but
  it looks high on imbalanced data because the huge number of easy legitimate transactions inflates it.
- **PR-AUC (0.614)**: the average precision over all recall levels. A random model scores the fraud rate
  (0.034), so 0.61 is about 18 times better than random. It only looks at how well fraud is found, so it is
  the more honest metric here and the one Optuna optimises.

The curves for each model are in `data/08_reporting/<model>_curves.png`.

### Per 10,000 transactions: what each threshold means in practice

LightGBM on the valid split, scaled to 10,000 transactions (344 of them fraud, worth $51.6k):

| Threshold | Blocked | Fraud caught (TP) | Good customers blocked (FP) | Fraud missed (FN) | Precision | Recall | Fraud $ stopped | Fraud $ lost |
|---|---|---|---|---|---|---|---|---|
| 0.05 | 899 | 256 | 643 | 88 | 0.28 | 0.74 | $36.8k | $14.8k |
| **0.07** (cost-optimal, see below) | 683 | 239 | 444 | 105 | 0.35 | 0.70 | $33.9k | $17.7k |
| 0.10 | 528 | 223 | 306 | 121 | 0.42 | 0.65 | $30.5k | $21.1k |
| 0.20 | 339 | 195 | 144 | 149 | 0.58 | 0.57 | $24.7k | $26.9k |
| 0.27 (1% FPR) | 280 | 185 | 96 | 159 | 0.66 | 0.54 | $22.9k | $28.6k |
| 0.36 (best F1) | 236 | 173 | 64 | 171 | 0.73 | 0.50 | $20.7k | $30.9k |
| 0.50 (default) | 190 | 154 | 36 | 190 | 0.81 | 0.45 | $17.7k | $33.9k |
| 0.70 | 144 | 127 | 17 | 217 | 0.88 | 0.37 | $13.9k | $37.7k |

How to read one row (0.5): out of 10,000 card payments we block 190. 154 of them are fraud and 36 are good
customers who now call support. 190 frauds still get through and cost $33.9k in chargebacks.

**Recall by count is not recall by money.** At 0.5 the model catches 45% of fraud transactions but only 34%
of fraud dollars: it misses large frauds more often than small ones. The business loses money in dollars,
so the dollar figure is the one that matters.

### Cost of a threshold

Each outcome has a price. For the example we use these **assumptions** (replace them with real figures):

| Outcome | Cost | Why |
|---|---|---|
| Missed fraud (FN) | the transaction amount | chargeback to the merchant/issuer |
| Every blocked transaction (TP + FP) | $5 | analyst review or step-up check (SMS, 3-D Secure) |
| Good customer blocked (FP) | +$15 | support call, abandoned purchase, risk the customer leaves |

Total cost per 10,000 transactions:

| Threshold | Missed fraud | Reviews | Customer friction | **Total** |
|---|---|---|---|---|
| no model (approve all) | $51.6k | $0 | $0 | **$51.6k** |
| 0.05 | $14.8k | $4.5k | $9.6k | **$28.9k** |
| **0.07** | $17.7k | $3.4k | $6.7k | **$27.8k** (lowest) |
| 0.10 | $21.1k | $2.6k | $4.6k | **$28.3k** |
| 0.27 (1% FPR) | $28.6k | $1.4k | $1.4k | **$31.5k** |
| 0.50 (default) | $33.9k | $1.0k | $0.5k | **$35.4k** |

With these costs the model **cuts fraud-related cost by 46%** ($51.6k → $27.8k per 10k transactions), and
the default threshold of 0.5 leaves $7.6k per 10k on the table. The cost-optimal threshold is much lower
than 0.5 because a missed fraud ($150 on average) costs far more than an unnecessary review.

**The best threshold depends on the cost of a blocked good customer.** If a false block costs $50
instead of $15 (for example, premium customers who leave), the optimum moves from 0.07 to **0.28**: 277
blocks per 10k, 93 of them good customers, recall 0.53. So the costs have to be agreed with the business
before a threshold can be chosen.

### What to focus on

1. **Compare models with PR-AUC** (and ROC-AUC for Kaggle). Set the threshold with **expected cost**, not 0.5 and not F1.
2. **Agree the cost matrix with the business first**: chargeback cost, review cost per alert, and the cost of a blocked good customer (complaints, churn).
3. **Respect review capacity.** If analysts can check 200 alerts per 10k transactions, the threshold cannot go below ~0.45 however cheap reviews are. In that case report precision at the top-N alerts.
4. **Use bands, not one cut-off**: approve when the score is low, send a step-up check (SMS / 3-D Secure) in the middle band, block when the score is high. A step-up check is much cheaper friction than a decline, which makes a low threshold affordable.
5. **Track dollars, not only counts**: the fraud dollars caught, the fraud dollars lost, and the number of good customers blocked per 10k.
6. **Re-check over time**: fraud patterns drift, so the threshold chosen on one month of data does not stay optimal.

These numbers come from the LightGBM baseline predictions on the valid split. The cost assumptions
are illustrative. The evaluation pipeline recomputes the cost-optimal threshold on every run with the costs in
`conf/base/parameters_evaluation.yml` (`evaluation.cost`): for the same LightGBM it finds 0.077 at $2,764 per
1k transactions ($27.6k per 10k), and the API uses that threshold.

## Model Card

See [docs/08_model_card.md](docs/08_model_card.md) for intended use, data, evaluation, limitations and fairness considerations.
The full documentation (infrastructure, pipeline diagrams, modeling approach, API) starts at
[docs/01_introduction.md](docs/01_introduction.md).

## Acknowledgments

- [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) dataset by Vesta Corporation and the IEEE Computational Intelligence Society
- [Kedro](https://kedro.org), [kedro-mlflow](https://github.com/Galileo-Galilei/kedro-mlflow), [MLflow](https://mlflow.org)
