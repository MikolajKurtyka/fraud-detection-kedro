# 5. Pipelines

[← 4. Data layers](04_data_layers.md) · [Table of contents](01_introduction.md#table-of-contents)

All pipeline code lives in `src/fraud_detection/pipelines/<name>/`. Each folder contains a
`nodes.py` (plain Python functions that you can test without Kedro) and a `pipeline.py` (wires the
nodes to catalog datasets and parameters). `pipeline_registry.py` registers them.

## Registered pipelines

| Name | Status | Nodes | Run with |
|---|---|---|---|
| `data_ingestion` | Done | 8 (4 per split) | `kedro run --pipelines data_ingestion` (`make ingest`) |
| `data_validation` | Planned, empty | 0 | — |
| `data_cleaning` | Done | 5 | `kedro run --pipelines data_cleaning` |
| `feature_engineering` | Done | 10 | `kedro run --pipelines feature_engineering` |
| `training` | Done | 24 (4 per model, 6 models) | `kedro run --pipelines training` (`make train`) |
| `training_lightgbm` / `_xgboost` / `_catboost` / `_logreg` / `_random_forest` / `_hist_gb` | Done | 4 | One model only. Not part of `__default__` |
| `evaluation` | Done (metrics, curves, submissions); cost threshold / SHAP planned | 13 (2 per model + blend of the 3 boosting models) | `kedro run --pipelines evaluation` (`make evaluate`) |
| `serving` | Done | 1 | `kedro run --pipelines serving` (`make register`) |
| `serving_xgboost` / `_catboost` / ... | Done | 1 | Register another model type. Not part of `__default__` |
| `__default__` | — | all of the above except `training_*` / `serving_*` | `kedro run` (`make run`) |

Useful variations:

```bash
uv run kedro run --pipelines data_cleaning,feature_engineering   # make features
uv run kedro run --pipelines training --params "lightgbm.model_options.n_trials=0,xgboost.model_options.n_trials=0,catboost.model_options.n_trials=0"   # make train-fast
uv run kedro run --tags features            # by tag: ingestion | cleaning | features | training | lightgbm …
uv run kedro run --pipelines training,evaluation --namespaces logreg,random_forest,hist_gb   # make baselines
uv run kedro run --from-nodes fit_feature_encoders   # resume from a node
uv run kedro viz run                        # interactive graph of everything below
```

## Overview diagram

```mermaid
flowchart TB
    subgraph L1["01_raw"]
        RTT[raw_train_transaction]
        RTI[raw_train_identity]
        RST[raw_test_transaction]
        RSI[raw_test_identity]
    end

    subgraph P1["data_ingestion"]
        ING_TR["type → normalize → merge<br/>(train)"]
        ING_TE["type → normalize → merge<br/>(test)"]
    end

    subgraph L2["02_intermediate"]
        INT_TR[int_train]
        INT_TE[int_test]
    end

    subgraph P2["data_cleaning"]
        SPLIT[split_by_time]
        FITCF[fit_column_filter]
        APPCF["apply_column_filter ×3"]
    end

    subgraph L3["03_primary"]
        CF[/column_filter.json/]
        PRM["prm_train · prm_valid · prm_test"]
    end

    subgraph P3["feature_engineering"]
        STL["add_stateless_features ×3"]
        FITFE[fit_feature_encoders]
        APPFE["apply_feature_encoders ×3"]
        BMI["build_model_input ×3"]
    end

    subgraph L4["04_feature / 05_model_input"]
        FE[/feature_encoders.pkl/]
        FEAT["feat_train · feat_valid · feat_test"]
        MDL["mdl_in_train · mdl_in_valid · mdl_in_test"]
    end

    subgraph P4["training (per model: lightgbm · xgboost · catboost · logreg · random_forest · hist_gb)"]
        TUNE[tune_hyperparameters]
        TRAIN[train_model]
        CV[cross_validate]
        REFIT[refit_model]
    end

    subgraph L6["06_models / 08_reporting"]
        MOD[/"{model}.model · {model}.final_model · {model}.best_params"/]
        REP["{model}.metrics · {model}.feature_importance · {model}.cv_metrics"]
    end

    RTT & RTI --> ING_TR --> INT_TR
    RST & RSI --> ING_TE --> INT_TE
    INT_TR --> SPLIT -->|train_split| FITCF --> CF
    SPLIT -->|"train_split, valid_split"| APPCF
    INT_TE --> APPCF
    CF --> APPCF --> PRM
    PRM --> STL -->|train_stateless| FITFE --> FE
    STL --> APPFE
    FE --> APPFE --> FEAT --> BMI --> MDL
    FE --> BMI
    MDL -->|"mdl_in_train, mdl_in_valid"| TUNE --> MOD
    MDL --> TRAIN
    TUNE -->|best_params| TRAIN --> MOD & REP
    TRAIN -->|model| CV --> REP
    TRAIN -->|model| REFIT --> MOD

    MOD & MDL --> EVAL["evaluation<br/>precision / recall, curves,<br/>Kaggle submissions + blend"]
```

Parallelogram nodes (`/…/`) are **fitted artifacts**. They are learned once from the train split
and reused for every other split.

## The fit / apply pattern

Every step that *learns* something from data is split into two functions:

```mermaid
flowchart LR
    T[train split] --> FIT["fit_*()"] --> ART[/artifact<br/>json / pkl/]
    T --> APP["apply_*()"]
    V[valid split] --> APP
    S[test split] --> APP
    R[single API row] -.-> APP
    ART --> APP --> OUT[transformed data]
```

- `fit_*` only ever sees **train**, so valid and test scores are honest and nothing from the
  future leaks into training.
- `apply_*` is a pure function of `(data, artifact)` that works for any number of rows,
  **including one**. The test
  `tests/pipelines/test_feature_engineering.py::test_single_row_matches_batch` checks that scoring
  one row gives exactly the same features as scoring the whole table.

The full **inference chain** for new data is therefore:

```
raw rows ─► type_raw ─► apply_column_filter ─► add_stateless_features ─► apply_feature_encoders ─► build_model_input ─► FraudModel.predict_proba
          (ingestion     (column_filter)                                  (feature_encoders)        (feature_encoders)     ({model}.pkl)
           schema)
```

`InferenceBundle.predict_proba` (`src/fraud_detection/inference.py`) runs exactly this chain,
calling the pipelines' own node functions. `type_raw` replaces the ingestion pipeline for request
rows:

- it types only the columns the filter keeps;
- columns absent from the request become null;
- `has_identity` is 1 when any identity column is set, unless the client sends it explicitly.

See [7. Serving API](07_serving_api.md).

---

## data_ingestion

**01_raw → 02_intermediate.** Types each raw table against an explicit schema, aligns train and
test column names, and left-joins identity onto transactions. The same nodes run for both splits;
the target column is optional.

```mermaid
flowchart LR
    RT[raw_split_transaction] --> TT[type_transaction] --> TTy[split_transaction_typed]
    RI[raw_split_identity] --> NI[normalize_identity] --> IN[split_identity_normalized] --> TI[type_identity] --> ITy[split_identity_typed]
    TTy --> M[merge_split]
    ITy --> M --> INT[int_split]
```

| Node | Function | What it does |
|---|---|---|
| `type_{split}_transaction` | `type_transaction` | Validates the columns against `schema.transaction` (fails if any column is **missing or unexpected**, which catches schema drift); `TransactionID` must be unique and non-null (→ int64); `isFraud` must be 0/1 (→ int8) if present; numerical columns → `float32` with overrides; categorical columns → `category` of strings via `to_key` |
| `normalize_{split}_identity` | `normalize_identity_columns` | `id-01` → `id_01` (the Kaggle test file uses hyphens) |
| `type_{split}_identity` | `type_identity` | Same typing for the identity schema |
| `merge_{split}` | `merge_transaction_identity` | Left join on `TransactionID` (`validate="one_to_one"`), warns about orphan identity rows, adds `has_identity` (int8) |

**Parameters** (`parameters_data_ingestion.yml`):

| Key | Value | Why |
|---|---|---|
| `schema.{table}.categorical/numerical` | lists with ranges such as `"V{1..339}"`, `"id_{01..11}"` | Expanded by `expand_columns`; zero padding is kept |
| `numerical_dtype` | `float32` | Halves memory |
| `numerical_dtype_overrides` | `TransactionDT: int64`, `TransactionAmt: float64` | Test reaches ~34M seconds, which is beyond float32's exact-integer range; keeps cents exact |

---

## data_validation

*Planned (the pipeline is empty).* It will run pandera checks on `int_train` / `int_test` (types,
value ranges, null rates per column group), check train/test column parity and `TransactionDT`
ordering, and write a readable report to `08_reporting/data_validation.json`, failing fast when a
check breaks.

---

## data_cleaning

**02_intermediate → 03_primary.** Creates the time-based train/valid split and removes
uninformative columns using a filter fitted on train only.

```mermaid
flowchart LR
    IT[int_train] --> S[split_by_time]
    S -->|train_split| F[fit_column_filter] --> CF[/column_filter/]
    S -->|train_split| A1[apply_column_filter_train] --> P1[prm_train]
    S -->|valid_split| A2[apply_column_filter_valid] --> P2[prm_valid]
    IS[int_test] --> A3[apply_column_filter_test] --> P3[prm_test]
    CF --> A1 & A2 & A3
```

| Node | Function | What it does |
|---|---|---|
| `split_by_time` | `split_by_time` | `day = TransactionDT // 86400`; cutoff = 80th percentile of days; split on **whole days**, so a day never ends up in both sets. Result: train days 1–140 (472,006 rows, fraud 3.51%), valid days 141–182 (118,534 rows, fraud 3.44%) |
| `fit_column_filter` | `fit_column_filter` | Rules in order, each column reported under the first rule that hits it: (1) null share > `null_threshold`; (2) most common value (including null) > `top_value_threshold`; (3) for `V` columns: group by identical null count, sort by number of unique values, and drop a column if \|corr\| > `v_correlation_threshold` with a column already kept. Protected columns and the target are never dropped |
| `apply_column_filter_{split}` | `apply_column_filter` | Keeps `kept` columns in their saved order; raises if any expected column (other than the target) is missing |

Result of the current run: **80 of 435 columns dropped** (24 by the null/top-value rules, 56
correlated `V`), **355 kept** (269 of them `V` columns). The reason for every drop is in
`column_filter.json`, for example `"V321": "|corr| > 0.95 with V320"` or
`"C3": "top value share 0.9960"`. Apart from `V` columns, only `C3` and `id_07/08/21–27` are dropped.

**Parameters** (`parameters_data_cleaning.yml`):

| Key | Value | Note |
|---|---|---|
| `train_fraction` | 0.8 | Share of days that go to train |
| `null_threshold` | 0.99 | |
| `top_value_threshold` | 0.99 | At 0.9 this rule would remove rare but predictive columns (D7, dist2, many `id_*`), and even `isFraud` itself (96.5% zeros) |
| `v_column_pattern` | `^V\d+$` | |
| `v_correlation_threshold` | 0.95 | |
| `protected_columns` | ID, time, amount, `has_identity`, and every raw column that feature_engineering uses | Prevents the filter from breaking later steps |

---

## feature_engineering

**03_primary → 04_feature → 05_model_input.**

```mermaid
flowchart LR
    P[prm_split] --> S[add_stateless_features_split] --> ST[split_stateless]
    ST -->|train only| F[fit_feature_encoders] --> FE[/feature_encoders/]
    ST --> A[apply_feature_encoders_split] --> FT[feat_split]
    FE --> A
    FT --> B[build_model_input_split] --> M[mdl_in_split]
    FE --> B
```

| Node | Function | Features produced |
|---|---|---|
| `add_stateless_features_{split}` | `add_stateless_features` | Row by row, no fitting: `day`, `hour`, `weekday`; `TransactionAmt_log`, `TransactionAmt_cents`; `D{n}_norm = day − D{n}` for 14 D columns (D9 excluded: it's a fraction of a day); email parts `P_emaildomain_1/2/3`, `R_emaildomain_1/2/3`; `uid = card1_addr1_D1_norm` |
| `fit_feature_encoders` | `fit_feature_encoders` | From `train_stateless` only: **group stats** (mean/std of a value per group, kept only for groups with ≥ `min_group_size` rows); **frequency maps** (share of train rows per value); **categorical encodings** (≤100 categories → native `category`, more → integer codes); the final `feature_columns` list |
| `apply_feature_encoders_{split}` | `apply_feature_encoders` | Adds `X_to_mean_by` / `X_to_std_by` ratios (or `X_mean_by` without ratio, for the UID), `col_freq`. Unseen group → NaN, unseen value → frequency 0, unseen category → NaN |
| `build_model_input_{split}` | `build_model_input` | `TransactionID` + `isFraud` (if present) + `feature_columns` in a fixed order |

Result: 20 group features, 13 frequency columns, 37 native + 11 coded categorical columns,
**395 model features**.

**Parameters** (`parameters_feature_engineering.yml`):

| Key | Value |
|---|---|
| `d_columns` | D1–D15 without D9 |
| `uid_columns` | `[card1, addr1, D1_norm]` |
| `min_group_size` | 5 |
| `group_features` | `TransactionAmt`, `id_02`, `D15` by `card1`/`card4`/`addr1`/`addr2` (ratios), plus `TransactionAmt`, `D15` by `uid` (raw stats) |
| `frequency_columns` | `card1-3,5`, `addr1-2`, both email domains, `DeviceInfo`, `id_30/31/33`, `uid` |
| `max_native_categories` | 100 |
| `drop_columns` | `TransactionDT`, `day`, `uid`, raw D columns. Pure time, or replaced by the normalized versions |

---

## training

**05_model_input → 06_models + 08_reporting.** One *namespaced* copy of the same four-node pipeline
per model (three gradient-boosting libraries and three scikit-learn baselines). Namespacing prefixes every dataset and parameter: `lightgbm.model`,
`params:lightgbm.model_options`, and so on. Only `mdl_in_train`, `mdl_in_valid`, `params:target`
and `params:feature_engineering.join_key` are shared.

```mermaid
flowchart LR
    TR[mdl_in_train] & VA[mdl_in_valid] --> T["{model}.tune_hyperparameters"]
    O[/"params:{model}.model_options"/] --> T
    T --> BP["{model}.best_params"]
    TR & VA & BP & O --> F["{model}.train_model"]
    F --> M["{model}.model"]
    F --> ME["{model}.metrics"]
    F --> FI["{model}.feature_importance"]
    M & TR & VA & BP --> CV["{model}.cross_validate"] --> CVM["{model}.cv_metrics"]
    M & TR & VA & BP --> R["{model}.refit_model"] --> FM["{model}.final_model"]
```

| Node | Function | What it does |
|---|---|---|
| `{model}.tune_hyperparameters` | `tune_hyperparameters` | With `n_trials: 0`, returns the base `params`. Otherwise runs an Optuna TPE search (seed 42) over `search_space`, merged on top of `params`. Each trial is fitted with early stopping on valid and scored by `tuning_metric` (`pr_auc`), stopping at `n_trials` or `timeout_minutes`. Each trial is logged as a nested MLflow run |
| `{model}.train_model` | `train_model` | Final fit with early stopping on valid. Computes train/valid ROC-AUC and PR-AUC, `roc_auc_gap`, `best_iteration`, `n_features`, `fit_seconds`, and feature importance (gain / impurity, or permutation importance on valid for models without a built-in one). Logged to a nested MLflow run |
| `{model}.cross_validate` | `cross_validate` | Only with `cv_folds > 0`: expanding-window time-series CV over train + valid (rows ordered by `TransactionID`). Mean ± std of ROC-AUC / PR-AUC go to `{model}_cv_metrics.json` and the model's MLflow run (`cv_*`). With `cv_folds: 0` it returns `{"cv_folds": 0}` immediately |
| `{model}.refit_model` | `refit_model` | With `refit: true`: refits on train + valid with `best_iteration × refit_iteration_multiplier` trees (no early stopping, nothing is left to stop on). Saved as `{model}_final.pkl` and used for the test submission. Logs `refit_rows`, `refit_iterations`, `refit_seconds` to the model's MLflow run |

`FraudModel` (`training/model.py`) wraps all six model types behind one
`fit(X, y, X_valid=None, y_valid=None, n_iterations=None)` / `predict_proba(X)` interface.
Categorical columns are passed natively to LightGBM, XGBoost (`enable_categorical`) and HistGB
(`categorical_features="from_dtype"`); for CatBoost and logistic regression (one-hot) they are
converted to strings, with `__missing__` for NaN; for random forest they become integer codes.
The `imbalance` option (`none` / `class_weight` / `undersample`) is applied inside `fit`, to the
train rows only.

**Parameters** (`parameters_training.yml`, per model):

| Key | LightGBM | XGBoost | CatBoost |
|---|---|---|---|
| `device` | cpu | cuda | GPU |
| `n_estimators` / `early_stopping_rounds` | 5000 / 200 | 5000 / 200 | 5000 / 200 |
| `n_trials` / `timeout_minutes` | 25 / 60 | 25 / 60 | 25 / 60 |
| Tuned | leaves, depth, min child samples, lr, colsample, subsample, L1/L2 | depth, lr, min child weight, subsample, colsample, L1/L2 | depth, lr, L2, random strength |

| Key | Logistic regression | Random forest | HistGB |
|---|---|---|---|
| `imbalance.method` | class_weight | none | none |
| `n_trials` | 0 | 0 | 0 |
| Main params | `C: 0.1`, one-hot `min_frequency: 100` | 300 trees, `min_samples_leaf: 50`, `max_samples: 0.5` | lr 0.05, 63 leaves, up to 2000 iterations, early stopping 50 |

All six share `imbalance`, `cv_folds: 0`, `refit: true` and `refit_iteration_multiplier: 1.2`.

---

## evaluation

**06_models + 05_model_input → 07_model_output + 08_reporting.** For each model, a namespaced
two-node pipeline (like training), plus one node that blends the boosting models.

```mermaid
flowchart LR
    M["{model}.model<br/>(train only)"] & VA[mdl_in_valid] --> E["{model}.evaluate_model"]
    P[/params:evaluation/] --> E
    E --> EJ["{model}.evaluation"]
    E --> EC["{model}.evaluation_curves"]
    FM["{model}.final_model<br/>(train + valid)"] & TE[mdl_in_test] --> S["{model}.predict_submission"] --> SUB["{model}.submission"]
    SUB -->|"lightgbm, xgboost, catboost only"| B[blend_submissions] --> BS[blend.submission]
```

| Node | Function | What it does |
|---|---|---|
| `{model}.evaluate_model` | `evaluate_model` | On valid, with the **train-only** model: ROC-AUC, PR-AUC, fraud rate, plus precision, recall, F1, FPR and TP/FP/FN/TN at three thresholds. **fixed**: `fixed_threshold` (0.5). **best_f1**: the threshold that maximises F1 on valid. **at_fpr**: the threshold at which FPR = `target_fpr` (1%). Draws PR and ROC curves with the two chosen points marked. Writes everything as `eval_*` metrics and `evaluation_curves.png` into the model's **training** MLflow run |
| `{model}.predict_submission` | `predict_submission` | Scores all test rows with the **final** (refitted) model → `TransactionID, isFraud` |
| `blend_submissions` | `blend_submissions` | Rank average (percentile ranks, mean, so the result is in 0–1) of the boosting submissions. Added only when ≥ 2 boosting models are in the pipeline; baselines stay out. Fails if the TransactionIDs don't match |

The training run is found through `FraudModel.mlflow_run_id`. Older pickles don't have it, so for
them the latest `stage = final` run of that model type is used.

**Parameters** (`parameters_evaluation.yml`):

| Key | Value | Note |
|---|---|---|
| `fixed_threshold` | 0.5 | Reference only. Weighted or undersampled models score higher, so 0.5 means something different for them |
| `target_fpr` | 0.01 | "How much fraud do we catch if we bother 1 in 100 legitimate customers?" |

The best-F1 threshold is picked on valid itself, so its numbers are slightly optimistic.

**Still planned:** a cost-based threshold (`Σ missed fraud × TransactionAmt + Σ false alarms ×
review_cost`) and SHAP importance. Registering the model and moving `@champion` is done by the
`serving` pipeline, see [7. Serving API](07_serving_api.md).

---

## Tests

| Test file | Covers |
|---|---|
| `tests/pipelines/test_data_ingestion.py` | Range expansion, column renaming, dtype casting, schema-drift and bad-target errors, the join |
| `tests/pipelines/test_data_cleaning.py` | Time ordering of the split, each filter rule, protected columns, single-row apply, missing-column error |
| `tests/pipelines/test_feature_engineering.py` | Stateless features, native vs coded encodings, **single row = batch**, unseen values |
| `tests/pipelines/test_training.py` | All six models train on CPU and score one row; class weights and undersampling (train only); refit tree count and `refit: false`; expanding CV folds; Optuna merges params; `n_trials=0` path |
| `tests/pipelines/test_evaluation.py` | Threshold metrics, submission format, rank-average blend; the blend uses only boosting models and submissions use `final_model` |
| `tests/api/test_api.py` | Liveness/readiness with and without a model, warm-up, single and batch scoring (order, size limits), threshold precedence, 422/500 error mapping, request-ID and timing headers, prediction log, `/metrics` |

```bash
uv run pytest      # make test
```

---
[← 4. Data layers](04_data_layers.md) · [Next: 6. Modeling approach →](06_modeling_approach.md)
