# 8. Model Card: Fraud Detection (IEEE-CIS)

[← 7. Serving API](07_serving_api.md) · [Table of contents](01_introduction.md#table-of-contents)

## Model Details

- **Task**: binary classification, the probability that a card-not-present transaction is fraudulent
- **Algorithms**: LightGBM, XGBoost and CatBoost, plus logistic regression, random forest and scikit-learn HistGradientBoosting as baselines, all trained on the same features and compared. Hyperparameter tuning with Optuna (PR-AUC) is built in but hasn't been run yet. LightGBM is currently the best
- **Input**: raw transaction + identity columns, turned into 395 features by `InferenceBundle` (see [5. Pipelines](05_pipelines.md#feature_engineering))
- **Variants**: research `{model}.pkl` (train only, the one with validated metrics) and `{model}_final.pkl` (refitted on train + valid). The **production** model (`production.pkl`, `production_final.pkl`) is one model type with its research params on its top-N features (`conf/base/parameters_production.yml`); the train-only one is served, the refit one writes `submission_production.csv`
- **Version**: tracked in the MLflow registry as `fraud-detection`; the served version carries the `champion` alias and every API response names it (`model_version`). The registered artifact is the full inference bundle (column filter + feature encoders + LightGBM), with its threshold, cost per 1k, holdout PR-AUC and training run ID as tags (see [7. Serving API](07_serving_api.md#how-a-model-reaches-the-api))
- **Owner**: Mikolaj Kurtyka

## Intended Use

- **Primary use**: a real-time risk score that routes transactions to approve / manual review / decline
- **Out of scope**: credit decisions, judging individual customers' intent, data from other payment providers without retraining

## Training Data

- IEEE-CIS Fraud Detection (Vesta): 590,540 labelled e-commerce transactions over ~182 days
- Positive rate ~3.5%; identity information is present for ~24% of transactions
- Most columns are anonymised (`V*`, `C*`, `D*`, `M*`, `id_*`), which limits domain interpretation
- Train split: days 1–140 (472,006 rows, 3.51% fraud). Validation split: days 141–182 (118,534 rows, 3.44% fraud)

## Evaluation

- **Split**: time-based (train on earlier days, validate on later days). A random split would leak future fraud patterns. The production model also keeps the later half of valid as a holdout for its reported metrics and the promotion
- **Metrics**: PR-AUC (main), ROC-AUC, train–valid gap; precision / recall / F1 / FPR and expected cost per 1k transactions at 0.5, at the best-F1 threshold, at 1% FPR and at the cost-optimal threshold; PR, ROC and cost curves
- **Threshold**: the one with the lowest expected cost on valid: missed fraud costs its `TransactionAmt`, every flagged transaction $5 (review), every flagged good customer $15 more. For LightGBM: 0.077, $2,764 per 1k transactions vs $5,156 with no model. The costs are assumptions to agree with the business. The threshold is stored with the registered model; `FRAUD_THRESHOLD` overrides it

**Results**: base parameters (Optuna tuning not run yet), validation on days 141–182:

| Model | ROC-AUC | PR-AUC | Recall @ 1% FPR | Best F1 (threshold) | Train ROC-AUC | Gap |
|---|---|---|---|---|---|---|
| LightGBM | 0.9315 | 0.6137 | 0.537 | 0.596 (0.36) | 0.9991 | 0.068 |
| XGBoost | 0.9292 | 0.6117 | 0.531 | 0.589 (0.26) | 0.9962 | 0.067 |
| HistGB (baseline) | 0.9197 | 0.5910 | 0.524 | 0.583 (0.33) | 0.9917 | 0.072 |
| CatBoost | 0.9056 | 0.5449 | 0.467 | 0.548 (0.37) | 0.9708 | 0.065 |
| Random forest (baseline) | 0.8884 | 0.4482 | 0.387 | 0.465 (0.25) | 0.9520 | 0.064 |
| Logistic regression (baseline) | 0.8361 | 0.1958 | 0.017 | 0.340 (0.83) | 0.8895 | 0.053 |

In plain terms: with LightGBM, flagging 1 in 100 legitimate transactions catches about 54% of
fraud. At its best-F1 threshold, 73% of flagged transactions are fraud and 50% of fraud is caught.

These scores are slightly optimistic: early stopping picked the number of trees on the same
validation split, and the best-F1 and min-cost thresholds were chosen on it too (see
[6. Modeling approach](06_modeling_approach.md#known-limitations)). The production model's
`holdout_*` metrics (`data/08_reporting/production_evaluation.json`) are the honest numbers. Kaggle submissions
(`data/07_model_output/`) have been written but not uploaded, so there's no test-set score yet.

## Explainability

- Built-in: feature importance per model (`data/08_reporting/{model}_feature_importance.csv`, also logged to MLflow): total gain for LightGBM and XGBoost, prediction-value change for CatBoost, impurity for random forest, permutation on valid for logistic regression and HistGB
- Monitoring: the API logs every prediction as JSON and exposes a probability histogram on `/metrics`, both inputs for drift checks
- Global: mean |SHAP| per feature on 5,000 valid rows for each boosting model (`data/08_reporting/{model}_shap_importance.csv`, also in MLflow). LightGBM's top features: C13, D1_norm, card1_freq, card6, D4_norm, TransactionAmt
- Local: `POST /explain` returns the features that pushed one transaction's score up or down most (SHAP values in log-odds)

## Limitations and Risks

- Fraud patterns drift, so the model needs monitoring and periodic retraining (weekly workflow in place; feature drift monitoring is not)
- The decision threshold rests on assumed costs ($5 review, $15 false block); with $50 per false block the optimum moves from 0.077 to ~0.28
- The train–valid gap (~0.07 ROC-AUC) suggests overfitting and/or remaining drift
- The client UID is an approximation built from card and address fields; mistakes in it can merge different clients, and clients not seen in train get no aggregate features
- Anonymised features make it hard to rule out proxy variables for protected attributes

## Fairness Considerations

- No protected attributes are available in the data, so a direct fairness audit isn't possible
- Planned: compare error rates across available segments (device type, email domain, product code) to detect uneven false-alarm rates

---
[← 7. Serving API](07_serving_api.md) · [Back to 1. Introduction](01_introduction.md)
