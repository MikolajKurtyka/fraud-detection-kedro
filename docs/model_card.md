# Model Card: Fraud Detection (IEEE-CIS)

*Template: sections marked TBD are completed once training and evaluation are implemented.*

## Model Details

- **Task**: binary classification, probability that a card-not-present transaction is fraudulent
- **Algorithm**: LightGBM gradient boosting, hyperparameters tuned with Optuna
- **Version**: tracked in the MLflow registry as `fraud-detection`; the served version carries the `champion` alias
- **Owner**: Mikolaj Kurtyka

## Intended Use

- **Primary use**: real-time risk score used to route transactions to approve / manual review / decline
- **Out of scope**: credit decisions, judging individual customers' intent, data from other payment providers without retraining

## Training Data

- IEEE-CIS Fraud Detection (Vesta), ~590k labelled e-commerce transactions over about 6 months
- Positive rate ~3.5%; identity information is present for ~24% of transactions
- Most columns are anonymised (`V*`, `C*`, `D*`, `M*`, `id_*`), which limits domain interpretation

## Evaluation

- **Split**: time-based (train on earlier `TransactionDT`, validate on later). A random split would leak future fraud patterns.
- **Metrics**: PR-AUC (main), ROC-AUC, recall at a fixed false-positive rate, expected cost per 1k transactions
- **Threshold**: chosen to minimise `FN × TransactionAmt + FP × review_cost`
- **Results**: TBD

## Explainability

- Global: SHAP feature importance, logged to MLflow per run
- Local: SHAP values per prediction (planned API option)

## Limitations and Risks

- Fraud patterns drift; the model needs monitoring and periodic retraining
- The client UID is an approximation built from card and address fields; mistakes in it can merge different clients
- Anonymised features make it hard to rule out proxy variables for protected attributes

## Fairness Considerations

- No protected attributes are available in the data, so a direct fairness audit isn't possible
- Planned: compare error rates across available segments (device type, email domain, product code) to detect uneven false-alarm rates
