# 3. Data Description

[← 2. Infrastructure](02_infrastructure.md) · [Table of contents](01_introduction.md#table-of-contents)

Source: Kaggle competition [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection).
Raw files live in `data/01_raw/` (see `_manifest.json` for checksums).

The task is to predict the probability that an online transaction is fraudulent,
as denoted by the binary target `isFraud`.

## Files

| File | Rows | Columns | Notes |
|---|---|---|---|
| `train_transaction.csv` | 590,540 | 394 | Includes target `isFraud` |
| `train_identity.csv` | 144,233 | 41 | |
| `test_transaction.csv` | 506,691 | 393 | No `isFraud` |
| `test_identity.csv` | 141,907 | 41 | Columns named `id-01`… (hyphen) instead of `id_01`… |
| `sample_submission.csv` | 506,691 | 2 | `TransactionID`, `isFraud` |

The data is split into two tables, **transaction** and **identity**, joined by `TransactionID`.
Not all transactions have corresponding identity information (~24% of train transactions do).

> **Gotcha:** `test_identity.csv` uses `id-XX` column names while `train_identity.csv` uses `id_XX`.
> Rename before concatenating train and test.

## Column statistics

Statistics in the tables below are computed on the **train** set (`data/02_intermediate/train.parquet`,
590,540 rows, transaction + identity joined). For a column group (e.g. `C1` – `C14`) the stats are
for the **first column only** (`C1`).

- **Example**: first non-null value
- **Null %**: share of missing values
- **Unique**: number of distinct non-null values (categorical columns)
- **Min / Max / Mean**: over non-null values (numerical columns)

## Transaction table

| Column(s) | Description | Type | Stats for | Example | Null % | Unique | Min | Max | Mean |
|---|---|---|---|---|---|---|---|---|---|
| `TransactionID` | Join key to identity table | ID | `TransactionID` | 2987000 | 0.0 | 590,540 | | | |
| `isFraud` | Target, 0/1 (train only) | Target | `isFraud` | 0 | 0.0 | 2 | 0 | 1 | 0.035 (fraud rate) |
| `TransactionDT` | Timedelta from a given reference datetime (not an actual timestamp) | Numerical | `TransactionDT` | 86400 | 0.0 | | 86,400 | 15,811,131 | 7,372,311 |
| `TransactionAmt` | Transaction payment amount in USD | Numerical | `TransactionAmt` | 68.5 | 0.0 | | 0.251 | 31,937.39 | 135.03 |
| `ProductCD` | Product code, the product for each transaction | **Categorical** | `ProductCD` | W | 0.0 | 5 | | | |
| `card1` – `card6` | Payment card information, such as card type, card category, issue bank, country, etc. | **Categorical** | `card1` | 13926 | 0.0 | 13,553 | | | |
| `addr1`, `addr2` | Address | **Categorical** | `addr1` | 315 | 11.1 | 332 | | | |
| `dist1`, `dist2` | Distance | Numerical | `dist1` | 19 | 59.7 | | 0 | 10,286 | 118.50 |
| `P_emaildomain` | Purchaser email domain | **Categorical** | `P_emaildomain` | gmail.com | 16.0 | 59 | | | |
| `R_emaildomain` | Recipient email domain | **Categorical** | `R_emaildomain` | gmail.com | 76.8 | 60 | | | |
| `C1` – `C14` | Counting, such as how many addresses are found to be associated with the payment card, etc. Actual meaning is masked. | Numerical | `C1` | 1 | 0.0 | | 0 | 4,685 | 14.09 |
| `D1` – `D15` | Timedelta, such as days between previous transaction, etc. | Numerical | `D1` | 14 | 0.2 | | 0 | 640 | 94.35 |
| `M1` – `M9` | Match, such as names on card and address, etc. | **Categorical** | `M1` | T | 45.9 | 2 | | | |
| `V1` – `V339` | Vesta engineered rich features, including ranking, counting, and other entity relations | Numerical | `V1` | 1 | 47.3 | | 0 | 1 | 0.9999 |

## Identity table

Variables in this table are identity information — network connection information
(IP, ISP, Proxy, etc.) and digital signature (UA/browser/os/version, etc.) associated
with transactions. Field names are masked.

Null % here is relative to all train transactions; most of it comes from the ~76% of
transactions that have no identity row at all.

| Column(s) | Description | Type | Stats for | Example | Null % | Unique | Min | Max | Mean |
|---|---|---|---|---|---|---|---|---|---|
| `TransactionID` | Join key to transaction table | ID | — | | | | | | |
| `id_01` – `id_11` | Masked identity / network signals | Numerical | `id_01` | 0 | 75.6 | | -100 | 0 | -10.17 |
| `id_12` – `id_38` | Masked identity / network / device signals | **Categorical** | `id_12` | NotFound | 75.6 | 2 | | | |
| `DeviceType` | Device type (e.g. desktop, mobile) | **Categorical** | `DeviceType` | mobile | 76.2 | 2 | | | |
| `DeviceInfo` | Device / OS string | **Categorical** | `DeviceInfo` | SAMSUNG SM-G892A Build/NRD90M | 79.9 | 1,786 | | | |

## Summary: categorical vs numerical

**Categorical — transaction**
- `ProductCD`
- `card1` – `card6`
- `addr1`, `addr2`
- `P_emaildomain`, `R_emaildomain`
- `M1` – `M9`

**Categorical — identity**
- `DeviceType`
- `DeviceInfo`
- `id_12` – `id_38`

**Numerical** — everything else:
- Transaction: `TransactionDT`, `TransactionAmt`, `dist1`, `dist2`, `C1` – `C14`, `D1` – `D15`, `V1` – `V339`
- Identity: `id_01` – `id_11`

Note: some categorical columns are stored as numbers in the CSVs (e.g. `card1`, `card2`, `card3`,
`card5`, `addr1`, `addr2`, and several `id_XX`). Pandas will read them as numeric, so cast them
explicitly — their values are codes, not quantities. The `data_ingestion` pipeline does this from
the schema in `conf/base/parameters_data_ingestion.yml`, and turns `315.0` into the string
`"315"` (`fraud_detection.keys.to_key`) so train, test and API values match.

## Data challenges

| Challenge | Evidence | How the project handles it |
|---|---|---|
| Class imbalance | `isFraud` mean 0.035 (train split 0.0351, valid split 0.0344) | Ranking metrics (PR-AUC, ROC-AUC), no resampling, cost-based threshold (planned) |
| Drift over time | `TransactionDT` covers ~182 days; test follows train. `D1`–`D15` grow with time | Time-based split; drop `TransactionDT`; `D_n_norm = day − D_n` |
| Many nulls | 214 of 435 columns are more than half null; identity columns ≥76% null; `V` columns up to 86% null | NaN kept as is (trees route it), `has_identity` flag, drop only columns >99% null |
| Redundant columns | The 339 `V` columns fall into 15 blocks with identical null shares, often highly correlated within a block | Per-block correlation filter (\|r\| > 0.95) → 56 `V` columns dropped |
| High-cardinality codes | `card1` 13,553 values, `DeviceInfo` 1,786 | Frequency encoding, integer codes (>100 categories), group statistics |

Details: [6. Modeling approach](06_modeling_approach.md).

---
[← 2. Infrastructure](02_infrastructure.md) · [Next: 4. Data layers and catalog →](04_data_layers.md)

## Types in `02_intermediate`

The `data_ingestion` pipeline types each raw table against the schema in
`conf/base/parameters_data_ingestion.yml`, then joins them. Resulting dtypes:

| Column(s) | dtype | Notes |
|---|---|---|
| `TransactionID` | `int64` | Checked non-null and unique |
| `isFraud` | `int8` | Checked 0/1. Optional: absent in test, the same pipeline still runs |
| Categorical columns | `category` | Numeric codes become strings without `.0` (`315.0` → `"315"`) |
| Numerical columns | `float32` | Default, to halve memory |
| `TransactionDT` | `int64` | Override: test values reach ~34M, beyond float32's exact-integer range |
| `TransactionAmt` | `float64` | Override: keeps cents exact |
| `has_identity` | `int8` | Added by the join: 1 if the transaction has an identity row |

A column in the file that is missing from the schema, or vice versa, fails the run.

The stats tables above were computed before this typing, so `card1` and `addr1` show numeric
examples; in `02_intermediate` they are categorical strings.
