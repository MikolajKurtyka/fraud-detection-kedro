# Data Description

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

## Transaction table

| Column(s) | Description | Type |
|---|---|---|
| `TransactionID` | Join key to identity table | ID |
| `isFraud` | Target, 0/1 (train only) | Target |
| `TransactionDT` | Timedelta from a given reference datetime (not an actual timestamp) | Numerical |
| `TransactionAmt` | Transaction payment amount in USD | Numerical |
| `ProductCD` | Product code, the product for each transaction | **Categorical** |
| `card1` – `card6` | Payment card information, such as card type, card category, issue bank, country, etc. | **Categorical** |
| `addr1`, `addr2` | Address | **Categorical** |
| `dist1`, `dist2` | Distance | Numerical |
| `P_emaildomain` | Purchaser email domain | **Categorical** |
| `R_emaildomain` | Recipient email domain | **Categorical** |
| `C1` – `C14` | Counting, such as how many addresses are found to be associated with the payment card, etc. Actual meaning is masked. | Numerical |
| `D1` – `D15` | Timedelta, such as days between previous transaction, etc. | Numerical |
| `M1` – `M9` | Match, such as names on card and address, etc. | **Categorical** |
| `V1` – `V339` | Vesta engineered rich features, including ranking, counting, and other entity relations | Numerical |

## Identity table

Variables in this table are identity information — network connection information
(IP, ISP, Proxy, etc.) and digital signature (UA/browser/os/version, etc.) associated
with transactions. Field names are masked.

| Column(s) | Description | Type |
|---|---|---|
| `TransactionID` | Join key to transaction table | ID |
| `id_01` – `id_11` | Masked identity / network signals | Numerical |
| `id_12` – `id_38` | Masked identity / network / device signals | **Categorical** |
| `DeviceType` | Device type (e.g. desktop, mobile) | **Categorical** |
| `DeviceInfo` | Device / OS string | **Categorical** |

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
explicitly — their values are codes, not quantities.
