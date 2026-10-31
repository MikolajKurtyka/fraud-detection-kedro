# Data Folders

`data/` follows Kedro's layered convention. Data flows down through the numbered
folders, getting cleaner and closer to model-ready at each step.

| Folder | Purpose | Contents |
|---|---|---|
| `01_raw` | Original files exactly as downloaded. Never edited. | The 5 Kaggle CSVs and `_manifest.json` |
| `02_intermediate` | Raw data with fixed types and file formats, plus simple joins. No business logic yet. | `train.parquet`, `test.parquet`: transaction + identity joined on `TransactionID` (written by the `data_ingestion` pipeline) |
| `03_primary` | Clean, validated data: one trusted table per thing | — |
| `04_feature` | Engineered features, such as aggregates and encodings | — |
| `05_model_input` | The final train/validation matrices given to the model | — |
| `06_models` | Trained model files, saved as pickles | — |
| `07_model_output` | Predictions and scores | — |
| `08_reporting` | Plots, metrics and reports | — |

`01_raw/_manifest.json` records the source competition (IEEE-CIS Fraud Detection), the download
time and a SHA256 checksum for each file.

Dataset paths for each layer are defined in `conf/base/catalog.yml`. Column-level details of
the raw files are in [data_description.md](data_description.md).
