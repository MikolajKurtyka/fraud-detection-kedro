"""Print and save the history of trained models from MLflow, newest first.

Reads every final model run (tag ``stage=final``) of the ``fraud-detection`` experiment and
writes a comparison table to ``data/08_reporting/model_comparison.{csv,md}``. Each row is one
model from one ``kedro run``, so the table is the full history of training runs.

Usage:
    uv run python scripts/compare_runs.py               # all final runs
    uv run python scripts/compare_runs.py --last 1      # only the latest kedro run
    uv run python scripts/compare_runs.py --tracking-uri http://localhost:5000
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

import mlflow
import pandas as pd

EXPERIMENT = "fraud-detection"
REPORT_DIR = Path(__file__).resolve().parents[1] / "data" / "08_reporting"
DEFAULT_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

COLUMNS = {
    "start_time": "Started",
    "tags.model_type": "Model",
    "params.device": "Device",
    "params.imbalance": "Imbalance",
    "metrics.valid_roc_auc": "Valid ROC-AUC",
    "metrics.valid_pr_auc": "Valid PR-AUC",
    "metrics.train_roc_auc": "Train ROC-AUC",
    "metrics.roc_auc_gap": "Gap",
    # time-series CV (empty unless cv_folds > 0)
    "metrics.cv_roc_auc_mean": "CV ROC-AUC",
    "metrics.cv_roc_auc_std": "CV ROC-AUC std",
    "metrics.cv_pr_auc_mean": "CV PR-AUC",
    "metrics.cv_pr_auc_std": "CV PR-AUC std",
    # from the evaluation pipeline (empty until `kedro run --pipelines evaluation` has run)
    "metrics.eval_fixed_precision": "Precision @0.5",
    "metrics.eval_fixed_recall": "Recall @0.5",
    "metrics.eval_best_f1_f1": "Best F1",
    "metrics.eval_best_f1_precision": "Precision @best F1",
    "metrics.eval_best_f1_recall": "Recall @best F1",
    "metrics.eval_at_fpr_recall": "Recall @1% FPR",
    "tuned": "Tuned",
    "metrics.best_iteration": "Best iteration",
    "metrics.n_features": "Features",
    "metrics.fit_seconds": "Fit (s)",
    "metrics.refit_rows": "Refit rows",
    "metrics.refit_iterations": "Refit iterations",
    "tags.mlflow.parentRunId": "Parent run",
    "run_id": "Run",
}


def load_runs(tracking_uri: str, last: int | None) -> pd.DataFrame:
    mlflow.set_tracking_uri(tracking_uri)
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT],
        filter_string="tags.stage = 'final'",
        order_by=["start_time DESC"],
    )
    if runs.empty:
        return runs
    if last is not None:
        parents = runs["tags.mlflow.parentRunId"].drop_duplicates().head(last)
        runs = runs[runs["tags.mlflow.parentRunId"].isin(parents)]

    # A model is tuned when the same kedro run also has a "<model>-tuning" run
    tuning = mlflow.search_runs(
        experiment_names=[EXPERIMENT], filter_string="tags.stage = 'tuning'"
    )
    tuned_keys = (
        set(zip(tuning["tags.mlflow.parentRunId"], tuning["tags.model_type"], strict=True))
        if not tuning.empty
        else set()
    )
    runs = runs.assign(
        tuned=[
            "yes" if key in tuned_keys else "no"
            for key in zip(runs["tags.mlflow.parentRunId"], runs["tags.model_type"], strict=True)
        ]
    )

    table = runs.reindex(columns=list(COLUMNS)).rename(columns=COLUMNS)
    table["Started"] = (
        pd.to_datetime(table["Started"])
        .dt.tz_convert(datetime.now().astimezone().tzinfo)  # local time
        .dt.strftime("%Y-%m-%d %H:%M")
    )
    table["Parent run"] = table["Parent run"].str[:8]
    table["Run"] = table["Run"].str[:8]
    for col in ("Best iteration", "Features", "Refit rows", "Refit iterations"):
        table[col] = table[col].astype("Int64")
    return table.round(4)


def to_markdown(table: pd.DataFrame) -> str:
    cells = table.astype(object).where(table.notna(), "").astype(str)
    lines = ["| " + " | ".join(cells.columns) + " |", "|" + "---|" * len(cells.columns)]
    lines += ["| " + " | ".join(row) + " |" for row in cells.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tracking-uri", default=DEFAULT_URI)
    parser.add_argument("--last", type=int, help="only the N most recent kedro runs")
    args = parser.parse_args()

    table = load_runs(args.tracking_uri, args.last)
    if table.empty:
        print(f"No final model runs in experiment {EXPERIMENT!r} at {args.tracking_uri}")
        return

    print(table.to_string(index=False))
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    (REPORT_DIR / "model_comparison.md").write_text(to_markdown(table), encoding="utf-8")
    print(f"\nSaved {len(table)} runs to {REPORT_DIR / 'model_comparison.csv'} and .md")


if __name__ == "__main__":
    main()
