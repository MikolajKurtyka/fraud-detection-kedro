"""One wrapper for all model types with a shared fit / predict interface.

Gradient boosting: LightGBM, XGBoost, CatBoost. Baselines: logistic regression, random
forest and scikit-learn's HistGradientBoosting. All get the same model input (``category``
columns with categories fixed at fit time). The per-library differences -- how those
columns are passed in, early stopping, class weights -- are hidden here, so training,
evaluation and the API all call ``predict_proba(df)``.
"""

import logging

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

BOOSTING_TYPES = ("lightgbm", "xgboost", "catboost")
BASELINE_TYPES = ("logreg", "random_forest", "hist_gb")
MODEL_TYPES = BOOSTING_TYPES + BASELINE_TYPES
# Models that add trees one by one: early stopping on valid, a fixed tree count without it
ITERATIVE_TYPES = (*BOOSTING_TYPES, "hist_gb")
IMBALANCE_METHODS = ("none", "class_weight", "undersample")
_MISSING_CATEGORY = "__missing__"


class _LightGBMLogger:
    """Routes LightGBM's training output to the logging module instead of stdout."""

    def info(self, msg: str) -> None:
        logger.info(msg)

    def warning(self, msg: str) -> None:
        logger.warning(msg)


lgb.register_logger(_LightGBMLogger())


def undersample(
    X: pd.DataFrame, y: pd.Series, ratio: float, seed: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    """Keep every positive and a random ``ratio`` negatives per positive, in original order."""
    labels = y.to_numpy()
    positives, negatives = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    n_negatives = min(len(negatives), round(len(positives) * ratio))
    kept_negatives = np.random.default_rng(seed).choice(negatives, n_negatives, replace=False)
    keep = np.sort(np.concatenate([positives, kept_negatives]))
    return X.iloc[keep], y.iloc[keep]


class FraudModel:
    """A fitted model plus everything needed to score new rows."""

    def __init__(self, model_type: str, params: dict, options: dict):
        if model_type not in MODEL_TYPES:
            raise ValueError(f"Unknown model_type {model_type!r}, expected one of {MODEL_TYPES}")
        imbalance = {"method": "none", **options.get("imbalance", {})}
        if imbalance["method"] not in IMBALANCE_METHODS:
            raise ValueError(
                f"Unknown imbalance method {imbalance['method']!r}, "
                f"expected one of {IMBALANCE_METHODS}"
            )
        self.model_type = model_type
        self.params = params
        self.options = options
        self.imbalance = imbalance
        self.feature_columns: list[str] = []
        self.categorical_columns: list[str] = []
        self.estimator = None
        self.best_iteration: int | None = None  # trees used; None for non-iterative models
        self.mlflow_run_id: str | None = None  # training run, so evaluation can log next to it

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[self.feature_columns]
        if self.model_type in ("catboost", "logreg"):
            # CatBoost and the one-hot encoder want categorical columns as strings without nulls.
            X = X.copy()
            for col in self.categorical_columns:
                X[col] = X[col].astype(object).fillna(_MISSING_CATEGORY).astype(str)
            if self.model_type == "logreg":
                numeric = [c for c in self.feature_columns if c not in self.categorical_columns]
                X[numeric] = X[numeric].astype("float32")  # halves the dense one-hot matrix
        elif self.model_type == "random_forest":
            # Random forest has no categorical support: category codes, NaN for missing.
            X = X.copy()
            for col in self.categorical_columns:
                codes = X[col].cat.codes
                X[col] = codes.where(codes >= 0).astype("float32")
        return X

    def _logreg_pipeline(self, params: dict):
        """Impute + scale numbers, one-hot categories (rare ones grouped), logistic regression."""
        numeric = [c for c in self.feature_columns if c not in self.categorical_columns]
        preprocess = ColumnTransformer(
            [
                (
                    "num",
                    make_pipeline(
                        SimpleImputer(strategy="median", add_indicator=True), StandardScaler()
                    ),
                    numeric,
                ),
                (
                    "cat",
                    OneHotEncoder(
                        handle_unknown="infrequent_if_exist",
                        min_frequency=self.options.get("one_hot_min_frequency", 100),
                        sparse_output=False,
                        dtype=np.float32,
                    ),
                    self.categorical_columns,
                ),
            ]
        )
        return make_pipeline(preprocess, LogisticRegression(**params))

    def _build_estimator(self, y: pd.Series, n_estimators: int | None, early_stopping: bool):
        opts, params = self.options, dict(self.params)
        weighted = self.imbalance["method"] == "class_weight"
        stopping_rounds = opts.get("early_stopping_rounds") if early_stopping else None

        if self.model_type == "lightgbm":
            if weighted:
                params["class_weight"] = "balanced"
            return lgb.LGBMClassifier(
                n_estimators=n_estimators,
                device_type=opts["device"],
                importance_type="gain",
                **params,
            )
        if self.model_type == "xgboost":
            if weighted:
                params["scale_pos_weight"] = float((y == 0).sum() / max((y == 1).sum(), 1))
            return xgb.XGBClassifier(
                n_estimators=n_estimators,
                early_stopping_rounds=stopping_rounds,
                eval_metric="auc",
                enable_categorical=True,
                device=opts["device"],
                **params,
            )
        if self.model_type == "catboost":
            if weighted:
                params["auto_class_weights"] = "Balanced"
            return cb.CatBoostClassifier(
                iterations=n_estimators,
                early_stopping_rounds=stopping_rounds,
                eval_metric="AUC",
                task_type=opts["device"],
                cat_features=self.categorical_columns,
                verbose=opts["log_every"],
                allow_writing_files=False,
                **params,
            )

        # scikit-learn models share the class_weight argument
        if weighted:
            params["class_weight"] = "balanced"
        if self.model_type == "logreg":
            return self._logreg_pipeline(params)
        if self.model_type == "random_forest":
            return RandomForestClassifier(**params)
        return HistGradientBoostingClassifier(
            max_iter=n_estimators,
            early_stopping=early_stopping,
            n_iter_no_change=opts.get("early_stopping_rounds", 10),
            scoring="roc_auc",
            categorical_features="from_dtype",
            **params,
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
        n_iterations: int | None = None,
    ) -> "FraudModel":
        """Fit on train; the imbalance method is applied to train only, never to valid.

        With a valid set, iterative models early-stop on it. Without one (the refit on all
        labelled data) they build exactly ``n_iterations`` trees.
        """
        early_stopping = X_valid is not None
        if not early_stopping and n_iterations is None and self.model_type in ITERATIVE_TYPES:
            raise ValueError(f"{self.model_type} without a valid set needs n_iterations")

        self.feature_columns = list(X.columns)
        self.categorical_columns = [
            c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)
        ]
        if self.imbalance["method"] == "undersample":
            n_rows = len(X)
            X, y = undersample(X, y, self.imbalance["undersample_ratio"])
            logger.info(
                "%s: undersampled train from %d to %d rows (%d negatives per positive)",
                self.model_type,
                n_rows,
                len(X),
                self.imbalance["undersample_ratio"],
            )

        n_estimators = n_iterations or self.options.get("n_estimators")
        self.estimator = self._build_estimator(y, n_estimators, early_stopping)
        X_fit = self._prepare(X)
        X_val = self._prepare(X_valid) if early_stopping else None
        log_every = self.options.get("log_every", 0)

        if self.model_type == "lightgbm":
            eval_kwargs = {}
            if early_stopping:
                eval_kwargs = {
                    "eval_X": (X_val,),
                    "eval_y": (y_valid,),
                    "eval_metric": "auc",
                    "callbacks": [
                        lgb.early_stopping(self.options["early_stopping_rounds"], verbose=False),
                        lgb.log_evaluation(log_every),
                    ],
                }
            self.estimator.fit(X_fit, y, **eval_kwargs)
            self.best_iteration = (
                int(self.estimator.best_iteration_) if early_stopping else n_estimators
            )
        elif self.model_type == "xgboost":
            eval_set = [(X_val, y_valid)] if early_stopping else None
            self.estimator.fit(X_fit, y, eval_set=eval_set, verbose=log_every)
            self.best_iteration = (
                int(self.estimator.best_iteration) + 1 if early_stopping else n_estimators
            )
        elif self.model_type == "catboost":
            if early_stopping:
                self.estimator.fit(X_fit, y, eval_set=(X_val, y_valid), use_best_model=True)
                self.best_iteration = int(self.estimator.get_best_iteration()) + 1
            else:
                self.estimator.fit(X_fit, y)
                self.best_iteration = n_estimators
        elif self.model_type == "hist_gb":
            if early_stopping:
                self.estimator.fit(X_fit, y, X_val=X_val, y_val=y_valid)
            else:
                self.estimator.fit(X_fit, y)
            n_iter = int(self.estimator.n_iter_)
            # HistGradientBoosting keeps the trees built while waiting for an improvement
            if early_stopping and n_iter < n_estimators:
                n_iter -= self.estimator.n_iter_no_change
            self.best_iteration = n_iter
        else:
            self.estimator.fit(X_fit, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Fraud probability per row. Works for any number of rows, including one."""
        X = self._prepare(X)
        if self.model_type == "xgboost":
            # Score on the CPU: the input lives in host memory, and serving has no GPU.
            self.estimator.set_params(device="cpu")
        return self.estimator.predict_proba(X)[:, 1]

    def feature_importance(self) -> pd.DataFrame | None:
        """Built-in importance (gain / impurity); None for models without one."""
        importances = getattr(self.estimator, "feature_importances_", None)
        if importances is None:
            return None
        return (
            pd.DataFrame({"feature": self.feature_columns, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
