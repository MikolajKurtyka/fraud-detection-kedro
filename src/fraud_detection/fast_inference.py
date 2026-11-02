"""Fast scoring path: raw rows as dicts -> fraud probability, on NumPy arrays instead of pandas.

The pandas path (``InferenceBundle.predict_proba``) chains the batch node functions. Each pandas
call costs ~0.1-1 ms whatever the row count, and there are thousands of them, so one row takes
~200 ms. ``RecordScorer`` computes the same features column by column on NumPy arrays, with the
dtypes the nodes use at every step, so the scores are equal (``tests/test_inference.py`` checks
it). It takes one row or a whole batch, and computes only the features the model reads.

Kept identical to the pandas path on purpose:
- numerical columns: one float64 cast of every kept column (so every value is validated, as in
  ``type_raw``), then one cast per storage dtype (float32 / int64 / float64);
- categorical keys: ``to_key`` applied per value (integral numbers without ".0", strings as-is);
- derived features: same formulas and dtypes as ``add_stateless_features`` and
  ``apply_feature_encoders`` (e.g. float32 ``D_n_norm``, float64 ratios cast to float32).
"""

import math
from functools import cached_property

import numpy as np
import pandas as pd

from fraud_detection.pipelines.feature_engineering.nodes import _SECONDS_PER_DAY


def _is_missing(value) -> bool:
    return value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value))


def _key(value) -> str | None:
    """``to_key`` for one value: None for nulls, "315" for 315.0, strings unchanged."""
    if _is_missing(value):
        return None
    if isinstance(value, str):
        return value
    number = float(value)
    if math.isnan(number):
        return None
    return str(int(number)) if number.is_integer() else str(number)


def _normalize_names(record: dict) -> dict:
    """``normalize_identity_columns`` for one row: ``id-01`` -> ``id_01``."""
    if "-" in " ".join(record):  # one C-level scan instead of a loop over ~400 names
        return {name.replace("-", "_"): value for name, value in record.items()}
    return record


class RecordScorer:
    """Compiled from an ``InferenceBundle`` once; ``predict(records)`` per request."""

    def __init__(self, bundle):
        columns = bundle._columns
        params = bundle.feature_params
        encoders = bundle.feature_encoders
        self.model = bundle.model
        self.features = list(bundle.model.feature_columns)
        self.key = bundle.ingestion_params["join_key"]
        self.params = params

        # Numerical raw columns, grouped by storage dtype for one cast per group
        self.numerical = list(columns["numerical"])
        self.dtype_groups: dict[str, list[int]] = {}
        self.position: dict[str, tuple[str, int]] = {}
        for i, (col, dtype) in enumerate(columns["numerical"].items()):
            group = self.dtype_groups.setdefault(dtype, [])
            self.position[col] = (dtype, len(group))
            group.append(i)
        self.categorical = set(columns["categorical"])
        self.identity = columns["identity"]

        # Fitted lookup tables, by the feature name they produce
        self.group_stats = {spec["name"]: spec for spec in encoders["group_stats"]}
        self.frequency = {f"{col}_freq": (col, m) for col, m in encoders["frequency"].items()}
        self.coded = {
            col: enc["codes"]
            for col, enc in encoders["categorical"].items()
            if enc["encoding"] == "code"
        }
        self.native = {
            col: list(enc["categories"])
            for col, enc in encoders["categorical"].items()
            if enc["encoding"] == "native"
        }
        if self.model.model_type == "lightgbm":
            # LightGBM codes a pandas category by its position in the categories seen at fit
            # time (pandas_categorical), whatever the input's own categories are.
            fitted = self.model.estimator.booster_.pandas_categorical or []
            self.native.update(zip(self.model.categorical_columns, fitted, strict=True))
        self.native_codes = {
            col: {value: code for code, value in enumerate(categories)}
            for col, categories in self.native.items()
        }

        self.d_norm = {f"{col}_norm": col for col in params["d_columns"]}
        self.email_parts = {
            f"{col}_{i + 1}": (col, i) for col in params["email_columns"] for i in range(3)
        }

    def feature_matrix(self, records: list[dict]) -> np.ndarray:
        """Model input as float64 (rows x model features); native categories as codes, NaN if
        missing or unseen -- the values LightGBM gets from the pandas model input."""
        batch = _Batch(self, records)
        X = np.empty((len(records), len(self.features)), order="F")  # filled column by column
        for j, name in enumerate(self.features):
            X[:, j] = batch.feature(name)
        return X

    def _frame(self, X: np.ndarray) -> pd.DataFrame:
        """The DataFrame the other libraries were fitted on, with category dtypes."""
        frame = pd.DataFrame(X, columns=self.features)
        for col in self.model.categorical_columns:
            codes = np.nan_to_num(frame[col].to_numpy(), nan=-1).astype("int64")
            frame[col] = pd.Categorical.from_codes(codes, categories=self.native[col])
        return frame

    def predict(self, records: list[dict]) -> np.ndarray:
        X = self.feature_matrix(records)
        if self.model.model_type == "lightgbm":
            return self.model.estimator.booster_.predict(X)
        return self.model.predict_proba(self._frame(X))

    @cached_property
    def _code_labels(self) -> dict[str, dict[int, str]]:
        return {
            col: {code: key for key, code in codes.items()} for col, codes in self.coded.items()
        }

    def label(self, name: str, value: float) -> float | str | None:
        """A model input value as a person reads it: category codes back to the category."""
        if math.isnan(value):
            return None
        if name in self.native:
            return str(self.native[name][int(value)])
        if name in self.coded:
            return self._code_labels[name].get(int(value))
        return float(value)

    def shap_values(self, records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        """SHAP values (rows x model features, log-odds) and the model input they explain.

        Only on request: TreeSHAP costs a few ms per row, far more than the prediction.
        """
        X = self.feature_matrix(records)
        if self.model.model_type == "lightgbm":
            values = self.model.estimator.booster_.predict(X, pred_contrib=True)
        else:
            values = self.model.shap_values(self._frame(X))
        return values[:, :-1], X


class _Batch:
    """The columns of one request, each computed on first use."""

    def __init__(self, scorer: RecordScorer, records: list[dict]):
        if not records:
            raise ValueError("No rows to score")
        self.s = scorer
        self.records = [_normalize_names(r) for r in records]
        self.cache: dict[str, np.ndarray | list] = {}

        key = scorer.key
        ids = [r.get(key) for r in self.records]
        if any(_is_missing(i) for i in ids) or len(set(ids)) < len(ids):
            raise ValueError(f"{key} must be non-null and unique")
        np.asarray(ids).astype("int64")

        # One cast for all numerical columns (raises on non-numeric strings, like type_raw)
        columns = scorer.numerical
        values = np.array([list(map(r.get, columns)) for r in self.records], dtype=np.float64)
        self.typed = {}
        for dtype, idx in scorer.dtype_groups.items():
            block = values[:, idx]
            if np.dtype(dtype).kind in "iu" and not np.isfinite(block).all():
                raise ValueError("Cannot convert non-finite values (NA or inf) to integer")
            self.typed[dtype] = block.astype(dtype)

    def keys(self, name: str) -> list[str | None]:
        """``to_key`` of a column: categorical columns are keys already, numbers get formatted."""
        column = self.base(name)
        if isinstance(column, list):
            return column
        return [_key(float(v)) for v in column]

    def base(self, name: str) -> np.ndarray | list:
        """A column after ``add_stateless_features``: arrays for numbers, lists for keys."""
        if name not in self.cache:
            self.cache[name] = self._compute_base(name)
        return self.cache[name]

    def _compute_base(self, name: str) -> np.ndarray | list:
        s, records = self.s, self.records
        if name in s.position:
            dtype, j = s.position[name]
            return self.typed[dtype][:, j]
        if name in s.categorical:
            return [_key(r.get(name)) for r in records]
        if name == "has_identity":
            if any("has_identity" in r for r in records):
                flags = [r.get("has_identity") for r in records]
                return np.array([0 if _is_missing(f) else f for f in flags]).astype("int8")
            return np.array(
                [any(not _is_missing(r.get(c)) for c in s.identity) for r in records]
            ).astype("int8")

        seconds = self.base(s.params["time_column"])
        if name == "day":
            return seconds // _SECONDS_PER_DAY
        if name == "hour":
            return ((seconds // 3600) % 24).astype("int8")
        if name == "weekday":
            return (self.base("day") % 7).astype("int8")
        if name == "TransactionAmt_log":
            return np.log1p(self.base("TransactionAmt")).astype("float32")
        if name == "TransactionAmt_cents":
            amount = self.base("TransactionAmt")
            return np.round(amount - np.floor(amount), 3).astype("float32")
        if name in s.d_norm:
            return (self.base("day") - self.base(s.d_norm[name])).astype("float32")
        if name in s.email_parts:
            col, i = s.email_parts[name]
            parts = [None if k is None else k.split(".", 2) for k in self.keys(col)]
            return [p[i] if p is not None and i < len(p) else None for p in parts]
        if name == "uid":
            columns = [self.keys(c) for c in s.params["uid_columns"]]
            rows = zip(*columns, strict=True)
            return ["_".join("nan" if k is None else k for k in row) for row in rows]
        raise ValueError(f"Missing feature columns: [{name!r}]")

    def feature(self, name: str) -> np.ndarray:
        """A model feature after ``apply_feature_encoders``."""
        s = self.s
        if name in s.group_stats:
            spec = s.group_stats[name]
            mapping = spec["mapping"]
            stat = np.array(
                [np.nan if k is None else mapping.get(k, np.nan) for k in self.keys(spec["by"])],
                dtype=np.float64,
            )
            if spec["ratio"]:
                with np.errstate(divide="ignore", invalid="ignore"):
                    stat = self.base(spec["value"]) / stat
                stat[np.isinf(stat)] = np.nan
            return stat.astype("float32")
        if name in s.frequency:
            col, mapping = s.frequency[name]
            freq = [np.nan if k is None else mapping.get(k, 0.0) for k in self.keys(col)]
            return np.array(freq, dtype=np.float64).astype("float32")
        if name in s.native_codes:
            codes = s.native_codes[name]
            return np.array(
                [np.nan if k is None else codes.get(k, np.nan) for k in self.keys(name)],
                dtype=np.float64,
            )
        if name in s.coded:
            codes = s.coded[name]
            return np.array(
                [np.nan if k is None else codes.get(k, np.nan) for k in self.keys(name)],
                dtype=np.float64,
            ).astype("float32")
        column = self.base(name)
        if isinstance(column, list):
            raise ValueError(f"Feature {name!r} is a string column without a fitted encoder")
        return column
