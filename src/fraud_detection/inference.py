"""Raw transaction rows -> fraud probability, with the exact steps the batch pipelines run.

``InferenceBundle`` holds the fitted artifacts (column filter, feature encoders, model) and
the parameters they were fitted with. ``predict_proba`` chains the same node functions as
data_ingestion -> data_cleaning -> feature_engineering, so serving cannot drift from
training. ``predict_records`` is the fast path the API uses: the same features on NumPy arrays
(see ``fast_inference``), tested to give the same scores. ``explain_records`` adds per-row
SHAP values on the same path. ``FraudPyfunc`` wraps a bundle as an
MLflow pyfunc model for the registry and API.
"""

import logging
import operator
import pickle
from dataclasses import dataclass
from functools import cached_property

import mlflow.pyfunc
import numpy as np
import pandas as pd

from fraud_detection.fast_inference import RecordScorer, _is_missing, _normalize_names
from fraud_detection.keys import to_key
from fraud_detection.pipelines.data_cleaning.nodes import apply_column_filter
from fraud_detection.pipelines.data_ingestion.nodes import (
    expand_columns,
    normalize_identity_columns,
)
from fraud_detection.pipelines.feature_engineering.nodes import (
    add_stateless_features,
    apply_feature_encoders,
    build_model_input,
)
from fraud_detection.pipelines.training.model import FraudModel

logger = logging.getLogger(__name__)

TABLES = ("transaction", "identity")
# Same operators as data_validation's pandera checks, with the sign for error messages
_RANGE_CHECKS = {
    "gt": (operator.gt, ">"),
    "ge": (operator.ge, ">="),
    "lt": (operator.lt, "<"),
    "le": (operator.le, "<="),
}


@dataclass
class InferenceBundle:
    model: FraudModel
    column_filter: dict
    feature_encoders: dict
    ingestion_params: dict
    feature_params: dict
    target: str
    # data_validation's required_columns / value_ranges, checked per request. None in bundles
    # registered before it existed: those get only the unknown-column check.
    validation_params: dict | None = None

    def __getstate__(self) -> dict:
        """Pickle the fitted artifacts only; the ``_``-prefixed caches are rebuilt on use."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    @cached_property
    def _columns(self) -> dict:
        """Kept raw columns by type, plus every identity column (for ``has_identity``)."""
        schema = self.ingestion_params["schema"]
        kept = set(self.column_filter["kept"])
        numerical = [c for t in TABLES for c in expand_columns(schema[t]["numerical"])]
        categorical = [c for t in TABLES for c in expand_columns(schema[t]["categorical"])]
        identity = [
            c for k in ("categorical", "numerical") for c in expand_columns(schema["identity"][k])
        ]
        overrides = self.ingestion_params["numerical_dtype_overrides"]
        return {
            "numerical": {
                c: overrides.get(c, self.ingestion_params["numerical_dtype"])
                for c in numerical
                if c in kept
            },
            "categorical": [c for c in categorical if c in kept],
            "identity": identity,
        }

    @cached_property
    def _encoders(self) -> dict:
        """Encoders with lookup dicts as indexed Series.

        ``Series.map(dict)`` rebuilds a Series from the whole dict (100k+ keys for uid) on
        every call. Mapping through a Series whose index has the same ``string`` dtype as
        ``to_key`` output reuses its hash index: ~0.5 ms instead of ~250 ms. Same results.
        """

        def lookup(mapping: dict) -> pd.Series:
            return pd.Series(list(mapping.values()), index=pd.Index(list(mapping), dtype="string"))

        enc = self.feature_encoders
        # Native categories as an Index too: pd.Categorical re-hashes a plain list per call.
        categorical = {
            col: {**e, "codes": lookup(e["codes"])}
            if e["encoding"] == "code"
            else {**e, "categories": pd.Index(e["categories"])}
            for col, e in enc["categorical"].items()
        }
        return {
            **enc,
            "group_stats": [{**s, "mapping": lookup(s["mapping"])} for s in enc["group_stats"]],
            "frequency": {col: lookup(m) for col, m in enc["frequency"].items()},
            "categorical": categorical,
        }

    @cached_property
    def _known_columns(self) -> frozenset[str]:
        """Every column a request may send: both raw tables, the key, target and has_identity."""
        schema = self.ingestion_params["schema"]
        names = {
            c
            for t in TABLES
            for kind in ("categorical", "numerical")
            for c in expand_columns(schema[t][kind])
        }
        return frozenset({*names, self.ingestion_params["join_key"], self.target, "has_identity"})

    def check_records(self, records: list[dict]) -> dict[str, int]:
        """Per-request version of the data_validation rules, for raw rows as dicts.

        Raises ``ValueError`` (the API's 422) if a ``required_columns`` entry is missing or
        null, or a value is outside ``value_ranges``: invalid data is rejected, never scored.
        Returns the columns no raw table defines, with how many rows sent each. They are
        ignored, so a misspelled name is scored as a missing value; the caller should warn.
        """
        rules = self.validation_params or {}
        key = self.ingestion_params["join_key"]
        errors: list[str] = []
        unknown: dict[str, int] = {}
        for record in records:
            record = _normalize_names(record)
            for col in record.keys() - self._known_columns:
                unknown[col] = unknown.get(col, 0) + 1
            row = f"{key} {record.get(key)}"
            for col in rules.get("required_columns", []):
                if _is_missing(record.get(col)):
                    errors.append(f"{col} is required ({row})")
            for col, bounds in rules.get("value_ranges", {}).items():
                value = record.get(col)
                if _is_missing(value):
                    continue  # null-or-not is required_columns' job
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    errors.append(f"{col} must be a number, got {value!r} ({row})")
                    continue
                for op, bound in bounds.items():
                    compare, sign = _RANGE_CHECKS[op]
                    if not compare(number, bound):
                        errors.append(f"{col} must be {sign} {bound}, got {value!r} ({row})")
        if errors:
            more = [f"... and {len(errors) - 5} more"] if len(errors) > 5 else []
            raise ValueError("; ".join(errors[:5] + more))
        return dict(sorted(unknown.items()))

    def type_raw(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Type raw rows with the data_ingestion rules; absent columns become null.

        Only the columns the column filter keeps are typed (the rest would be dropped next).
        Requests carry transaction and identity columns in one row, so instead of a join
        ``has_identity`` is 1 when any identity column is set (unless the client sends it).
        """
        raw = normalize_identity_columns(raw)
        cols = self._columns
        key = self.ingestion_params["join_key"]
        if raw[key].isna().any() or raw[key].duplicated().any():
            raise ValueError(f"{key} must be non-null and unique")

        numerical = raw.reindex(columns=list(cols["numerical"])).astype("float64")
        # Integer columns (TransactionDT: day, hour and D_n_norm derive from it) cannot be null
        required = [c for c, dtype in cols["numerical"].items() if dtype.startswith("int")]
        absent = [c for c in required if numerical[c].isna().any()]
        if absent:
            raise ValueError(f"Required columns missing or null: {absent}")
        numerical = numerical.astype(cols["numerical"])
        categorical = {c: to_key(raw[c]) if c in raw else None for c in cols["categorical"]}
        if "has_identity" in raw.columns:
            has_identity = raw["has_identity"].fillna(0)
        else:
            has_identity = raw.reindex(columns=cols["identity"]).notna().any(axis=1)
        return pd.concat(
            [
                raw[[key]].astype("int64"),
                numerical,
                pd.DataFrame(categorical, index=raw.index, dtype="string"),
                has_identity.astype("int8").rename("has_identity"),
            ],
            axis=1,
        ).copy()  # one consolidated block per dtype; the next steps add columns

    def model_input(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Raw rows -> the model input table (join key + features), as in 05_model_input."""
        df = apply_column_filter(self.type_raw(raw), self.column_filter)
        df = add_stateless_features(df, self.feature_params)
        df = apply_feature_encoders(df, self._encoders)
        return build_model_input(df, self.feature_encoders, self.feature_params, self.target)

    def predict_proba(self, raw: pd.DataFrame) -> np.ndarray:
        X = self.model_input(raw).drop(columns=[self.ingestion_params["join_key"]])
        return self.model.predict_proba(X)

    @cached_property
    def _scorer(self) -> RecordScorer:
        return RecordScorer(self)

    def predict_records(self, records: list[dict]) -> np.ndarray:
        """Fast path: raw rows as dicts (one or many) -> the same scores as ``predict_proba``."""
        return self._scorer.predict(records)

    def explain_records(self, records: list[dict], top_k: int = 10) -> list[list[dict]]:
        """Per row, the ``top_k`` features with the largest |SHAP| (log-odds), largest first,
        with the model input value each one had (categories as labels, not codes)."""
        scorer = self._scorer
        values, X = scorer.shap_values(records)
        top = np.argsort(-np.abs(values), axis=1, kind="stable")[:, :top_k]
        return [
            [
                {
                    "feature": scorer.features[j],
                    "value": scorer.label(scorer.features[j], X[i, j]),
                    "shap": float(values[i, j]),
                }
                for j in row
            ]
            for i, row in enumerate(top)
        ]


class FraudPyfunc(mlflow.pyfunc.PythonModel):
    """MLflow wrapper: ``predict(raw rows) -> fraud probabilities``.

    A list of dicts (what the API sends) takes the NumPy fast path; a DataFrame the pandas
    path of the batch pipelines.

    The bundle is a pickle artifact; the serving environment must have this package
    installed (the API image does).
    """

    def load_context(self, context):
        with open(context.artifacts["bundle"], "rb") as f:
            self.bundle: InferenceBundle = pickle.load(f)

    def predict(self, context, model_input, params=None):
        if isinstance(model_input, pd.DataFrame):
            return self.bundle.predict_proba(model_input)
        return self.bundle.predict_records(model_input)

    def check_records(self, records: list[dict]) -> dict[str, int]:
        """Request validation (see ``InferenceBundle.check_records``); the API calls it first."""
        return self.bundle.check_records(records)

    def explain(self, records: list[dict], top_k: int = 10) -> list[list[dict]]:
        """Top SHAP contributions per row (see ``InferenceBundle.explain_records``)."""
        return self.bundle.explain_records(records, top_k)
