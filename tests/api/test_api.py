"""API tests with a fake pyfunc model injected through ``load_model`` (no MLflow, no training)."""

import logging
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from fraud_detection.api import main
from fraud_detection.api.main import LoadedModel, app


class FakeModel:
    """Pyfunc look-alike: probability = TransactionAmt / 100, clipped to [0, 1].

    Gets what the real model gets from the API: a list of dicts, one per transaction.
    """

    def __init__(self, metadata=None):
        self.metadata = SimpleNamespace(metadata=metadata or {})
        self.calls = []

    def predict(self, records):
        self.calls.append(records)
        if any("boom" in r for r in records):
            raise ValueError("bad column boom")
        if any("crash" in r for r in records):
            raise RuntimeError("internal error")
        amt = [r.get("TransactionAmt", 0) for r in records]
        return np.clip(np.asarray(amt, dtype=float) / 100, 0, 1)

    def explain(self, records, top_k):
        """Two fixed contributions per row; the amount pushes towards fraud."""
        if any("boom" in r for r in records):
            raise ValueError("bad column boom")
        contributions = [
            {"feature": "TransactionAmt", "value": records[0].get("TransactionAmt"), "shap": 1.5},
            {"feature": "ProductCD", "value": "W", "shap": -0.2},
        ]
        return [contributions[:top_k] for _ in records]


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("FRAUD_THRESHOLD", raising=False)
    monkeypatch.delenv("MODEL_URI", raising=False)


@pytest.fixture
def fake_model(monkeypatch):
    model = FakeModel({"threshold": 0.3, "model_type": "lightgbm"})

    def _load(uri):
        return LoadedModel(pyfunc=model, uri="models:/fraud-detection/7", version="7")

    monkeypatch.setattr(main, "load_model", _load)
    return model


@pytest.fixture
def no_model(monkeypatch):
    def _load(uri):
        raise RuntimeError("registry unreachable")

    monkeypatch.setattr(main, "load_model", _load)


def tx(tid, amt, **extra):
    return {"TransactionID": tid, "features": {"TransactionAmt": amt, **extra}}


def test_live_always_ok(no_model):
    with TestClient(app) as client:
        resp = client.get("/health/live")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ready_503_without_model_and_health_still_200(no_model):
    with TestClient(app) as client:
        ready = client.get("/health/ready")
        health = client.get("/health")
    assert ready.status_code == 503
    assert ready.json()["model_loaded"] is False
    assert "registry unreachable" in ready.json()["detail"]
    assert health.status_code == 200
    assert health.json()["status"] == "ok"


def test_predict_503_without_model(no_model):
    with TestClient(app) as client:
        resp = client.post("/predict", json=tx(1, 5))
        batch = client.post("/predict/batch", json=[tx(1, 5)])
    assert resp.status_code == 503
    assert batch.status_code == 503


def test_ready_reports_model(fake_model):
    with TestClient(app) as client:
        resp = client.get("/health/ready")
    body = resp.json()
    assert resp.status_code == 200
    assert body["model_uri"] == "models:/fraud-detection@champion"
    assert body["resolved_model_uri"] == "models:/fraud-detection/7"
    assert body["model_version"] == "7"
    assert body["model_type"] == "lightgbm"
    assert body["threshold"] == 0.3


def test_warm_up_runs_at_startup(fake_model):
    with TestClient(app):
        pass
    # A valid minimal row: TransactionDT is required by the inference bundle
    assert list(fake_model.calls[0][0]) == ["TransactionID", "TransactionDT", "TransactionAmt"]


def test_predict_single(fake_model):
    with TestClient(app) as client:
        resp = client.post("/predict", json=tx(42, 50, ProductCD="W", card4=None))
    body = resp.json()
    assert resp.status_code == 200
    assert body == {
        "TransactionID": 42,
        "fraud_probability": 0.5,
        "is_fraud": True,
        "threshold": 0.3,
        "model_version": "7",
        "model_uri": "models:/fraud-detection/7",
    }
    record = fake_model.calls[-1][0]
    assert record["TransactionID"] == 42
    assert record["ProductCD"] == "W"


def test_threshold_env_overrides_metadata(fake_model, monkeypatch):
    monkeypatch.setenv("FRAUD_THRESHOLD", "0.9")
    with TestClient(app) as client:
        body = client.post("/predict", json=tx(1, 50)).json()
    assert body["threshold"] == 0.9
    assert body["is_fraud"] is False


def test_threshold_defaults_to_half_without_metadata(monkeypatch):
    monkeypatch.setattr(main, "load_model", lambda uri: LoadedModel(FakeModel(), uri, "1"))
    with TestClient(app) as client:
        assert client.post("/predict", json=tx(1, 40)).json()["threshold"] == 0.5


def test_batch_keeps_order_in_one_call(fake_model):
    payload = [tx(3, 90), tx(1, 10), tx(2, 50)]
    with TestClient(app) as client:
        n_calls = len(fake_model.calls)
        resp = client.post("/predict/batch", json=payload)
    body = resp.json()
    assert resp.status_code == 200
    assert [p["TransactionID"] for p in body] == [3, 1, 2]
    assert [p["fraud_probability"] for p in body] == [0.9, 0.1, 0.5]
    assert [p["is_fraud"] for p in body] == [True, False, True]
    assert len(fake_model.calls) == n_calls + 1


def test_batch_size_limits(fake_model):
    with TestClient(app) as client:
        assert client.post("/predict/batch", json=[]).status_code == 422
        assert (
            client.post("/predict/batch", json=[tx(i, 1) for i in range(1001)]).status_code == 422
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"features": {"TransactionAmt": 5}},  # missing TransactionID
        {"TransactionID": "abc", "features": {}},
        {"TransactionID": 1, "features": {"a": [1, 2]}},  # nested values
        {"TransactionID": 1, "features": {f"c{i}": 1 for i in range(601)}},  # too many
    ],
)
def test_bad_payload_422(fake_model, payload):
    with TestClient(app) as client:
        assert client.post("/predict", json=payload).status_code == 422


def test_model_errors_map_to_422_and_500(fake_model):
    with TestClient(app) as client:
        bad = client.post("/predict", json=tx(1, 5, boom=1))
        crash = client.post("/predict", json=tx(1, 5, crash=1))
    assert bad.status_code == 422
    assert "boom" in bad.json()["detail"]
    assert crash.status_code == 500


def test_request_id_echo_and_timing(fake_model):
    with TestClient(app) as client:
        echoed = client.post("/predict", json=tx(1, 5), headers={"X-Request-ID": "abc-123"})
        generated = client.get("/health/live")
    assert echoed.headers["X-Request-ID"] == "abc-123"
    assert float(echoed.headers["X-Process-Time-Ms"]) >= 0
    assert generated.headers["X-Request-ID"]


def test_predictions_are_logged(fake_model, caplog):
    with (
        caplog.at_level(logging.INFO, logger="fraud_detection.api.predictions"),
        TestClient(app) as client,
    ):
        client.post("/predict", json=tx(77, 50))
    records = [r for r in caplog.records if r.name == "fraud_detection.api.predictions"]
    assert any('"TransactionID": 77' in r.getMessage() for r in records)


def test_metrics_endpoint(fake_model):
    with TestClient(app) as client:
        client.post("/predict", json=tx(1, 50))
        resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "fraud_prediction_probability_bucket" in resp.text
    assert 'fraud_predictions_total{is_fraud="true",model_version="7"}' in resp.text
    assert "http_request_duration" in resp.text


def test_explain_returns_prediction_and_contributions(fake_model):
    with TestClient(app) as client:
        resp = client.post("/explain?top_k=1", json=tx(5, 50))
    body = resp.json()
    assert resp.status_code == 200
    assert body["fraud_probability"] == 0.5
    assert body["is_fraud"] is True
    assert body["contributions"] == [{"feature": "TransactionAmt", "value": 50.0, "shap": 1.5}]


def test_explain_errors(fake_model, monkeypatch):
    with TestClient(app) as client:
        assert client.post("/explain?top_k=0", json=tx(1, 5)).status_code == 422
        assert client.post("/explain", json=tx(1, 5, boom=1)).status_code == 422
    monkeypatch.delattr(FakeModel, "explain")
    with TestClient(app) as client:
        assert client.post("/explain", json=tx(1, 5)).status_code == 501


def test_explain_503_without_model(no_model):
    with TestClient(app) as client:
        assert client.post("/explain", json=tx(1, 5)).status_code == 503
