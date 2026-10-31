from fastapi.testclient import TestClient

from fraud_detection.api.main import app


def test_health_without_model():
    with TestClient(app) as client:
        body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is False


def test_predict_returns_503_without_model():
    with TestClient(app) as client:
        resp = client.post("/predict", json={"TransactionID": 1, "features": {"TransactionAmt": 5}})
    assert resp.status_code == 503
