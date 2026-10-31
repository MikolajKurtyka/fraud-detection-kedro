"""Micro-batching of /predict (MICROBATCH_WAIT_MS > 0): concurrent requests, one model call.

Runs the real app lifespan with a fake model and sends concurrent requests over the ASGI
transport, so the batcher sees them at the same time.
"""

import asyncio
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from fraud_detection.api import main
from fraud_detection.api.main import LoadedModel, app
from fraud_detection.api.microbatch import MicroBatcher


class BatchModel:
    """probability = TransactionAmt / 100; records every model call (warm-up excluded)."""

    metadata = SimpleNamespace(metadata={"threshold": 0.3})

    def __init__(self, fail_on: str = "boom"):
        self.calls: list[list[dict]] = []
        self.fail_on = fail_on

    def predict(self, records):
        if records[0]["TransactionID"] != 0:  # 0 is the startup warm-up
            self.calls.append(records)
        if any(self.fail_on in r for r in records):
            raise ValueError(f"bad column {self.fail_on}")
        return np.array([r.get("TransactionAmt", 0) / 100 for r in records])


@pytest.fixture
def model(monkeypatch):
    model = BatchModel()
    monkeypatch.setattr(
        main, "load_model", lambda uri: LoadedModel(model, "models:/fraud-detection/1", "1")
    )
    monkeypatch.delenv("FRAUD_THRESHOLD", raising=False)
    monkeypatch.delenv("MICROBATCH_WAIT_MS", raising=False)
    monkeypatch.delenv("MICROBATCH_MAX_SIZE", raising=False)
    return model


def tx(tid, amt, **extra):
    return {"TransactionID": tid, "features": {"TransactionAmt": amt, **extra}}


def post_concurrently(payloads: list[dict]) -> tuple[list[httpx.Response], object]:
    """Start the app (lifespan included), send all requests at once; return responses and the
    batcher the app ran with."""

    async def run():
        async with app.router.lifespan_context(app):
            batcher = app.state.batcher
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://api") as client:
                responses = await asyncio.gather(
                    *(client.post("/predict", json=p) for p in payloads)
                )
            return responses, batcher

    return asyncio.run(run())


def test_off_by_default_one_call_per_request(model):
    responses, batcher = post_concurrently([tx(i, i) for i in range(1, 21)])
    assert batcher is None
    assert all(r.status_code == 200 for r in responses)
    assert [len(c) for c in model.calls] == [1] * 20


def test_concurrent_requests_share_model_calls(model, monkeypatch):
    monkeypatch.setenv("MICROBATCH_WAIT_MS", "50")
    payloads = [tx(i, i) for i in range(1, 41)]
    responses, batcher = post_concurrently(payloads)

    assert all(r.status_code == 200 for r in responses)
    assert sum(len(c) for c in model.calls) == 40
    assert len(model.calls) < 40 and max(batcher.batch_sizes) > 1
    # Each request gets its own transaction's score, threshold and request ID
    for payload, resp in zip(payloads, responses, strict=True):
        body = resp.json()
        assert body["TransactionID"] == payload["TransactionID"]
        assert body["fraud_probability"] == pytest.approx(payload["TransactionID"] / 100)
        assert body["is_fraud"] == (body["fraud_probability"] >= 0.3)
        assert resp.headers["X-Request-ID"]


def test_batches_respect_max_size(model, monkeypatch):
    monkeypatch.setenv("MICROBATCH_WAIT_MS", "50")
    monkeypatch.setenv("MICROBATCH_MAX_SIZE", "4")
    responses, batcher = post_concurrently([tx(i, i) for i in range(1, 21)])
    assert all(r.status_code == 200 for r in responses)
    assert max(len(c) for c in model.calls) <= 4
    assert max(batcher.batch_sizes) == 4


def test_one_bad_transaction_does_not_fail_the_others(model, monkeypatch):
    monkeypatch.setenv("MICROBATCH_WAIT_MS", "50")
    payloads = [tx(i, i) for i in range(1, 11)]
    payloads[3] = tx(4, 4, boom=1)
    responses, _ = post_concurrently(payloads)

    assert responses[3].status_code == 422
    assert "boom" in responses[3].json()["detail"]
    ok = [r for i, r in enumerate(responses) if i != 3]
    assert all(r.status_code == 200 for r in ok)
    assert [r.json()["TransactionID"] for r in ok] == [
        p["TransactionID"] for p in payloads if "boom" not in p["features"]
    ]


def test_no_model_gives_503_with_batching_on(monkeypatch):
    monkeypatch.setenv("MICROBATCH_WAIT_MS", "20")

    def fail(uri):
        raise RuntimeError("registry unreachable")

    monkeypatch.setattr(main, "load_model", fail)
    responses, _ = post_concurrently([tx(1, 5), tx(2, 5)])
    assert [r.status_code for r in responses] == [503, 503]


def test_batcher_rejects_bad_settings_and_unstarted_use():
    with pytest.raises(ValueError):
        MicroBatcher(lambda items: items, max_wait_ms=0, max_size=8)
    with pytest.raises(ValueError):
        MicroBatcher(lambda items: items, max_wait_ms=5, max_size=0)

    async def submit_unstarted():
        await MicroBatcher(lambda items: items, max_wait_ms=5, max_size=8).submit(1)

    with pytest.raises(RuntimeError, match="not running"):
        asyncio.run(submit_unstarted())


def test_batcher_survives_a_failing_score_function():
    calls = []

    def score(items):
        calls.append(items)
        if len(calls) == 1:
            raise RuntimeError("crash")
        return [i * 2 for i in items]

    async def run():
        batcher = MicroBatcher(score, max_wait_ms=5, max_size=8)
        batcher.start()
        try:
            with pytest.raises(RuntimeError, match="crash"):
                await batcher.submit(1)
            return await batcher.submit(21)
        finally:
            await batcher.stop()

    assert asyncio.run(run()) == 42
