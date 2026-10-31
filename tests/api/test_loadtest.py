"""The load test script, run in-process against the API app (ASGI transport, fake model).

Same grid shape as the real run (single and batched requests, open loop), scaled down to a
fraction of a second so it runs with the unit tests.
"""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from pydantic import TypeAdapter

from fraud_detection.api.main import LoadedModel, ServiceState, app
from fraud_detection.api.schemas import Transaction, TransactionBatch

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "loadtest.py"
spec = importlib.util.spec_from_file_location("loadtest", SCRIPT)
loadtest = sys.modules["loadtest"] = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loadtest)


class CountingModel:
    """Scores TransactionAmt / 1000 and counts the transactions it saw."""

    metadata = SimpleNamespace(metadata={"threshold": 0.5})

    def __init__(self):
        self.rows = 0

    def predict(self, records):
        self.rows += len(records)
        return np.array([r["TransactionAmt"] / 1000 for r in records])


@pytest.fixture
def model():
    model = CountingModel()
    app.state.service = ServiceState(
        model_uri="models:/fraud-detection@champion",
        model=LoadedModel(pyfunc=model, uri="models:/fraud-detection/1", version="1"),
    )
    yield model
    del app.state.service


@pytest.fixture
def rows():
    """Synthetic rows in the request shape (the raw data files are not there in CI)."""
    return [
        {"TransactionID": i, "features": {**loadtest.SAMPLE, "TransactionAmt": float(i % 500)}}
        for i in range(200)
    ]


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://api")


@pytest.mark.parametrize("batch_size", [1, 10, 100])
def test_bodies_are_valid_requests(rows, batch_size):
    bodies = loadtest.make_bodies(rows, batch_size, n_bodies=3)
    for body in bodies:
        payload = json.loads(body)
        if batch_size == 1:
            Transaction.model_validate(payload)
        else:
            batch = TypeAdapter(TransactionBatch).validate_python(payload)
            ids = [tx.TransactionID for tx in batch]
            assert len(ids) == batch_size == len(set(ids))


def test_grid_sends_every_request_and_row(model, rows):
    requests, batch_sizes = [1, 10, 40], [1, 10, 100]

    async def run():
        async with _client() as client:
            assert await loadtest.check_ready(client, rows) is None
            warm_rows = model.rows
            summaries = await loadtest.run_grid(
                client, requests, batch_sizes, rows, duration=0.2, log=lambda _: None
            )
            return summaries, model.rows - warm_rows

    summaries, scored = asyncio.run(run())
    assert len(summaries) == len(requests) * len(batch_sizes)
    for s in summaries:
        assert s["errors"] == 0 and s["dropped"] == 0
        assert s["ok"] == s["sent"]
    assert scored == sum(n * b for n in requests for b in batch_sizes)
    singles = [s for s in summaries if s["batch_size"] == 1]
    assert all(s["budget"] == "PASS" for s in singles)
    assert all(s["budget"] == "-" for s in summaries if s["batch_size"] > 1)
    # Scaled to a minute: 40 requests in 0.2 s = 12000 requests per minute
    assert summaries[2]["requests_per_min"] == 12_000


def test_requests_over_the_in_flight_limit_are_dropped(rows):
    """Open loop: when answers are slow, the client drops requests instead of queueing them."""

    async def slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.2)
        return httpx.Response(200, headers={"X-Process-Time-Ms": "200"})

    async def run():
        transport = httpx.MockTransport(slow)
        async with httpx.AsyncClient(transport=transport, base_url="http://api") as client:
            bodies = loadtest.make_bodies(rows, 1, n_bodies=2)
            return await loadtest.run_scenario(
                client, 20, 1, bodies, duration=0.05, max_in_flight=5
            )

    result = asyncio.run(run())
    assert (result.sent, result.dropped, result.ok) == (5, 15, 5)
    assert result.summary()["budget"] == "FAIL"


def test_api_errors_are_counted(rows):
    async def run():
        async with _client() as client:  # no model loaded -> 503
            app.state.service = ServiceState(model_uri="models:/fraud-detection@champion")
            try:
                bodies = loadtest.make_bodies(rows, 10, n_bodies=2)
                return await loadtest.run_scenario(client, 4, 10, bodies, duration=0.05)
            finally:
                del app.state.service

    result = asyncio.run(run())
    assert result.ok == 0 and result.errors == {"HTTP 503": 4}
