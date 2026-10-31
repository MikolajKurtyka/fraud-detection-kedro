"""Async open-loop load test for the scoring API.

A scenario sends R requests of B transactions each, spread evenly over one minute: request i
leaves at i * 60 / R seconds whether or not the earlier ones have answered, so the measured
latency includes queueing when the API falls behind (a closed loop would hide that). B = 1
uses ``/predict``, B > 1 ``/predict/batch``.

Default grid: R in {1, 10, 1000, 10000} requests per minute x B in {1, 10, 100, 1000}
transactions per request. Payloads are real raw rows from data/01_raw (transaction + identity,
nulls left out), as a client would send them; synthetic rows if the raw files are missing.

If more than --max-in-flight requests are waiting, new ones are dropped and counted, so an
overloaded API cannot exhaust the client. Idle connections are kept open longer than the gap
between requests, so slow scenarios do not pay a new connection each time. Use 127.0.0.1, not
localhost: on Windows a new connection to localhost tries IPv6 first and costs ~2 s.
The p99 < 50 ms budget is checked for B = 1. Writes data/08_reporting/{--name}.{csv,md}.

    uv run python scripts/loadtest.py                                  # full grid, ~12 min
    uv run python scripts/loadtest.py --requests 1000 --batch-sizes 1  # one scenario
    uv run python scripts/loadtest.py --duration 10                    # same counts in 10 s
    uv run python scripts/loadtest.py --clients 4                      # 4 client processes

One asyncio client process tops out around 150 requests per second; above that the client,
not the API, sets the latency. --clients N splits each scenario over N processes
(--max-in-flight applies per process).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "data" / "08_reporting"
BUDGET_P99_MS = 50.0
MAX_BATCH_SIZE = 1000  # schemas.MAX_BATCH_SIZE
HEADERS = {"Content-Type": "application/json"}

SAMPLE = {
    "TransactionDT": 18403224,
    "TransactionAmt": 31.95,
    "ProductCD": "W",
    "card1": 10409,
    "addr1": 315,
    "P_emaildomain": "gmail.com",
}


def load_rows(n_rows: int, seed: int = 0) -> list[dict]:
    """``n_rows`` raw rows as ``{"TransactionID", "features"}`` dicts, nulls left out."""
    tx_path = ROOT / "data/01_raw/train_transaction.csv"
    id_path = ROOT / "data/01_raw/train_identity.csv"
    if not (tx_path.exists() and id_path.exists()):
        print(f"{tx_path} not found, using synthetic rows")
        rng = random.Random(seed)
        return [
            {
                "TransactionID": 3_000_000 + i,
                "features": {**SAMPLE, "TransactionAmt": round(rng.uniform(1, 500), 2)},
            }
            for i in range(n_rows)
        ]
    tx = pd.read_csv(tx_path, nrows=n_rows).drop(columns="isFraud")
    identity = pd.read_csv(id_path)
    raw = tx.merge(identity, on="TransactionID", how="left")
    rows = []
    for record in raw.astype(object).to_dict("records"):
        tid = int(record.pop("TransactionID"))
        rows.append({"TransactionID": tid, "features": _drop_nulls(record)})
    return rows


def _drop_nulls(record: dict) -> dict:
    return {k: v for k, v in record.items() if not (isinstance(v, float) and math.isnan(v))}


def make_bodies(rows: list[dict], batch_size: int, n_bodies: int) -> list[bytes]:
    """Pre-serialized request bodies: one transaction, or a list of ``batch_size`` distinct ones.

    Bodies cycle through ``rows``; IDs are unique within a body (the API requires it).
    """
    if batch_size > len(rows):
        raise ValueError(f"batch_size {batch_size} > {len(rows)} sample rows")
    bodies = []
    for b in range(n_bodies):
        start = b * batch_size
        chunk = [rows[(start + i) % len(rows)] for i in range(batch_size)]
        bodies.append(json.dumps(chunk[0] if batch_size == 1 else chunk).encode())
    return bodies


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(q / 100 * len(ordered)) - 1))]


@dataclass
class ScenarioResult:
    requests: int  # planned per duration
    batch_size: int
    duration: float
    sent: int = 0
    ok: int = 0
    dropped: int = 0
    errors: Counter = field(default_factory=Counter)
    latencies_ms: list[float] = field(default_factory=list)
    server_ms: list[float] = field(default_factory=list)
    elapsed: float = 0.0

    @classmethod
    def merge(cls, parts: list[ScenarioResult]) -> ScenarioResult:
        """One result from the shares several client processes sent."""
        out = cls(sum(p.requests for p in parts), parts[0].batch_size, parts[0].duration)
        for p in parts:
            out.sent += p.sent
            out.ok += p.ok
            out.dropped += p.dropped
            out.errors.update(p.errors)
            out.latencies_ms += p.latencies_ms
            out.server_ms += p.server_ms
        out.elapsed = max(p.elapsed for p in parts)
        return out

    def summary(self) -> dict:
        lat = self.latencies_ms
        p99 = percentile(lat, 99)
        single = self.batch_size == 1
        clean = self.ok == self.requests and not self.errors
        return {
            "requests_per_min": round(self.requests * 60 / self.duration),
            "batch_size": self.batch_size,
            "rows_per_min": round(self.requests * self.batch_size * 60 / self.duration),
            "sent": self.sent,
            "ok": self.ok,
            "errors": sum(self.errors.values()),
            "dropped": self.dropped,
            "req_per_s": round(self.ok / self.elapsed, 1) if self.elapsed else 0.0,
            "rows_per_s": round(self.ok * self.batch_size / self.elapsed) if self.elapsed else 0,
            "p50_ms": round(percentile(lat, 50), 1),
            "p95_ms": round(percentile(lat, 95), 1),
            "p99_ms": round(p99, 1),
            "max_ms": round(max(lat), 1) if lat else float("nan"),
            "server_p50_ms": round(percentile(self.server_ms, 50), 1),
            "ms_per_row_p50": round(percentile(lat, 50) / self.batch_size, 3),
            "budget": ("PASS" if clean and p99 < BUDGET_P99_MS else "FAIL") if single else "-",
            "error_kinds": dict(self.errors) or "",
        }


async def run_scenario(
    client: httpx.AsyncClient,
    n_requests: int,
    batch_size: int,
    bodies: list[bytes],
    duration: float = 60.0,
    max_in_flight: int = 256,
    start_offset: float = 0.0,
) -> ScenarioResult:
    """Send ``n_requests`` evenly over ``duration`` seconds without waiting for answers.

    ``start_offset`` shifts the schedule, so client processes sharing a scenario interleave.
    """
    path = "/predict" if batch_size == 1 else "/predict/batch"
    result = ScenarioResult(n_requests, batch_size, duration)
    in_flight: set[asyncio.Task] = set()

    async def send(body: bytes) -> None:
        start = time.perf_counter()
        try:
            resp = await client.post(path, content=body, headers=HEADERS)
        except httpx.HTTPError as exc:
            result.errors[type(exc).__name__] += 1
            return
        if resp.status_code != 200:
            result.errors[f"HTTP {resp.status_code}"] += 1
            return
        result.ok += 1
        result.latencies_ms.append((time.perf_counter() - start) * 1000)
        if server := resp.headers.get("X-Process-Time-Ms"):
            result.server_ms.append(float(server))

    loop = asyncio.get_running_loop()
    interval = duration / n_requests
    start = loop.time()
    for i in range(n_requests):
        delay = start + start_offset + i * interval - loop.time()
        if delay > 0:
            await asyncio.sleep(delay)
        if len(in_flight) >= max_in_flight:
            result.dropped += 1
            continue
        result.sent += 1
        task = asyncio.create_task(send(bodies[i % len(bodies)]))
        in_flight.add(task)
        task.add_done_callback(in_flight.discard)
    await asyncio.gather(*in_flight)
    result.elapsed = loop.time() - start
    return result


def client_shares(n_requests: int, clients: int) -> list[tuple[int, float]]:
    """Split a scenario over client processes: (requests, offset in schedule intervals).

    Client k sends requests k, k + clients, k + 2 * clients, ... of the combined schedule, so
    together they keep the same even spacing.
    """
    shares = []
    for k in range(min(clients, n_requests)):
        n_k = len(range(k, n_requests, clients))
        shares.append((n_k, k))
    return shares


# One client process: rows are sent once per process (pool initializer), not per scenario
_ROWS: list[dict] = []


def _init_client_process(rows: list[dict]) -> None:
    _ROWS[:] = rows


def _client_process(
    url: str,
    timeout: float,
    n_requests: int,
    batch_size: int,
    duration: float,
    max_in_flight: int,
    start_offset: float,
) -> ScenarioResult:
    bodies = make_bodies(_ROWS, batch_size, n_bodies=_n_bodies(len(_ROWS), batch_size))

    async def run() -> ScenarioResult:
        limits = httpx.Limits(max_connections=max_in_flight, keepalive_expiry=duration + 5)
        async with httpx.AsyncClient(base_url=url, timeout=timeout, limits=limits) as client:
            return await run_scenario(
                client, n_requests, batch_size, bodies, duration, max_in_flight, start_offset
            )

    return asyncio.run(run())


def _n_bodies(n_rows: int, batch_size: int) -> int:
    """Distinct bodies so every request is not the same transaction; fewer for big batches."""
    return max(2, min(50, n_rows // batch_size))


async def run_grid(
    client: httpx.AsyncClient,
    requests: list[int],
    batch_sizes: list[int],
    rows: list[dict],
    duration: float = 60.0,
    max_in_flight: int = 256,
    log=print,
    pool: ProcessPoolExecutor | None = None,
    clients: int = 1,
) -> list[dict]:
    """Run every (requests, batch size) scenario; with ``pool``, each one is split over
    ``clients`` processes (one asyncio client each) and the results merged."""
    summaries = []
    loop = asyncio.get_running_loop()
    for batch_size in batch_sizes:
        bodies = make_bodies(rows, batch_size, n_bodies=_n_bodies(len(rows), batch_size))
        for n in requests:
            log(f"-> {n} requests x {batch_size} rows over {duration:.0f} s ...")
            if pool is None or clients == 1:
                result = await run_scenario(client, n, batch_size, bodies, duration, max_in_flight)
            else:
                interval = duration / n
                parts = await asyncio.gather(
                    *(
                        loop.run_in_executor(
                            pool,
                            _client_process,
                            str(client.base_url),
                            client.timeout.read,
                            n_k,
                            batch_size,
                            duration,
                            max_in_flight,
                            k * interval,
                        )
                        for n_k, k in client_shares(n, clients)
                    )
                )
                result = ScenarioResult.merge(parts)
            summary = result.summary()
            log(
                f"   ok={summary['ok']}/{n} errors={summary['errors']} "
                f"dropped={summary['dropped']} p50={summary['p50_ms']}ms "
                f"p99={summary['p99_ms']}ms rows/s={summary['rows_per_s']} {summary['budget']}"
            )
            summaries.append(summary)
    return summaries


async def check_ready(client: httpx.AsyncClient, rows: list[dict]) -> str | None:
    """Warm both endpoints; return an error message if the API cannot score."""
    try:
        single = await client.post("/predict", json=rows[0])
        single.raise_for_status()
        batch = await client.post("/predict/batch", json=rows[:2])
        batch.raise_for_status()
    except httpx.HTTPError as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def to_markdown(table: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(table.columns) + " |", "|" + "---|" * len(table.columns)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in table.itertuples(False)]
    return "\n".join(lines) + "\n"


def write_report(summaries: list[dict], args: argparse.Namespace) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(summaries)
    table.to_csv(REPORT_DIR / f"{args.name}.csv", index=False)
    header = (
        f"# Load test\n\nAPI: `{args.url}`, each scenario spread over {args.duration:.0f} s, "
        f"{args.clients} client process(es), max {args.max_in_flight} requests in flight each. "
        f"Budget: p99 < {BUDGET_P99_MS:.0f} ms for single transactions. "
        "Latency is client-side and includes queueing.\n\n"
    )
    (REPORT_DIR / f"{args.name}.md").write_text(header + to_markdown(table), encoding="utf-8")
    print(table.drop(columns="error_kinds").to_string(index=False))
    print(f"\nWrote {REPORT_DIR / args.name}.csv and .md")


async def main_async(args: argparse.Namespace) -> int:
    rows = load_rows(max(2 * max(args.batch_sizes), 100))
    # Keep idle connections longer than the gap between requests (60 s at 1 request per minute)
    limits = httpx.Limits(max_connections=args.max_in_flight, keepalive_expiry=args.duration + 5)
    async with httpx.AsyncClient(base_url=args.url, timeout=args.timeout, limits=limits) as client:
        if error := await check_ready(client, rows):
            print(f"API not ready at {args.url}: {error}")
            return 1
        pool = None
        if args.clients > 1:
            pool = ProcessPoolExecutor(
                args.clients, initializer=_init_client_process, initargs=(rows,)
            )
        try:
            summaries = await run_grid(
                client,
                args.requests,
                args.batch_sizes,
                rows,
                args.duration,
                args.max_in_flight,
                pool=pool,
                clients=args.clients,
            )
        finally:
            if pool is not None:
                pool.shutdown()
    write_report(summaries, args)
    return 0 if all(s["budget"] != "FAIL" for s in summaries) else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, nargs="+", default=[1, 10, 1000, 10000])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 10, 100, 1000])
    parser.add_argument("--duration", type=float, default=60.0, help="seconds per scenario")
    parser.add_argument("--max-in-flight", type=int, default=256, help="per client process")
    parser.add_argument("--clients", type=int, default=1, help="client processes")
    parser.add_argument("--timeout", type=float, default=30.0, help="seconds per request")
    parser.add_argument("--name", default="loadtest", help="report file name in 08_reporting")
    args = parser.parse_args()
    if max(args.batch_sizes) > MAX_BATCH_SIZE or min(args.batch_sizes) < 1:
        parser.error(f"batch sizes must be 1..{MAX_BATCH_SIZE}")
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
