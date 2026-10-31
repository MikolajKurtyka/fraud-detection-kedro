"""Optional micro-batching: score concurrent single-transaction requests in one model call.

Off by default. With ``MICROBATCH_WAIT_MS`` > 0, ``/predict`` hands its transaction to a
``MicroBatcher`` instead of scoring it alone. The batcher takes the first waiting request,
collects more for up to ``MICROBATCH_WAIT_MS`` (or until ``MICROBATCH_MAX_SIZE``), scores them
together in a worker thread, and answers each request with its own result. While one batch is
scored, the next one fills up, so under load batches grow by themselves and when traffic is low
a request waits at most ``MICROBATCH_WAIT_MS`` extra.

Each item gets its own result or exception, so one bad transaction does not fail the others
(the scoring function decides how; see ``main._score_items``).
"""

import asyncio
import contextlib
import logging
from collections.abc import Callable
from typing import Any

from starlette.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)


class MicroBatcher:
    def __init__(
        self,
        score: Callable[[list[Any]], list[Any]],
        max_wait_ms: float,
        max_size: int,
    ):
        """``score(items)`` returns one result or exception per item, in order."""
        if max_wait_ms <= 0 or max_size < 1:
            raise ValueError("max_wait_ms must be > 0 and max_size >= 1")
        self.score = score
        self.max_wait = max_wait_ms / 1000
        self.max_size = max_size
        self.queue: asyncio.Queue[tuple[Any, asyncio.Future]] = asyncio.Queue()
        self.task: asyncio.Task | None = None
        self.batch_sizes: list[int] = []  # recent batch sizes, for tests and debugging

    def start(self) -> None:
        self.task = asyncio.create_task(self._run(), name="microbatcher")

    async def stop(self) -> None:
        if self.task is not None:
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task
            self.task = None

    async def submit(self, item: Any) -> Any:
        """Queue one item and wait for its result (or raise its exception)."""
        if self.task is None or self.task.done():
            raise RuntimeError("Micro-batcher is not running")
        future = asyncio.get_running_loop().create_future()
        await self.queue.put((item, future))
        return await future

    async def _collect(self) -> list[tuple[Any, asyncio.Future]]:
        batch = [await self.queue.get()]
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.max_wait
        while len(batch) < self.max_size:
            timeout = deadline - loop.time()
            if timeout <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(self.queue.get(), timeout))
            except TimeoutError:
                break
        return batch

    async def _run(self) -> None:
        while True:
            batch = await self._collect()
            items = [item for item, _ in batch]
            try:
                results = await run_in_threadpool(self.score, items)
                if len(results) != len(items):
                    raise RuntimeError(f"{len(results)} results for {len(items)} items")
            except Exception as exc:  # unexpected: fail this batch, keep serving
                logger.exception("Micro-batch of %d failed", len(items))
                results = [exc] * len(items)
            self.batch_sizes = [*self.batch_sizes[-999:], len(items)]
            for (_, future), result in zip(batch, results, strict=True):
                if future.done():  # the client went away
                    continue
                if isinstance(result, BaseException):
                    future.set_exception(result)
                else:
                    future.set_result(result)
