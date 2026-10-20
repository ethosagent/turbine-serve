"""Block 3: async <-> engine-thread bridge; the API's only door to the engine."""

import asyncio
import itertools
import threading
from collections.abc import AsyncIterator, Callable
from typing import Protocol

from spark_engine.log import get_logger
from spark_engine.types import SamplingParams, Sequence, TokenOutput

log = get_logger("engine_client")

_JOIN_TIMEOUT_S = 5


class EngineLike(Protocol):
    """The slice of Engine the client uses; keeps this file off engine.py's block imports."""

    on_output: Callable[[list[TokenOutput]], None] | None

    def add_request(self, seq: Sequence) -> None: ...
    def abort(self, seq_id: str) -> None: ...
    @property
    def num_waiting(self) -> int: ...
    def run_forever(self) -> None: ...
    def stop(self) -> None: ...


class QueueFull(Exception):
    """Too many requests are already waiting for admission."""


class EngineClient:
    """Runs the engine on a daemon thread and streams its outputs to per-request asyncio queues."""

    def __init__(self, engine: EngineLike, max_queue: int) -> None:
        self._engine = engine
        self._max_queue = max_queue
        self._queues: dict[str, asyncio.Queue[TokenOutput]] = {}
        self._counter = itertools.count()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Call from inside the running event loop."""
        self._loop = asyncio.get_running_loop()
        self._engine.on_output = self._on_output
        self._thread = threading.Thread(target=self._engine.run_forever, name="engine", daemon=True)
        self._thread.start()

    async def stop(self) -> None:
        self._engine.stop()
        if self._thread is not None:
            await asyncio.to_thread(self._thread.join, _JOIN_TIMEOUT_S)

    async def generate(
        self, prompt_ids: list[int], params: SamplingParams, request_id: str
    ) -> AsyncIterator[TokenOutput]:
        if self._engine.num_waiting >= self._max_queue:
            raise QueueFull(f"{self._engine.num_waiting} requests waiting")
        seq_id = f"s{next(self._counter)}"
        seq = Sequence(
            seq_id=seq_id, request_id=request_id, prompt_ids=list(prompt_ids), params=params
        )
        out_q: asyncio.Queue[TokenOutput] = asyncio.Queue()
        self._queues[seq_id] = out_q  # register first so no output can be missed
        finished = False
        try:
            self._engine.add_request(seq)
            log.info(
                "seq.submitted",
                request_id=request_id,
                seq_id=seq_id,
                prompt_tokens=len(prompt_ids),
            )
            while not finished:
                out = await out_q.get()
                finished = out.finished
                yield out
        finally:
            del self._queues[seq_id]
            if not finished:
                self._engine.abort(seq_id)
                log.info("seq.aborted", request_id=request_id, seq_id=seq_id)

    def _on_output(self, outs: list[TokenOutput]) -> None:
        """Engine thread: hand each output to its request's queue on the event loop."""
        loop = self._loop
        if loop is None:
            return
        for out in outs:
            out_q = self._queues.get(out.seq_id)
            if out_q is None:  # already cancelled
                continue
            try:
                loop.call_soon_threadsafe(out_q.put_nowait, out)
            except RuntimeError:  # loop closed during shutdown
                return
