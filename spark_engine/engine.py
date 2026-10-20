"""Block 4: composition root and the step loop (schedule -> execute -> sample -> update -> emit)."""

import queue
import threading
import time
from collections.abc import Callable

from spark_engine.interfaces import ModelRunner, Sampler, Scheduler
from spark_engine.log import get_logger
from spark_engine.types import (
    SamplingParams,
    SchedulerOutput,
    SeqStatus,
    Sequence,
    TokenOutput,
)

log = get_logger("engine")

_IDLE_WAIT_S = 0.05


def _ms(start: float, end: float) -> float:
    return (end - start) * 1000


class Engine:
    """Owns the step loop. Scheduler calls run under one lock; forward and sampling do not.

    add_request runs scheduler.add on the caller's thread (so ValueError reaches the caller);
    abort is queued and applied at the start of the next step.
    """

    def __init__(
        self,
        scheduler: Scheduler,
        runner: ModelRunner,
        sampler: Sampler,
        on_output: Callable[[list[TokenOutput]], None] | None = None,
    ) -> None:
        self._scheduler = scheduler
        self._runner = runner
        self._sampler = sampler
        self.on_output = on_output
        self._lock = threading.Lock()
        self._aborts: queue.Queue[str] = queue.Queue()
        self._added: list[Sequence] = []  # pruned to WAITING seqs; backs num_waiting
        self._wake = threading.Event()
        self._stopped = threading.Event()

    def add_request(self, seq: Sequence) -> None:
        """Thread-safe. Raises ValueError if the request can never fit in the KV cache."""
        seq.arrival_ts = time.monotonic()
        with self._lock:
            self._scheduler.add(seq)
            self._added.append(seq)
        self._wake.set()

    def abort(self, seq_id: str) -> None:
        """Thread-safe; applied at the start of the next step."""
        self._aborts.put(seq_id)
        self._wake.set()

    @property
    def num_waiting(self) -> int:
        with self._lock:
            self._prune_added()
            return len(self._added)

    def step(self) -> list[TokenOutput]:
        start = time.perf_counter()
        with self._lock:
            self._drain_aborts()
            sched = self._scheduler.schedule()
            self._prune_added()
        if not sched.seqs:
            return []
        outs = self._execute(sched, _ms(start, time.perf_counter()))
        try:
            self._emit(sched, outs)
        except Exception as exc:  # a consumer bug must not stop the engine loop
            log.exception("emit.failed", step_id=sched.step_id, exc=repr(exc))
        return outs

    def run_forever(self) -> None:
        """Step while there is work; idle-wait for add_request/abort otherwise. stop() ends it."""
        log.info("engine.started")
        while not self._stopped.is_set():
            self._wake.clear()
            with self._lock:
                busy = self._scheduler.has_work() or not self._aborts.empty()
            if busy:
                self.step()
            else:
                self._wake.wait(_IDLE_WAIT_S)
        log.info("engine.stopped")

    def stop(self) -> None:
        self._stopped.set()
        self._wake.set()

    def generate_batch(self, prompts: list[list[int]], params: SamplingParams) -> list[list[int]]:
        """Offline, synchronous; never call while run_forever is running."""
        seqs = [Sequence(f"s{i}", "offline", list(p), params) for i, p in enumerate(prompts)]
        for seq in seqs:
            self.add_request(seq)
        while True:
            with self._lock:
                if not self._scheduler.has_work():
                    break
            self.step()
        return [list(seq.output_ids) for seq in seqs]

    def generate(self, prompt_ids: list[int], params: SamplingParams) -> list[int]:
        return self.generate_batch([prompt_ids], params)[0]

    def _drain_aborts(self) -> None:
        while True:
            try:
                seq_id = self._aborts.get_nowait()
            except queue.Empty:
                return
            self._scheduler.abort(seq_id)

    def _prune_added(self) -> None:
        self._added = [s for s in self._added if s.status is SeqStatus.WAITING]

    def _execute(self, sched: SchedulerOutput, schedule_ms: float) -> list[TokenOutput]:
        """Forward + sample + update; any exception fails only this step's sequences."""
        seqs = [item.seq for item in sched.seqs]
        try:
            t0 = time.perf_counter()
            logits = self._runner.execute(sched)
            t1 = time.perf_counter()
            tokens = self._sampler.sample(logits, seqs)
            t2 = time.perf_counter()
            if len(tokens) != len(seqs):  # check before update mutates any seq
                raise RuntimeError(f"sampler returned {len(tokens)} tokens for {len(seqs)} seqs")
            with self._lock:
                outs = self._scheduler.update(sched, tokens)
        except Exception as exc:
            log.exception("step.failed", step_id=sched.step_id, num_seqs=len(seqs), exc=repr(exc))
            with self._lock:
                failed = [self._scheduler.fail(seq.seq_id, repr(exc)) for seq in seqs]
            return [out for out in failed if out is not None]
        log.debug(
            "step.done",
            step_id=sched.step_id,
            num_seqs=len(seqs),
            num_tokens=sum(item.num_new_tokens for item in sched.seqs),
            schedule_ms=schedule_ms,
            forward_ms=_ms(t0, t1),
            sample_ms=_ms(t1, t2),
        )
        return outs

    def _emit(self, sched: SchedulerOutput, outs: list[TokenOutput]) -> None:
        """Stamp first-token time, log finished sequences, hand outputs to on_output."""
        now = time.monotonic()
        by_id = {item.seq.seq_id: item.seq for item in sched.seqs}
        for out in outs:
            seq = by_id[out.seq_id]
            if out.finish_reason != "error" and seq.first_token_ts is None:
                seq.first_token_ts = now
            if out.finished:
                self._log_finished(seq, out, now)
        if self.on_output is not None:
            self.on_output(outs)

    @staticmethod
    def _log_finished(seq: Sequence, out: TokenOutput, now: float) -> None:
        completion = len(seq.output_ids)
        first = seq.first_token_ts
        log.info(
            "seq.finished",
            request_id=seq.request_id,
            seq_id=seq.seq_id,
            finish_reason=out.finish_reason,
            prompt_tokens=len(seq.prompt_ids),
            completion_tokens=completion,
            ttft_ms=None if first is None else _ms(seq.arrival_ts, first),
            tpot_ms=None if first is None else _ms(first, now) / max(1, completion - 1),
        )
