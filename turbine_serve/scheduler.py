"""Block 5: WAITING/RUNNING queues, FCFS admission with KV reservation, EOS/max_tokens finish."""

from collections import deque

from turbine_serve.interfaces import KVCacheManager
from turbine_serve.log import get_logger
from turbine_serve.types import (
    ScheduledSeq,
    SchedulerOutput,
    SeqStatus,
    Sequence,
    TokenOutput,
)

log = get_logger("scheduler")


class FCFSScheduler:
    """Strict first-come-first-served; reserves prompt + max_tokens of KV at admission."""

    def __init__(
        self, kv: KVCacheManager, max_num_seqs: int, eos_token_ids: tuple[int, ...]
    ) -> None:
        self._kv = kv
        self._max_num_seqs = max_num_seqs
        self._eos = frozenset(eos_token_ids)
        self._waiting: deque[Sequence] = deque()
        self._running: deque[Sequence] = deque()
        self._by_id: dict[str, Sequence] = {}
        self._step_id = 0

    @property
    def num_waiting(self) -> int:
        return len(self._waiting)

    @property
    def num_running(self) -> int:
        return len(self._running)

    def has_work(self) -> bool:
        return self.num_waiting + self.num_running > 0

    @staticmethod
    def _reserve_tokens(seq: Sequence) -> int:
        return len(seq.prompt_ids) + seq.params.max_tokens

    def add(self, seq: Sequence) -> None:
        if self._kv.blocks_needed(self._reserve_tokens(seq)) > self._kv.num_blocks:
            raise ValueError("prompt + max_tokens exceeds KV capacity")
        seq.status = SeqStatus.WAITING
        self._waiting.append(seq)
        self._by_id[seq.seq_id] = seq

    def schedule(self) -> SchedulerOutput:
        self._step_id += 1
        scheduled = [ScheduledSeq(seq, 1) for seq in self._running]
        while self._waiting:
            seq = self._waiting[0]
            if self.num_running >= self._max_num_seqs:
                log.debug("seq.waiting", seq_id=seq.seq_id, reason="max_seqs")
                break
            if not self._kv.can_allocate(self._reserve_tokens(seq)):
                log.debug("seq.waiting", seq_id=seq.seq_id, reason="kv_full")
                break
            self._waiting.popleft()
            self._kv.allocate(seq, self._reserve_tokens(seq))
            seq.status = SeqStatus.RUNNING
            self._running.append(seq)
            scheduled.append(ScheduledSeq(seq, len(seq.prompt_ids)))
            log.info(
                "seq.admitted",
                request_id=seq.request_id,
                seq_id=seq.seq_id,
                step_id=self._step_id,
                blocks_reserved=len(seq.block_table),
                waiting=self.num_waiting,
                running=self.num_running,
            )
        return SchedulerOutput(self._step_id, scheduled)

    def update(self, out: SchedulerOutput, tokens: list[int]) -> list[TokenOutput]:
        """tokens[i] is the sampled token for out.seqs[i]; blocks were reserved at admission."""
        results: list[TokenOutput] = []
        for item, token in zip(out.seqs, tokens, strict=True):
            seq = item.seq
            if seq.status is not SeqStatus.RUNNING:  # aborted/failed after schedule()
                continue
            seq.output_ids.append(token)
            seq.num_computed += item.num_new_tokens
            reason: str | None = None
            if token in self._eos:
                reason = "stop"
            elif len(seq.output_ids) >= seq.params.max_tokens:
                reason = "length"
            if reason is not None:
                self._finish(seq, reason)
            results.append(
                TokenOutput(seq.request_id, seq.seq_id, token, reason is not None, reason)
            )
        return results

    def abort(self, seq_id: str) -> None:
        seq = self._by_id.get(seq_id)
        if seq is not None:
            self._finish(seq, "abort")

    def fail(self, seq_id: str, error: str) -> TokenOutput | None:
        seq = self._by_id.get(seq_id)
        if seq is None:
            return None
        log.error("seq.failed", request_id=seq.request_id, seq_id=seq_id, error=error)
        self._finish(seq, "error")
        return TokenOutput(seq.request_id, seq.seq_id, -1, True, "error")

    def _finish(self, seq: Sequence, reason: str) -> None:
        if seq.status is SeqStatus.RUNNING:
            self._running.remove(seq)
        else:
            self._waiting.remove(seq)
        del self._by_id[seq.seq_id]
        self._kv.free(seq)
        seq.status = SeqStatus.FINISHED
        seq.finish_reason = reason
        log.info(
            "seq.finished",
            request_id=seq.request_id,
            seq_id=seq.seq_id,
            finish_reason=reason,
            prompt_tokens=len(seq.prompt_ids),
            completion_tokens=len(seq.output_ids),
        )
