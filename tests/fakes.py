"""Test doubles for the control plane: a deterministic ModelRunner and an engine builder."""

import torch
from torch import Tensor
from turbine_serve.engine import Engine
from turbine_serve.kv_cache import PagedKVCache
from turbine_serve.sampler import TorchSampler
from turbine_serve.scheduler import FCFSScheduler
from turbine_serve.types import SchedulerOutput


class FakeRunner:
    """Logits peak at (last token + 1) % vocab, so greedy decoding counts up."""

    def __init__(self, vocab: int = 16, fail_on_step: int | None = None) -> None:
        self.vocab = vocab
        self.fail_on_step = fail_on_step

    def execute(self, sched: SchedulerOutput) -> Tensor:
        if sched.step_id == self.fail_on_step:
            raise RuntimeError("boom")
        logits = torch.zeros(len(sched.seqs), self.vocab, dtype=torch.float32)  # [B, V]
        for row, item in enumerate(sched.seqs):
            seq = item.seq
            last = seq.output_ids[-1] if seq.output_ids else seq.prompt_ids[-1]
            logits[row, (last + 1) % self.vocab] = 1.0
        return logits


def make_engine(
    num_blocks: int = 64,
    max_num_seqs: int = 8,
    eos: tuple[int, ...] = (7,),
    runner: FakeRunner | None = None,
) -> tuple[Engine, PagedKVCache]:
    kv = PagedKVCache(1, 1, 1, 16, 0, "cpu", num_blocks=num_blocks)
    scheduler = FCFSScheduler(kv, max_num_seqs, eos)
    return Engine(scheduler, runner or FakeRunner(), TorchSampler()), kv
