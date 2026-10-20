"""Block 10: greedy / temperature -> top-k -> top-p, seeded by (seed, position)."""

import time

import torch
from torch import Tensor

from turbine_serve.log import get_logger
from turbine_serve.types import Sequence

log = get_logger("sampler")


def draw_seed(seed: int, position: int) -> int:
    """Per-draw RNG seed: the same (seed, position) always yields the same token."""
    return (seed * 1_000_003 + position) & (2**63 - 1)


def _filter(row: Tensor, temperature: float, top_k: int, top_p: float) -> Tensor:
    """Temperature-scale one [V] logits row, then mask with top-k and top-p."""
    row = row.float() / temperature  # [V]
    if 0 < top_k < row.numel():
        kth = torch.topk(row, top_k).values[-1]  # scalar: k-th largest logit
        row = row.masked_fill(row < kth, float("-inf"))  # [V]
    sorted_logits, order = torch.sort(row, descending=True)  # [V], [V]
    probs = torch.softmax(sorted_logits, dim=-1)  # [V]
    drop = (probs.cumsum(dim=-1) - probs) > top_p  # [V] the token crossing top_p stays
    drop_mask = torch.zeros_like(drop).scatter(0, order, drop)  # [V] back in vocab order
    return row.masked_fill(drop_mask, float("-inf"))  # [V]


class TorchSampler:
    def sample(self, logits: Tensor, seqs: list[Sequence]) -> list[int]:
        start = time.perf_counter()
        out: list[int] = []
        for row, seq in zip(logits, seqs, strict=True):  # row: [V]
            p = seq.params
            if p.temperature == 0:
                out.append(int(row.argmax().item()))
                continue
            probs = torch.softmax(_filter(row, p.temperature, p.top_k, p.top_p), dim=-1)  # [V]
            gen = None
            if p.seed is not None:
                gen = torch.Generator(device=logits.device)
                gen.manual_seed(draw_seed(p.seed, seq.num_tokens))
            out.append(int(torch.multinomial(probs, 1, generator=gen).item()))
        log.debug("sample.done", batch=len(seqs), ms=(time.perf_counter() - start) * 1000)
        return out
