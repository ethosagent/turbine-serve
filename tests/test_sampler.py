"""Sampler tests: greedy, seeded reproducibility, top-k / top-p filtering, row independence."""

import torch
from spark_engine.sampler import TorchSampler, draw_seed
from spark_engine.types import SamplingParams, Sequence

V = 16
DRAWS = 200


def _seq(params: SamplingParams, n_prompt: int = 3) -> Sequence:
    return Sequence(seq_id="s", request_id="r", prompt_ids=list(range(n_prompt)), params=params)


def _gap_logits(top: list[int]) -> torch.Tensor:
    row = torch.zeros(1, V)  # [1, V]
    row[0, top] = 10.0
    return row


def _draws(logits: torch.Tensor, params: SamplingParams) -> set[int]:
    sampler = TorchSampler()
    out: set[int] = set()
    for i in range(DRAWS):
        seq = _seq(params, n_prompt=i + 1)  # a new position changes the draw seed
        out.add(sampler.sample(logits, [seq])[0])
    return out


def test_draw_seed_formula() -> None:
    assert draw_seed(2, 5) == (2 * 1_000_003 + 5) & (2**63 - 1)


def test_greedy_is_argmax() -> None:
    logits = torch.randn(4, V)  # [4, V]
    seqs = [_seq(SamplingParams(temperature=0.0)) for _ in range(4)]
    out = TorchSampler().sample(logits, seqs)
    assert out == logits.argmax(dim=-1).tolist()
    assert all(type(t) is int for t in out)


def test_seeded_draw_is_reproducible() -> None:
    logits = torch.randn(1, V)  # [1, V]
    params = SamplingParams(temperature=1.0, top_p=1.0, top_k=0, seed=7)
    a = TorchSampler().sample(logits, [_seq(params)])
    b = TorchSampler().sample(logits, [_seq(params)])
    assert a == b


def test_seed_changes_with_position() -> None:
    logits = torch.zeros(1, V)  # [1, V] uniform
    params = SamplingParams(temperature=1.0, top_p=1.0, top_k=0, seed=7)
    sampler = TorchSampler()
    picks = {sampler.sample(logits, [_seq(params, n_prompt=n)])[0] for n in range(1, 40)}
    assert len(picks) > 1


def test_top_k_one_equals_argmax() -> None:
    logits = torch.randn(1, V)  # [1, V]
    params = SamplingParams(temperature=1.0, top_p=1.0, top_k=1, seed=3)
    assert TorchSampler().sample(logits, [_seq(params)]) == [int(logits.argmax())]


def test_top_k_never_picks_outside_k() -> None:
    top = [2, 5, 9]
    params = SamplingParams(temperature=1.0, top_p=1.0, top_k=3, seed=11)
    picks = _draws(_gap_logits(top), params)
    assert picks <= set(top)
    assert len(picks) > 1


def test_top_p_restricts_to_nucleus() -> None:
    logits = torch.full((1, V), -20.0)  # [1, V]
    logits[0, 4] = 5.0
    logits[0, 8] = 4.0  # p(4) ~ 0.73, p(8) ~ 0.27
    params = SamplingParams(temperature=1.0, top_p=0.5, top_k=0, seed=13)
    assert _draws(logits, params) == {4}  # the crossing token stays; the next one is dropped


def test_batch_rows_are_independent() -> None:
    p = SamplingParams(temperature=1.0, top_p=1.0, top_k=0, seed=5)
    g = SamplingParams(temperature=0.0)
    logits = torch.randn(2, V)  # [2, V]
    sampler = TorchSampler()
    both = sampler.sample(logits, [_seq(p), _seq(g)])
    assert both[0] == sampler.sample(logits[0:1], [_seq(p)])[0]
    assert both[1] == int(logits[1].argmax())
