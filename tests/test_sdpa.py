import math

import pytest
import torch
from spark_engine.attention import ATTENTION_BACKENDS, get_attention_backend
from spark_engine.attention.sdpa import SDPAAttention
from spark_engine.types import SamplingParams, ScheduledSeq, SchedulerOutput, Sequence
from torch import Tensor

HQ, HKV, D, BS = 4, 2, 8, 4
NUM_BLOCKS = 8


def make_seq(seq_id: str, num_computed: int, block_table: list[int]) -> Sequence:
    return Sequence(
        seq_id=seq_id,
        request_id="r",
        prompt_ids=[1],
        params=SamplingParams(),
        num_computed=num_computed,
        block_table=block_table,
    )


def make_caches() -> tuple[Tensor, Tensor]:
    k = torch.randn(NUM_BLOCKS, BS, HKV, D)  # [NB, bs, Hkv, D]
    v = torch.randn(NUM_BLOCKS, BS, HKV, D)  # [NB, bs, Hkv, D]
    return k, v


def reference(q: Tensor, k: Tensor, v: Tensor, causal: bool) -> Tensor:
    """q [q_len, Hq, D]; k, v [ctx, Hkv, D] -> [q_len, Hq, D]; explicit masked softmax."""
    rep = HQ // HKV
    k = k.repeat_interleave(rep, dim=1)  # [ctx, Hq, D]
    v = v.repeat_interleave(rep, dim=1)  # [ctx, Hq, D]
    qh = q.transpose(0, 1)  # [Hq, q_len, D]
    kh = k.transpose(0, 1)  # [Hq, ctx, D]
    vh = v.transpose(0, 1)  # [Hq, ctx, D]
    scores = qh @ kh.transpose(1, 2) / math.sqrt(D)  # [Hq, q_len, ctx]
    if causal:
        mask = torch.triu(torch.ones(q.shape[0], k.shape[0], dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))  # [Hq, q_len, ctx]
    probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)  # [Hq, q_len, ctx]
    return (probs @ vh).transpose(0, 1)  # [q_len, Hq, D]


def gather(cache: Tensor, table: list[int], ctx: int) -> Tensor:
    return cache[torch.tensor(table)].reshape(-1, HKV, D)[:ctx]  # [ctx, Hkv, D]


def run(seqs: list[tuple[Sequence, int]], q: Tensor, k_cache: Tensor, v_cache: Tensor) -> Tensor:
    backend = SDPAAttention(HQ, HKV, D)
    sched = SchedulerOutput(0, [ScheduledSeq(s, n) for s, n in seqs])
    meta = backend.build_metadata(sched, BS)
    return backend.forward(q, k_cache, v_cache, meta)


def test_prefill_single_sequence_matches_reference() -> None:
    torch.manual_seed(0)
    k_cache, v_cache = make_caches()
    table = [5, 1, 3]
    seq = make_seq("a", 0, table)
    q = torch.randn(10, HQ, D)  # [T, Hq, D]
    out = run([(seq, 10)], q, k_cache, v_cache)
    expected = reference(q, gather(k_cache, table, 10), gather(v_cache, table, 10), True)
    assert out.shape == (10, HQ, D)
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


def test_decode_reads_full_context() -> None:
    torch.manual_seed(1)
    k_cache, v_cache = make_caches()
    table = [2, 6, 0]
    seq = make_seq("a", 9, table)
    q = torch.randn(1, HQ, D)  # [1, Hq, D]
    out = run([(seq, 1)], q, k_cache, v_cache)
    expected = reference(q, gather(k_cache, table, 10), gather(v_cache, table, 10), False)
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


def test_mixed_batch_prefill_and_decode() -> None:
    torch.manual_seed(2)
    k_cache, v_cache = make_caches()
    pre = make_seq("p", 0, [4, 7])
    dec = make_seq("d", 6, [1, 2])
    q = torch.randn(6, HQ, D)  # [T, Hq, D]: 5 prefill + 1 decode
    out = run([(pre, 5), (dec, 1)], q, k_cache, v_cache)
    exp_pre = reference(q[:5], gather(k_cache, [4, 7], 5), gather(v_cache, [4, 7], 5), True)
    exp_dec = reference(q[5:], gather(k_cache, [1, 2], 7), gather(v_cache, [1, 2], 7), False)
    torch.testing.assert_close(out[:5], exp_pre, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out[5:], exp_dec, atol=1e-5, rtol=1e-5)


def test_metadata_offsets() -> None:
    backend = SDPAAttention(HQ, HKV, D)
    sched = SchedulerOutput(
        0, [ScheduledSeq(make_seq("p", 0, [4, 7]), 5), ScheduledSeq(make_seq("d", 6, [1, 2]), 1)]
    )
    meta = backend.build_metadata(sched, BS)
    assert meta.q_start == [0, 5]
    assert meta.q_len == [5, 1]
    assert meta.ctx_len == [5, 7]
    assert meta.block_tables == [[4, 7], [1, 2]]


def test_gqa_head_mapping() -> None:
    """Q heads 0,1 must read KV head 0; Q heads 2,3 must read KV head 1."""
    k_cache = torch.zeros(NUM_BLOCKS, BS, HKV, D)  # [NB, bs, Hkv, D]
    v_cache = torch.zeros(NUM_BLOCKS, BS, HKV, D)  # [NB, bs, Hkv, D]
    v_cache[:, :, 0, :] = 1.0
    v_cache[:, :, 1, :] = 2.0
    seq = make_seq("a", 3, [0])
    q = torch.randn(1, HQ, D)  # [1, Hq, D]
    out = run([(seq, 1)], q, k_cache, v_cache)
    expected = torch.tensor([1.0, 1.0, 2.0, 2.0]).view(1, HQ, 1).expand(1, HQ, D)
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


def test_rejects_partial_prefill() -> None:
    k_cache, v_cache = make_caches()
    seq = make_seq("a", 4, [0, 1])  # 4 computed + 3 new: neither decode nor full prefill
    q = torch.randn(3, HQ, D)  # [T, Hq, D]
    with pytest.raises(AssertionError, match="q_len"):
        run([(seq, 3)], q, k_cache, v_cache)


def test_registry_resolves_sdpa() -> None:
    assert ATTENTION_BACKENDS == {"sdpa": SDPAAttention}
    assert get_attention_backend("sdpa") is SDPAAttention


def test_registry_unknown_name_lists_known() -> None:
    with pytest.raises(ValueError, match="unknown attention backend 'nope'; known: \\['sdpa'\\]"):
        get_attention_backend("nope")
