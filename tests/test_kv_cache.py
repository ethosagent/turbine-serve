import pytest
import torch
from spark_engine.kv_cache import PagedKVCache, compute_num_blocks
from spark_engine.types import SamplingParams, Sequence

HKV, D, BS = 2, 8, 16


def make_cache(num_blocks: int = 4, num_layers: int = 2) -> PagedKVCache:
    return PagedKVCache(
        num_layers, HKV, D, BS, 0, "cpu", dtype=torch.float32, num_blocks=num_blocks
    )


def make_seq(seq_id: str = "s1") -> Sequence:
    return Sequence(seq_id=seq_id, request_id="r1", prompt_ids=[1, 2, 3], params=SamplingParams())


def test_budget_matches_qwen3_numbers() -> None:
    assert compute_num_blocks(36, 8, 128, 16, 30, 2) == 13653


def test_allocate_and_free_roundtrip() -> None:
    cache = make_cache()
    seq = make_seq()
    cache.allocate(seq, 20)  # 2 blocks of 16
    assert len(seq.block_table) == 2
    assert cache.num_free_blocks == 2
    cache.free(seq)
    assert cache.num_free_blocks == 4
    assert seq.block_table == []


def test_cannot_allocate_more_than_free() -> None:
    cache = make_cache(num_blocks=2)
    assert cache.can_allocate(32)
    assert not cache.can_allocate(33)
    seq = make_seq()
    with pytest.raises(ValueError):
        cache.allocate(seq, 33)
    assert cache.num_free_blocks == 2
    assert seq.block_table == []


def test_layer_cache_shape() -> None:
    cache = make_cache(num_blocks=4, num_layers=3)
    k, v = cache.layer_cache(1)
    assert k.shape == (4, BS, HKV, D)
    assert v.shape == (4, BS, HKV, D)
    assert cache.block_size == BS
    assert cache.num_blocks == 4


def test_free_is_idempotent() -> None:
    cache = make_cache()
    seq = make_seq()
    cache.allocate(seq, 16)
    cache.free(seq)
    cache.free(seq)
    assert cache.num_free_blocks == 4


def test_allocate_extends_block_table() -> None:
    cache = make_cache()
    seq = make_seq()
    cache.allocate(seq, 16)
    first = list(seq.block_table)
    cache.allocate(seq, 17)
    assert seq.block_table[:1] == first
    assert len(seq.block_table) == 3
    assert len(set(seq.block_table)) == 3


def test_slot_write_then_gather_roundtrip() -> None:
    cache = make_cache()
    seq = make_seq()
    cache.allocate(seq, 3)
    b = seq.block_table[0]
    k, _ = cache.layer_cache(0)
    data = torch.randn(3, HKV, D)  # [3, Hkv, D]
    slots = torch.tensor([b * BS + 0, b * BS + 1, b * BS + 2])  # [3]
    k.view(-1, HKV, D)[slots] = data
    assert torch.equal(cache.layer_cache(0)[0][b, :3], data)
