"""FCFSScheduler: admission with KV reservation, EOS/length finish, abort/fail."""

import pytest
from spark_engine.kv_cache import PagedKVCache
from spark_engine.scheduler import FCFSScheduler
from spark_engine.types import SamplingParams, SeqStatus, Sequence

EOS = (7, 11)


def make_kv(num_blocks: int = 8) -> PagedKVCache:
    return PagedKVCache(1, 1, 1, 16, 0, "cpu", num_blocks=num_blocks)


def make_seq(seq_id: str, prompt_len: int = 4, max_tokens: int = 4) -> Sequence:
    return Sequence(
        seq_id=seq_id,
        request_id=f"req-{seq_id}",
        prompt_ids=list(range(100, 100 + prompt_len)),
        params=SamplingParams(max_tokens=max_tokens),
    )


def make_sched(kv: PagedKVCache | None = None, max_num_seqs: int = 4) -> FCFSScheduler:
    return FCFSScheduler(kv or make_kv(), max_num_seqs, EOS)


def test_prefill_then_decode_shapes() -> None:
    sched = make_sched()
    seq = make_seq("a", prompt_len=5, max_tokens=3)
    sched.add(seq)
    assert (sched.num_waiting, sched.num_running) == (1, 0)

    out = sched.schedule()
    assert [(s.seq.seq_id, s.num_new_tokens) for s in out.seqs] == [("a", 5)]
    assert seq.status is SeqStatus.RUNNING
    assert (sched.num_waiting, sched.num_running) == (0, 1)

    results = sched.update(out, [50])
    assert [(r.token_id, r.finished) for r in results] == [(50, False)]
    assert seq.num_computed == 5
    assert seq.output_ids == [50]

    out2 = sched.schedule()
    assert [(s.seq.seq_id, s.num_new_tokens) for s in out2.seqs] == [("a", 1)]
    sched.update(out2, [51])
    assert seq.num_computed == 6


def test_admission_reserves_prompt_plus_max_tokens() -> None:
    kv = make_kv(8)
    sched = make_sched(kv)
    seq = make_seq("a", prompt_len=10, max_tokens=30)  # 40 tokens -> 3 blocks
    sched.add(seq)
    sched.schedule()
    assert len(seq.block_table) == 3
    assert kv.num_free_blocks == 5


def test_eos_finishes_and_frees() -> None:
    kv = make_kv(8)
    sched = make_sched(kv)
    seq = make_seq("a")
    sched.add(seq)
    out = sched.schedule()
    [res] = sched.update(out, [7])
    assert res.finished and res.finish_reason == "stop" and res.token_id == 7
    assert (res.request_id, res.seq_id) == ("req-a", "a")
    assert seq.status is SeqStatus.FINISHED and seq.finish_reason == "stop"
    assert seq.output_ids == [7]  # EOS counts as a completion token
    assert kv.num_free_blocks == 8
    assert (sched.num_running, sched.has_work()) == (0, False)


def test_second_eos_id_also_stops() -> None:
    sched = make_sched()
    sched.add(make_seq("a"))
    [res] = sched.update(sched.schedule(), [11])
    assert res.finished and res.finish_reason == "stop"


def test_max_tokens_finishes_with_length() -> None:
    sched = make_sched()
    seq = make_seq("a", max_tokens=2)
    sched.add(seq)
    [r1] = sched.update(sched.schedule(), [50])
    assert not r1.finished
    [r2] = sched.update(sched.schedule(), [51])
    assert r2.finished and r2.finish_reason == "length"
    assert seq.finish_reason == "length"
    assert not sched.has_work()


def test_max_num_seqs_holds_back() -> None:
    sched = make_sched(max_num_seqs=1)
    a, b = make_seq("a"), make_seq("b")
    sched.add(a)
    sched.add(b)
    out = sched.schedule()
    assert [s.seq.seq_id for s in out.seqs] == ["a"]
    assert b.status is SeqStatus.WAITING
    assert (sched.num_waiting, sched.num_running) == (1, 1)

    sched.update(out, [7])  # a finishes
    out2 = sched.schedule()
    assert [s.seq.seq_id for s in out2.seqs] == ["b"]
    assert out2.seqs[0].num_new_tokens == 4


def test_kv_full_holds_back_in_fcfs_order() -> None:
    kv = make_kv(4)  # 64 tokens
    sched = make_sched(kv)
    big = make_seq("big", prompt_len=30, max_tokens=30)  # 60 tokens -> 4 blocks
    small = make_seq("small", prompt_len=2, max_tokens=2)  # would fit only if big skipped
    sched.add(make_seq("first", prompt_len=2, max_tokens=2))  # 1 block
    sched.add(big)
    sched.add(small)
    out = sched.schedule()
    # big does not fit after "first"; small must NOT jump the queue
    assert [s.seq.seq_id for s in out.seqs] == ["first"]
    assert (sched.num_waiting, sched.num_running) == (2, 1)
    assert small.status is SeqStatus.WAITING


def test_add_rejects_impossible_request() -> None:
    sched = make_sched(make_kv(2))  # 32 tokens
    with pytest.raises(ValueError, match="prompt \\+ max_tokens exceeds KV capacity"):
        sched.add(make_seq("a", prompt_len=30, max_tokens=3))
    assert sched.num_waiting == 0


def test_abort_frees_blocks() -> None:
    kv = make_kv(8)
    sched = make_sched(kv)
    seq = make_seq("a")
    sched.add(seq)
    sched.schedule()
    assert kv.num_free_blocks < 8
    sched.abort("a")
    assert kv.num_free_blocks == 8
    assert seq.status is SeqStatus.FINISHED and seq.finish_reason == "abort"
    assert not sched.has_work()
    sched.abort("a")  # unknown now: no-op
    sched.abort("nope")


def test_abort_waiting_sequence() -> None:
    sched = make_sched(max_num_seqs=1)
    sched.add(make_seq("a"))
    b = make_seq("b")
    sched.add(b)
    sched.schedule()
    sched.abort("b")
    assert b.finish_reason == "abort"
    assert (sched.num_waiting, sched.num_running) == (0, 1)


def test_fail_returns_error_output() -> None:
    kv = make_kv(8)
    sched = make_sched(kv)
    seq = make_seq("a")
    sched.add(seq)
    sched.schedule()
    res = sched.fail("a", "boom")
    assert res is not None
    assert (res.request_id, res.seq_id) == ("req-a", "a")
    assert res.finished and res.finish_reason == "error"
    assert seq.finish_reason == "error"
    assert kv.num_free_blocks == 8
    assert sched.fail("a", "again") is None
    assert sched.fail("nope", "x") is None


def test_step_id_increments() -> None:
    sched = make_sched()
    assert sched.schedule().step_id == 1  # empty output still carries a step id
    assert sched.schedule().step_id == 2
    assert sched.schedule().seqs == []
