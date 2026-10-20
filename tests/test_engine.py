"""Engine: step loop over the FakeRunner, per-step failure isolation, thread lifecycle."""

import json
import threading
from pathlib import Path

import pytest
import torch
from spark_engine.engine import Engine
from spark_engine.kv_cache import PagedKVCache
from spark_engine.log import configure_logging
from spark_engine.sampler import TorchSampler
from spark_engine.scheduler import FCFSScheduler
from spark_engine.types import SamplingParams, Sequence, TokenOutput

from tests.fakes import FakeRunner, make_engine

GREEDY = SamplingParams(temperature=0, max_tokens=10)


def make_seq(seq_id: str, prompt: list[int], params: SamplingParams = GREEDY) -> Sequence:
    return Sequence(seq_id=seq_id, request_id=f"req-{seq_id}", prompt_ids=prompt, params=params)


def test_greedy_counts_up_until_eos() -> None:
    engine, _ = make_engine()
    assert engine.generate([1, 2], GREEDY) == [3, 4, 5, 6, 7]


def test_length_stop() -> None:
    engine, _ = make_engine()
    outs: list[TokenOutput] = []
    engine.on_output = outs.extend
    params = SamplingParams(temperature=0, max_tokens=3)
    assert engine.generate([10], params) == [11, 12, 13]
    assert (outs[-1].finished, outs[-1].finish_reason) == (True, "length")


def test_batch_matches_single() -> None:
    prompts = [[1, 2], [8, 9], [3], [12, 13, 14]]
    singles = [make_engine()[0].generate(p, GREEDY) for p in prompts]
    engine, _ = make_engine()
    assert engine.generate_batch(prompts, GREEDY) == singles
    assert singles[3] == [15, 0, 1, 2, 3, 4, 5, 6, 7]


def test_step_failure_fails_only_that_step_seqs() -> None:
    # max_num_seqs=1: step 1 prefills a, step 2 decodes a only (b waits) and raises.
    engine, kv = make_engine(max_num_seqs=1, runner=FakeRunner(fail_on_step=2))
    outs: list[TokenOutput] = []
    engine.on_output = outs.extend
    a, b = make_seq("a", [1, 2]), make_seq("b", [1, 2])
    engine.add_request(a)
    engine.add_request(b)

    assert [(o.seq_id, o.token_id) for o in engine.step()] == [("a", 3)]
    failed = engine.step()
    assert failed == [TokenOutput("req-a", "a", -1, True, "error")]
    assert (a.finish_reason, b.finish_reason) == ("error", None)

    while engine.step():
        pass
    assert b.output_ids == [3, 4, 5, 6, 7]
    assert b.finish_reason == "stop"
    assert kv.num_free_blocks == kv.num_blocks
    assert [o.token_id for o in outs if o.seq_id == "a"] == [3, -1]


def test_kv_fully_freed_after_batch() -> None:
    engine, kv = make_engine(num_blocks=8)
    engine.generate_batch([[1], [2, 3], [9, 10, 11]], GREEDY)
    assert kv.num_free_blocks == kv.num_blocks


def test_on_output_receives_every_token() -> None:
    engine, _ = make_engine()
    outs: list[TokenOutput] = []
    engine.on_output = outs.extend
    results = engine.generate_batch([[1, 2], [4]], GREEDY)
    for seq_id, expected in zip(("s0", "s1"), results, strict=True):
        mine = [o for o in outs if o.seq_id == seq_id]
        assert [o.token_id for o in mine] == expected
        assert [o.finished for o in mine] == [False] * (len(mine) - 1) + [True]


def test_abort_is_applied_at_next_step() -> None:
    engine, kv = make_engine()
    seq = make_seq("a", [1, 2])
    engine.add_request(seq)
    assert engine.num_waiting == 1
    engine.step()
    assert engine.num_waiting == 0
    engine.abort("a")
    assert engine.step() == []
    assert seq.finish_reason == "abort"
    assert kv.num_free_blocks == kv.num_blocks


def test_run_forever_and_stop_thread() -> None:
    engine, _ = make_engine()
    done = threading.Event()
    outs: list[TokenOutput] = []

    def collect(batch: list[TokenOutput]) -> None:
        outs.extend(batch)
        if any(o.finished for o in batch):
            done.set()

    engine.on_output = collect
    thread = threading.Thread(target=engine.run_forever, daemon=True)
    thread.start()
    try:
        engine.add_request(make_seq("a", [1, 2]))
        assert done.wait(2.0)
        assert [o.token_id for o in outs] == [3, 4, 5, 6, 7]
    finally:
        engine.stop()
        thread.join(2.0)
    assert not thread.is_alive()


def test_on_output_error_does_not_end_run_forever() -> None:
    engine, _ = make_engine()
    done = threading.Event()
    calls = 0

    def flaky(batch: list[TokenOutput]) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("consumer bug")
        if any(o.finished and o.seq_id == "b" for o in batch):
            done.set()

    engine.on_output = flaky
    thread = threading.Thread(target=engine.run_forever, daemon=True)
    thread.start()
    try:
        engine.add_request(make_seq("a", [1, 2]))
        engine.add_request(make_seq("b", [4]))
        assert done.wait(2.0)
        assert thread.is_alive()
    finally:
        engine.stop()
        thread.join(2.0)
    assert not thread.is_alive()


class ShortSampler(TorchSampler):
    """Returns one token too few."""

    def sample(self, logits: torch.Tensor, seqs: list[Sequence]) -> list[int]:
        return super().sample(logits, seqs)[:-1]


def test_token_count_mismatch_fails_every_scheduled_seq() -> None:
    kv = PagedKVCache(1, 1, 1, 16, 0, "cpu", num_blocks=64)
    engine = Engine(FCFSScheduler(kv, 8, (7,)), FakeRunner(), ShortSampler())
    a, b = make_seq("a", [1, 2]), make_seq("b", [4])
    engine.add_request(a)
    engine.add_request(b)
    outs = engine.step()
    assert outs == [
        TokenOutput("req-a", "a", -1, True, "error"),
        TokenOutput("req-b", "b", -1, True, "error"),
    ]
    assert (a.output_ids, b.output_ids) == ([], [])
    assert kv.num_free_blocks == kv.num_blocks


def test_impossible_request_raises_on_add() -> None:
    engine, _ = make_engine(num_blocks=1)
    big = make_seq("a", [1, 2], SamplingParams(temperature=0, max_tokens=100))
    with pytest.raises(ValueError, match="KV capacity"):
        engine.add_request(big)
    assert engine.num_waiting == 0


def test_finish_logs_seq_completed_once_with_latency(tmp_path: Path) -> None:
    log_file = tmp_path / "engine.jsonl"
    engine, _ = make_engine()
    configure_logging("info", str(log_file))
    try:
        engine.generate([1, 2], GREEDY)
    finally:
        configure_logging("info", None)
    lines = [json.loads(line) for line in log_file.read_text().splitlines()]
    completed = [line for line in lines if line["event"] == "seq.completed"]
    assert len(completed) == 1 and completed[0]["component"] == "engine"
    assert completed[0]["ttft_ms"] is not None and completed[0]["tpot_ms"] is not None
    finished = [line for line in lines if line["event"] == "seq.finished"]
    assert [line["component"] for line in finished] == ["scheduler"]
