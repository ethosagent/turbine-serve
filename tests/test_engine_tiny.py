"""Engine on the real tiny model: batch parity, KV hygiene, seeded sampling, step timings."""

import json
from pathlib import Path

import pytest
from spark_engine.config import EngineConfig
from spark_engine.engine import Engine, build_engine
from spark_engine.log import configure_logging
from spark_engine.tokenizer import HFTokenizer
from spark_engine.types import SamplingParams

GREEDY = SamplingParams(temperature=0, max_tokens=16)
QUESTIONS = [
    "Explain the paged KV cache in two sentences.",
    "What is RoPE?",
    "Why does Qwen3 apply RMSNorm to q and k?",
    "Name three open-source inference servers and one thing each does well.",
]


@pytest.fixture(scope="module")
def tiny_engine(tiny_model_dir: Path) -> tuple[Engine, HFTokenizer]:
    config = EngineConfig(model=str(tiny_model_dir), device="cpu", dtype="float32", kv_gb=0.01)
    return build_engine(config)


@pytest.fixture(scope="module")
def prompts(tiny_engine: tuple[Engine, HFTokenizer]) -> list[list[int]]:
    tokenizer = tiny_engine[1]
    return [tokenizer.encode_chat([{"role": "user", "content": q}]) for q in QUESTIONS]


def test_four_concurrent_match_single(
    tiny_engine: tuple[Engine, HFTokenizer], prompts: list[list[int]]
) -> None:
    engine, _ = tiny_engine
    singles = [engine.generate(p, GREEDY) for p in prompts]
    assert all(len(s) == GREEDY.max_tokens for s in singles)
    assert engine.generate_batch(prompts, GREEDY) == singles


def test_kv_all_free_after_batch_and_rerun_identical(
    tiny_engine: tuple[Engine, HFTokenizer], prompts: list[list[int]]
) -> None:
    engine, _ = tiny_engine
    kv = engine._runner.kv  # the cache is wired privately; the invariant needs it
    first = engine.generate_batch(prompts, GREEDY)
    assert kv.num_free_blocks == kv.num_blocks
    assert engine.generate_batch(prompts, GREEDY) == first
    assert kv.num_free_blocks == kv.num_blocks


def test_seeded_sampling_reproducible(
    tiny_engine: tuple[Engine, HFTokenizer], prompts: list[list[int]]
) -> None:
    engine, _ = tiny_engine
    seed7 = SamplingParams(temperature=0.7, max_tokens=16, seed=7)
    seed8 = SamplingParams(temperature=0.7, max_tokens=16, seed=8)
    first = engine.generate(prompts[0], seed7)
    assert engine.generate(prompts[0], seed7) == first
    assert engine.generate(prompts[0], seed8) != first


def test_step_done_logged_with_timings(
    tiny_engine: tuple[Engine, HFTokenizer], prompts: list[list[int]], tmp_path: Path
) -> None:
    engine, _ = tiny_engine
    log_file = tmp_path / "engine.jsonl"
    configure_logging("debug", str(log_file))
    try:
        engine.generate(prompts[0], SamplingParams(temperature=0, max_tokens=4))
    finally:
        configure_logging("info", None)
    lines = [json.loads(line) for line in log_file.read_text().splitlines()]
    steps = [line for line in lines if line["event"] == "step.done"]
    assert [s["num_tokens"] for s in steps] == [len(prompts[0]), 1, 1, 1]
    for step in steps:
        assert step["component"] == "engine"
        assert step["forward_ms"] > 0
        assert step["sample_ms"] >= 0
        assert step["schedule_ms"] >= 0
    prepared = [line for line in lines if line["event"] == "batch.prepared"]
    assert [p["num_tokens"] for p in prepared] == [len(prompts[0]), 1, 1, 1]
    assert all(p["component"] == "model_runner" for p in prepared)
