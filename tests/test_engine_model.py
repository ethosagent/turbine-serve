"""Engine on a real checkpoint (tiny cpu fp32; Qwen3-4B cuda fp32): batching, KV, sampling, logs."""

import json
from pathlib import Path

import pytest
from turbine_serve.engine import Engine
from turbine_serve.log import configure_logging
from turbine_serve.tokenizer import HFTokenizer
from turbine_serve.types import SamplingParams

from tests.model_cases import ModelCase, engine_logits, top2

GREEDY = SamplingParams(temperature=0, max_tokens=16)
QUESTIONS = [
    "Explain the paged KV cache in two sentences.",
    "What is RoPE?",
    "Why does Qwen3 apply RMSNorm to q and k?",
    "Name three open-source inference servers and one thing each does well.",
]


@pytest.fixture(scope="module")
def prompts(model_engine: tuple[Engine, HFTokenizer]) -> list[list[int]]:
    tokenizer = model_engine[1]
    return [tokenizer.encode_chat([{"role": "user", "content": q}]) for q in QUESTIONS]


def test_four_concurrent_match_single(
    model_case: ModelCase, model_engine: tuple[Engine, HFTokenizer], prompts: list[list[int]]
) -> None:
    engine, _ = model_engine
    singles = [engine.generate(p, GREEDY) for p in prompts]
    assert all(len(s) == GREEDY.max_tokens for s in singles)
    batched = engine.generate_batch(prompts, GREEDY)
    for i, (b, s) in enumerate(zip(batched, singles, strict=True)):
        k = next((j for j, (x, y) in enumerate(zip(b, s, strict=True)) if x != y), None)
        if k is None:
            continue
        row = engine_logits(engine, prompts[i] + s[:k])[-1]  # [V] single-seq logits at k
        print(
            f"[{model_case.name}] seq {i} batch/single diverge at token {k}: batch {b[k]} "
            f"single {s[k]}; single-seq logits {b[k]}={row[b[k]]:.4f} "
            f"{s[k]}={row[s[k]]:.4f} (top2 {top2(row)})"
        )
    assert batched == singles


def test_kv_all_free_after_batch_and_rerun_identical(
    model_engine: tuple[Engine, HFTokenizer], prompts: list[list[int]]
) -> None:
    engine, _ = model_engine
    kv = engine._runner.kv  # the cache is wired privately; the invariant needs it
    first = engine.generate_batch(prompts, GREEDY)
    assert kv.num_free_blocks == kv.num_blocks
    assert engine.generate_batch(prompts, GREEDY) == first
    assert kv.num_free_blocks == kv.num_blocks


def test_seeded_sampling_reproducible(
    model_engine: tuple[Engine, HFTokenizer], prompts: list[list[int]]
) -> None:
    engine, _ = model_engine
    seed7 = SamplingParams(temperature=0.7, max_tokens=16, seed=7)
    seed8 = SamplingParams(temperature=0.7, max_tokens=16, seed=8)
    first = engine.generate(prompts[0], seed7)
    assert engine.generate(prompts[0], seed7) == first
    assert engine.generate(prompts[0], seed8) != first


def test_step_done_logged_with_timings(
    model_case: ModelCase,
    model_engine: tuple[Engine, HFTokenizer],
    prompts: list[list[int]],
    tmp_path: Path,
) -> None:
    engine, _ = model_engine
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
    decode_ms = [round(s["forward_ms"], 2) for s in steps[1:]]
    print(
        f"[{model_case.name}] prefill {len(prompts[0])} tokens forward_ms "
        f"{steps[0]['forward_ms']:.2f}; 1-token decode forward_ms {decode_ms}"
    )


def test_real_tokenizer_eos_ids(real_model_dir: str) -> None:
    """The real checkpoint's tokenizer stops on both <|im_end|> and <|endoftext|>."""
    assert HFTokenizer(real_model_dir, 8192).eos_token_ids == (151645, 151643)
