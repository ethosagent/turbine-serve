"""bf16 agreement with transformers on Qwen3-4B: reported, never gated (ADR 0004).

M2 is gated on fp32-on-cuda parity (tests/test_parity.py). bf16 is never bit-identical to HF:
HF rounds RoPE (cos/sin cast to bf16, rotation multiplied in bf16) and SwiGLU (silu rounded to
bf16 before the multiply by up) to bf16 earlier than we do; we compute both in fp32 and cast
once. Near-ties (top-2 gap within a bf16 ulp) therefore flip either way. This test measures the
gap and writes it to stdout and logs/bf16_parity.json; it asserts only structural sanity (no NaN,
argmax agreement >= 90% of positions), so a near-tie never fails the suite.
"""

import json
import time
from pathlib import Path

import pytest
import torch
from spark_engine.engine import build_engine
from spark_engine.types import SamplingParams

from tests.model_cases import (
    PARITY_MESSAGES,
    ModelCase,
    engine_logits,
    free_gpu_memory,
    hf_greedy,
    hf_logits,
    load_hf,
)

pytestmark = [pytest.mark.gpu, pytest.mark.report]

REPORT = Path(__file__).resolve().parents[1] / "logs" / "bf16_parity.json"
MIN_AGREEMENT = 0.9  # structural sanity only: a cast-order or layout bug drops far below this
QUESTIONS = [
    "Explain the paged KV cache in two sentences.",
    "What is RoPE?",
    "Why does Qwen3 apply RMSNorm to q and k?",
    "Name three open-source inference servers and one thing each does well.",
]


def _first_divergence(a: list[int], b: list[int]) -> int | None:
    k = next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), None)
    if k is None and len(a) != len(b):
        return min(len(a), len(b))
    return k


def test_bf16_agreement_report(real_model_dir: str) -> None:
    case = ModelCase("real-bf16", real_model_dir, "cuda", "bfloat16", 4, 64)
    free_gpu_memory()
    engine, tokenizer = build_engine(case.config)
    prompt = tokenizer.encode_chat(PARITY_MESSAGES)
    hf = load_hf(case)
    try:
        ref_ids = hf_greedy(hf, prompt, case.parity_tokens, case.device, tokenizer.hf.eos_token_id)
        ids = prompt + ref_ids
        ref = hf_logits(hf, ids, case.device)  # [T, V]
    finally:
        del hf
        free_gpu_memory()
    ours = engine_logits(engine, ids)  # [T, V]
    agree = ours.argmax(dim=-1) == ref.argmax(dim=-1)  # [T]
    greedy = engine.generate(prompt, SamplingParams(temperature=0, max_tokens=case.parity_tokens))
    params = SamplingParams(temperature=0, max_tokens=16)
    prompts = [tokenizer.encode_chat([{"role": "user", "content": q}]) for q in QUESTIONS]
    singles = [engine.generate(p, params) for p in prompts]
    batched = engine.generate_batch(prompts, params)
    report = {
        "model": case.model_dir,
        "dtype": case.dtype,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "positions": len(ids),
        "prompt_len": len(prompt),
        "argmax_agree": int(agree.sum()),
        "argmax_agreement": float(agree.float().mean()),
        "mismatch_positions": torch.nonzero(~agree).flatten().tolist(),
        "max_abs_logit_diff": float((ours - ref).abs().max()),
        "greedy_tokens_ours": len(greedy),
        "greedy_tokens_hf": len(ref_ids),
        "greedy_first_divergence": _first_divergence(greedy, ref_ids),
        "batch_single_first_divergence": [
            _first_divergence(b, s) for b, s in zip(batched, singles, strict=True)
        ],
    }
    del engine
    free_gpu_memory()
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[bf16 report] {json.dumps(report)}")
    assert not bool(torch.isnan(ours).any()), "NaN in our bf16 logits"
    assert report["argmax_agreement"] >= MIN_AGREEMENT
    assert len(greedy) > 0
