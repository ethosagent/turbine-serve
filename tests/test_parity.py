"""Parity with transformers, the M2 gate (ARCHITECTURE.md §V): tiny cpu fp32 and real Qwen3-4B cuda fp32.

bf16 is not gated here; tests/test_bf16_report.py reports it.
"""

import re
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from spark_engine.engine import Engine
from spark_engine.loader import iter_weights
from spark_engine.models import get_model_cls
from spark_engine.models.qwen3 import Qwen3Model
from spark_engine.tokenizer import HFTokenizer
from spark_engine.types import SamplingParams
from transformers import AutoConfig, Qwen3ForCausalLM

from tests.model_cases import (
    PARITY_MESSAGES,
    ModelCase,
    engine_logits,
    free_gpu_memory,
    hf_greedy,
    hf_logits,
    load_hf,
    top2,
)

TOL = 1e-4  # tiny cpu fp32: same kernels as HF, so logits agree to fp32 noise
# Real cuda fp32: measured max abs logit diff 2.7e-2 over 87 positions (logits ~30, so ~1e-3
# relative; cuda GEMM/SDPA reduction order differs from HF's). 0.1 is ~4x headroom and still far
# below the 0.8 a bf16 run shows. The gate is exact argmax at every position plus this bound.
REAL_ATOL = 0.1


@pytest.fixture(scope="module")
def hf_model(model_case: ModelCase) -> Iterator[Qwen3ForCausalLM]:
    """The transformers reference in the case's dtype; only one real copy is ever alive."""
    if model_case.name == "tiny":
        yield load_hf(model_case)
        return
    free_gpu_memory()
    model = load_hf(model_case)
    yield model
    del model
    free_gpu_memory()


@pytest.fixture(scope="module")
def prompt(model_engine: tuple[Engine, HFTokenizer]) -> list[int]:
    return model_engine[1].encode_chat(PARITY_MESSAGES)


@pytest.fixture(scope="module")
def ref_ids(
    hf_model: Qwen3ForCausalLM,
    model_engine: tuple[Engine, HFTokenizer],
    model_case: ModelCase,
    prompt: list[int],
) -> list[int]:
    """HF greedy continuation of the prompt, `parity_tokens` tokens."""
    tokenizer = model_engine[1]
    n = model_case.parity_tokens
    ids = hf_greedy(hf_model, prompt, n, model_case.device, tokenizer.hf.eos_token_id)
    # Qwen3-4B answers this prompt in 63 tokens and stops on <|im_end|>; the engine must too.
    assert len(ids) == n or ids[-1] in tokenizer.eos_token_ids
    return ids


def test_teacher_forced_logits_match_hf(
    model_case: ModelCase,
    model_engine: tuple[Engine, HFTokenizer],
    hf_model: Qwen3ForCausalLM,
    prompt: list[int],
    ref_ids: list[int],
) -> None:
    ids = prompt + ref_ids
    ours = engine_logits(model_engine[0], ids)  # [T, V]
    assert ours.dtype == torch.float32
    ref = hf_logits(hf_model, ids, model_case.device)  # [T, V]
    diff = (ours - ref).abs().amax(dim=-1)  # [T]
    print(f"[{model_case.name}] max abs logit diff {diff.max():.3e} over {len(ids)} positions")
    tol = TOL if model_case.name == "tiny" else REAL_ATOL
    bad = torch.nonzero(diff > tol)
    if bad.numel():
        pos = int(bad[0])
        print(f"first bad position {pos} (prompt len {len(prompt)}): diff {diff[pos]:.3e}")
    agree = ours.argmax(dim=-1) == ref.argmax(dim=-1)  # [T]
    print(f"[{model_case.name}] argmax agreement {int(agree.sum())}/{len(ids)}")
    for pos in torch.nonzero(~agree).flatten().tolist():
        print(
            f"[{model_case.name}] argmax mismatch at {pos}: ours {top2(ours[pos])} | hf {top2(ref[pos])}"
        )
    if model_case.name == "tiny":
        torch.testing.assert_close(ours, ref, atol=TOL, rtol=TOL)
    else:
        torch.testing.assert_close(ours, ref, atol=REAL_ATOL, rtol=0)
    assert bool(agree.all()), "teacher-forced argmax differs from transformers"


def test_greedy_tokens_match_hf(
    model_case: ModelCase,
    model_engine: tuple[Engine, HFTokenizer],
    hf_model: Qwen3ForCausalLM,
    prompt: list[int],
    ref_ids: list[int],
) -> None:
    engine, _ = model_engine
    n = model_case.parity_tokens
    ours = engine.generate(prompt, SamplingParams(temperature=0, max_tokens=n))
    if ours != ref_ids:
        k = next((i for i, (a, b) in enumerate(zip(ours, ref_ids, strict=False)) if a != b), n)
        ctx = prompt + ref_ids[:k]
        if k < min(len(ours), len(ref_ids)):
            hf_row = hf_logits(hf_model, ctx, model_case.device)[-1]  # [V]
            our_row = engine_logits(engine, ctx)[-1]  # [V]
            a, b = ours[k], ref_ids[k]
            print(
                f"[{model_case.name}] greedy diverges at token {k}/{n}: ours {a} hf {b}; "
                f"hf logits {a}={hf_row[a]:.4f} {b}={hf_row[b]:.4f}; "
                f"our logits {a}={our_row[a]:.4f} {b}={our_row[b]:.4f}"
            )
    else:
        print(f"[{model_case.name}] greedy {len(ours)}/{n} tokens match transformers")
    assert ours == ref_ids


def test_missing_lm_head_falls_back_to_embedding(tiny_model_dir: Path) -> None:
    cfg = AutoConfig.from_pretrained(str(tiny_model_dir))
    model = Qwen3Model(cfg, "cpu", torch.float32)
    weights = dict(iter_weights(str(tiny_model_dir), "cpu", torch.float32))
    assert "lm_head.weight" not in weights
    model.load_weights(weights.items())
    assert model.weights["lm_head.weight"] is model.weights["model.embed_tokens.weight"]


def test_load_weights_missing_names_raise_keyerror(tiny_model_dir: Path) -> None:
    cfg = AutoConfig.from_pretrained(str(tiny_model_dir))
    model = Qwen3Model(cfg, "cpu", torch.float32)
    partial = [(n, t) for n, t in iter_weights(str(tiny_model_dir), "cpu") if "layers.1" not in n]
    with pytest.raises(KeyError, match=re.escape("model.layers.1.self_attn.q_norm.weight")):
        model.load_weights(partial)


def test_registry_resolves_qwen3_and_eos(tiny_engine: tuple[Engine, HFTokenizer]) -> None:
    engine, tokenizer = tiny_engine
    assert get_model_cls("qwen3") is Qwen3Model
    with pytest.raises(ValueError, match="llama"):
        get_model_cls("llama")
    assert tokenizer.eos_token_ids == (151645, 151643)
    # build_engine wiring: the scheduler must stop on the tokenizer's EOS ids, not the config's.
    assert engine._scheduler._eos == frozenset((151645, 151643))
