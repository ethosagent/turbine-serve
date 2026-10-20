"""Parity with transformers (tiny cpu fp32; real Qwen3-4B cuda bf16): logits and greedy tokens."""

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

from tests.model_cases import ModelCase, engine_logits, free_gpu_memory, top2

MESSAGES = [{"role": "user", "content": "Explain the paged KV cache in two sentences."}]
TOL = 1e-4  # tiny fp32 logits; the real bf16 case compares argmax instead (not assert_close-stable)


@pytest.fixture(scope="module")
def hf_model(model_case: ModelCase) -> Iterator[Qwen3ForCausalLM]:
    """The transformers reference in the case's dtype; only one real copy is ever alive."""
    if model_case.name == "tiny":
        yield Qwen3ForCausalLM.from_pretrained(model_case.model_dir, dtype=torch.float32).eval()
        return
    free_gpu_memory()
    # .to("cuda") rather than device_map="cuda": device_map needs `accelerate`, not a dependency.
    model = Qwen3ForCausalLM.from_pretrained(model_case.model_dir, dtype=torch.bfloat16)
    model = model.to("cuda").eval()
    yield model
    del model
    free_gpu_memory()


@pytest.fixture(scope="module")
def prompt(model_engine: tuple[Engine, HFTokenizer]) -> list[int]:
    return model_engine[1].encode_chat(MESSAGES)


@pytest.fixture(scope="module")
def ref_ids(
    hf_model: Qwen3ForCausalLM,
    model_engine: tuple[Engine, HFTokenizer],
    model_case: ModelCase,
    prompt: list[int],
) -> list[int]:
    """HF greedy continuation of the prompt, `parity_tokens` tokens."""
    input_ids = torch.tensor([prompt], device=model_case.device)  # [1, P]
    with torch.inference_mode():
        out = hf_model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=model_case.parity_tokens,
            do_sample=False,
            pad_token_id=model_engine[1].hf.eos_token_id,
        )  # [1, P + N]
    ids: list[int] = out[0, len(prompt) :].tolist()
    # Qwen3-4B answers this prompt in 63 tokens and stops on <|im_end|>; the engine must too.
    eos = model_engine[1].eos_token_ids
    assert len(ids) == model_case.parity_tokens or ids[-1] in eos
    return ids


def _hf_logits(hf_model: Qwen3ForCausalLM, ids: list[int], device: str) -> torch.Tensor:
    with torch.inference_mode():
        return hf_model(torch.tensor([ids], device=device)).logits[0].float()  # [T, V]


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
    ref = _hf_logits(hf_model, ids, model_case.device)  # [T, V]
    diff = (ours - ref).abs().amax(dim=-1)  # [T]
    print(f"[{model_case.name}] max abs logit diff {diff.max():.3e} over {len(ids)} positions")
    if model_case.name == "tiny":
        bad = torch.nonzero(diff > TOL)
        if bad.numel():
            pos = int(bad[0])
            print(f"first bad position {pos} (prompt len {len(prompt)}): diff {diff[pos]:.3e}")
        torch.testing.assert_close(ours, ref, atol=TOL, rtol=TOL)
        return
    agree = ours.argmax(dim=-1) == ref.argmax(dim=-1)  # [T]
    print(f"[real] argmax agreement {int(agree.sum())}/{len(ids)} (prompt len {len(prompt)})")
    for pos in torch.nonzero(~agree).flatten().tolist():
        print(f"[real] argmax mismatch at {pos}: ours {top2(ours[pos])} | hf {top2(ref[pos])}")
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
            hf_row = _hf_logits(hf_model, ctx, model_case.device)[-1]  # [V]
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
