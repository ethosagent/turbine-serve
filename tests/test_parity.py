"""CPU parity with transformers on the tiny Qwen3: teacher-forced logits and greedy tokens."""

import re
from pathlib import Path

import pytest
import torch
from spark_engine.attention.sdpa import SDPAAttention
from spark_engine.config import EngineConfig
from spark_engine.engine import Engine, build_engine
from spark_engine.kv_cache import PagedKVCache
from spark_engine.loader import iter_weights
from spark_engine.model_runner import GPUModelRunner
from spark_engine.models import get_model_cls
from spark_engine.models.qwen3 import Qwen3Model
from spark_engine.tokenizer import HFTokenizer
from spark_engine.types import SamplingParams, ScheduledSeq, SchedulerOutput, Sequence
from transformers import AutoConfig, Qwen3ForCausalLM

MESSAGES = [{"role": "user", "content": "Explain the paged KV cache in two sentences."}]
MAX_NEW = 16
TOL = 1e-4

Parts = tuple[Qwen3Model, PagedKVCache, SDPAAttention, GPUModelRunner]


@pytest.fixture(scope="module")
def tiny_engine(tiny_model_dir: Path) -> tuple[Engine, HFTokenizer]:
    config = EngineConfig(model=str(tiny_model_dir), device="cpu", dtype="float32", kv_gb=0.01)
    return build_engine(config)


@pytest.fixture(scope="module")
def tiny_parts(tiny_model_dir: Path) -> Parts:
    """The data-plane blocks on their own, so the teacher-forced test bypasses the scheduler."""
    cfg = AutoConfig.from_pretrained(str(tiny_model_dir))
    model = Qwen3Model(cfg, "cpu", torch.float32)
    model.load_weights(iter_weights(str(tiny_model_dir), "cpu", torch.float32))
    kv = PagedKVCache(
        cfg.num_hidden_layers,
        cfg.num_key_value_heads,
        cfg.head_dim,
        16,
        0,
        "cpu",
        dtype=torch.float32,
        num_blocks=16,
    )
    attn = SDPAAttention(cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim)
    return model, kv, attn, GPUModelRunner(model, kv, attn, "cpu")


@pytest.fixture(scope="module")
def hf_model(tiny_model_dir: Path) -> Qwen3ForCausalLM:
    return Qwen3ForCausalLM.from_pretrained(str(tiny_model_dir), dtype=torch.float32).eval()


@pytest.fixture(scope="module")
def prompt(tiny_engine: tuple[Engine, HFTokenizer]) -> list[int]:
    return tiny_engine[1].encode_chat(MESSAGES)


@pytest.fixture(scope="module")
def ref_ids(
    hf_model: Qwen3ForCausalLM, tiny_engine: tuple[Engine, HFTokenizer], prompt: list[int]
) -> list[int]:
    """HF greedy continuation of the prompt, MAX_NEW tokens."""
    input_ids = torch.tensor([prompt])  # [1, P]
    out = hf_model.generate(
        input_ids,
        max_new_tokens=MAX_NEW,
        do_sample=False,
        pad_token_id=tiny_engine[1].hf.eos_token_id,
    )  # [1, P + MAX_NEW]
    ids: list[int] = out[0, len(prompt) :].tolist()
    assert len(ids) == MAX_NEW
    return ids


def test_teacher_forced_logits_match_hf(
    tiny_parts: Parts, hf_model: Qwen3ForCausalLM, prompt: list[int], ref_ids: list[int]
) -> None:
    model, kv, attn, runner = tiny_parts
    ids = prompt + ref_ids
    seq = Sequence("tf", "r", ids, SamplingParams())
    kv.allocate(seq, len(ids))
    sched = SchedulerOutput(1, [ScheduledSeq(seq, len(ids))])
    with torch.inference_mode():
        batch = runner.prepare(sched)
        hidden = model.forward(batch, kv, attn)  # [T, C]
        ours = model.compute_logits(hidden, torch.arange(len(ids)))  # [T, V]
        ref = hf_model(torch.tensor([ids])).logits[0]  # [T, V]
    kv.free(seq)
    assert ours.dtype == torch.float32
    diff = (ours - ref).abs().amax(dim=-1)  # [T]
    bad = torch.nonzero(diff > TOL)
    if bad.numel():
        pos = int(bad[0])
        print(f"first bad position {pos} (prompt len {len(prompt)}): max abs diff {diff[pos]:.3e}")
    print(f"max abs logit diff {diff.max():.3e} over {len(ids)} positions")
    torch.testing.assert_close(ours, ref, atol=TOL, rtol=TOL)


def test_greedy_16_tokens_match_hf(
    tiny_engine: tuple[Engine, HFTokenizer], prompt: list[int], ref_ids: list[int]
) -> None:
    engine, _ = tiny_engine
    assert engine.generate(prompt, SamplingParams(temperature=0, max_tokens=MAX_NEW)) == ref_ids


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
