"""Loader and tiny checkpoint: real tensor names, shapes, passthrough, tokenizer, idempotency."""

from pathlib import Path

import pytest
import torch
from spark_engine.loader import iter_weights, resolve_model_path
from spark_engine.testing.tiny_model import make_tiny_qwen3
from transformers import AutoTokenizer


def test_tiny_checkpoint_has_real_names(tiny_model_dir: Path) -> None:
    names = {
        n: tuple(t.shape)
        for n, t in iter_weights(str(tiny_model_dir), device="cpu", dtype=torch.float32)
    }
    assert names["model.embed_tokens.weight"] == (151936, 32)
    assert names["model.layers.0.self_attn.q_proj.weight"] == (32, 32)  # Hq*D = 4*8
    assert names["model.layers.0.self_attn.k_proj.weight"] == (16, 32)  # Hkv*D = 2*8
    assert names["model.layers.0.self_attn.q_norm.weight"] == (8,)
    assert names["model.layers.1.mlp.down_proj.weight"] == (32, 64)
    assert "lm_head.weight" not in names  # tied: HF drops it
    assert len(names) == 2 * 11 + 2


def test_iter_weights_casts_dtype(tiny_model_dir: Path) -> None:
    dtypes = {
        t.dtype for _, t in iter_weights(str(tiny_model_dir), device="cpu", dtype=torch.bfloat16)
    }
    assert dtypes == {torch.bfloat16}


def test_iter_weights_without_shards_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=str(tmp_path)):
        list(iter_weights(str(tmp_path), device="cpu"))


def test_resolve_local_dir_passthrough(tiny_model_dir: Path) -> None:
    assert resolve_model_path(str(tiny_model_dir)) == str(tiny_model_dir)


def test_tokenizer_saved_alongside(tiny_model_dir: Path) -> None:
    tok = AutoTokenizer.from_pretrained(str(tiny_model_dir))
    assert tok.eos_token_id == 151645


def test_make_tiny_is_idempotent(tiny_model_dir: Path) -> None:
    shard = tiny_model_dir / "model.safetensors"
    before = shard.stat().st_mtime_ns
    assert make_tiny_qwen3(tiny_model_dir) == tiny_model_dir
    assert shard.stat().st_mtime_ns == before
