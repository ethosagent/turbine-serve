"""Writes a 2-layer random-weight Qwen3 checkpoint with the real tokenizer (Plan A Task 11)."""

import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM


def make_tiny_qwen3(out_dir: Path, seed: int = 0) -> Path:
    """Save a tiny Qwen3 (real names, full vocab, tied head) plus the Qwen3-4B tokenizer.

    Idempotent: returns `out_dir` untouched when it already holds a config.json.
    """
    out_dir = Path(out_dir)
    if (out_dir / "config.json").exists():
        return out_dir
    config = Qwen3Config(
        num_hidden_layers=2,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=151936,
        rope_parameters={"rope_type": "default", "rope_theta": 1e6},
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        max_position_embeddings=512,
    )
    torch.manual_seed(seed)
    model = Qwen3ForCausalLM(config)  # type: ignore[no-untyped-call]  # transformers is untyped
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained("Qwen/Qwen3-4B").save_pretrained(out_dir)
    return out_dir


if __name__ == "__main__":
    print(make_tiny_qwen3(Path(sys.argv[1])))
