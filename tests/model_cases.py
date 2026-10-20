"""The tiny/real model cases the data-plane tests are parametrized over (fixtures: conftest.py)."""

import gc
from dataclasses import dataclass
from typing import Literal

import torch
from spark_engine.config import EngineConfig
from spark_engine.engine import Engine
from spark_engine.types import SamplingParams, ScheduledSeq, SchedulerOutput, Sequence
from transformers import Qwen3ForCausalLM

REAL_MODEL = "Qwen/Qwen3-4B"
PARITY_MESSAGES = [{"role": "user", "content": "Explain the paged KV cache in two sentences."}]


@dataclass(frozen=True)
class ModelCase:
    """One model the engine tests run against, with the settings that fit its numerics."""

    name: str  # "tiny" | "real"
    model_dir: str
    device: str
    dtype: Literal["bfloat16", "float16", "float32"]
    kv_gb: float
    parity_tokens: int  # greedy tokens compared against transformers

    @property
    def torch_dtype(self) -> torch.dtype:
        return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
            self.dtype
        ]

    @property
    def config(self) -> EngineConfig:
        return EngineConfig(
            model=self.model_dir,
            device=self.device,
            dtype=self.dtype,
            kv_gb=self.kv_gb,
        )


def free_gpu_memory() -> None:
    """Drop unreachable tensors and hand cached CUDA blocks back before the next real model."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def engine_logits(engine: Engine, ids: list[int]) -> torch.Tensor:
    """Teacher-forced logits [T, V] from the engine's own model, KV cache and attention."""
    runner = engine._runner  # the data-plane blocks are wired privately; parity needs them
    seq = Sequence("tf", "r", ids, SamplingParams())
    runner.kv.allocate(seq, len(ids))
    try:
        with torch.inference_mode():
            batch = runner.prepare(SchedulerOutput(1, [ScheduledSeq(seq, len(ids))]))
            hidden = runner.model.forward(batch, runner.kv, runner.attn)  # [T, C]
            idx = torch.arange(len(ids), device=hidden.device)  # [T]
            return runner.model.compute_logits(hidden, idx)  # [T, V]
    finally:
        runner.kv.free(seq)


def top2(row: torch.Tensor) -> str:
    vals, toks = row.topk(2)
    return f"{int(toks[0])}={float(vals[0]):.4f} {int(toks[1])}={float(vals[1]):.4f}"


def load_hf(case: ModelCase) -> Qwen3ForCausalLM:
    """The transformers reference in the case's dtype on the case's device."""
    # .to(device) rather than device_map: device_map needs `accelerate`, not a dependency.
    model = Qwen3ForCausalLM.from_pretrained(case.model_dir, dtype=case.torch_dtype)
    return model.to(case.device).eval()


def hf_logits(hf_model: Qwen3ForCausalLM, ids: list[int], device: str) -> torch.Tensor:
    """Teacher-forced HF logits [T, V] in fp32."""
    with torch.inference_mode():
        return hf_model(torch.tensor([ids], device=device)).logits[0].float()  # [T, V]


def hf_greedy(
    hf_model: Qwen3ForCausalLM, prompt: list[int], n: int, device: str, pad_id: int
) -> list[int]:
    """HF greedy continuation of `prompt`, at most `n` tokens (stops early on EOS)."""
    input_ids = torch.tensor([prompt], device=device)  # [1, P]
    with torch.inference_mode():
        out = hf_model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=n,
            do_sample=False,
            pad_token_id=pad_id,
        )  # [1, P + N]
    return list(out[0, len(prompt) :].tolist())
