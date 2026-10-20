"""Startup: safetensors shards -> bf16 GPU tensors."""

import time
from collections.abc import Iterator
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from turbine_serve.log import get_logger

log = get_logger("loader")


def resolve_model_path(model: str) -> str:
    """A local directory is returned as is; anything else is a Hub id and is downloaded."""
    if Path(model).is_dir():
        return model
    log.info("weights.downloading", model=model)
    return snapshot_download(model, allow_patterns=["*.safetensors", "*.json", "*.txt"])


def iter_weights(
    model_path: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield (checkpoint name, tensor cast to dtype) from every shard, in sorted shard order."""
    files = sorted(Path(model_path).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no *.safetensors files in {model_path}")
    start = time.perf_counter()
    num_tensors = 0
    num_bytes = 0
    for path in files:
        with safe_open(path, framework="pt", device=device) as f:
            for name in f.keys():
                tensor = f.get_tensor(name).to(dtype)
                num_tensors += 1
                num_bytes += tensor.numel() * tensor.element_size()
                yield name, tensor
    log.info(
        "weights.loaded",
        num_tensors=num_tensors,
        gb=round(num_bytes / 1e9, 3),
        ms=round((time.perf_counter() - start) * 1000, 1),
    )
