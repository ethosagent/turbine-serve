"""EngineConfig (pydantic): every runtime knob with its default."""

from typing import Literal

import torch
from pydantic import BaseModel

from spark_engine.log import get_logger

log = get_logger("config")

_TORCH_DTYPES: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class EngineConfig(BaseModel):
    model: str
    kv_gb: float = 30
    block_size: int = 16
    max_num_seqs: int = 8
    max_model_len: int = 8192
    max_queue: int = 64
    attention_backend: str = "sdpa"
    device: str = "cuda"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    log_level: str = "info"
    log_file: str = "logs/engine.jsonl"
    host: str = "127.0.0.1"
    port: int = 8000

    @property
    def torch_dtype(self) -> torch.dtype:
        return _TORCH_DTYPES[self.dtype]


def log_resolved(config: EngineConfig) -> None:
    """Emit the fully resolved config as one `config.resolved` event."""
    log.info("config.resolved", **config.model_dump())
