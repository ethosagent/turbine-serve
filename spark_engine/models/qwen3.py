"""Block 8: Qwen3 decoder (QK-Norm, RoPE theta 1e6, tied LM head)."""

import time
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import PretrainedConfig

from spark_engine.interfaces import AttentionBackend, KVCacheManager
from spark_engine.layers.ops import apply_rope, rms_norm, rope_cos_sin, silu_and_mul
from spark_engine.log import get_logger
from spark_engine.types import ForwardBatch

log = get_logger("qwen3")

EMBED = "model.embed_tokens.weight"
LM_HEAD = "lm_head.weight"  # absent from the shards when tie_word_embeddings; aliases EMBED
FINAL_NORM = "model.norm.weight"
LAYER_WEIGHTS = (
    "input_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


def rope_theta(cfg: PretrainedConfig) -> float:
    """transformers 5 keeps theta in rope_parameters; older configs expose rope_theta."""
    theta = getattr(cfg, "rope_theta", None)
    if theta is None:
        theta = cfg.rope_parameters["rope_theta"]
    return float(theta)


class Qwen3Model:
    """Weights are plain tensors keyed by checkpoint name; no nn.Module, no autograd."""

    def __init__(self, cfg: PretrainedConfig, device: str, dtype: torch.dtype) -> None:
        self.num_layers = int(cfg.num_hidden_layers)
        self.num_q_heads = int(cfg.num_attention_heads)
        self.num_kv_heads = int(cfg.num_key_value_heads)
        self.head_dim = int(cfg.head_dim)
        self.eps = float(cfg.rms_norm_eps)
        self.theta = rope_theta(cfg)
        self.device = torch.device(device)
        self.dtype = dtype
        self.weights: dict[str, Tensor] = {}

    def required_names(self) -> list[str]:
        names = [EMBED, FINAL_NORM, LM_HEAD]
        for i in range(self.num_layers):
            names.extend(f"model.layers.{i}.{suffix}" for suffix in LAYER_WEIGHTS)
        return names

    def load_weights(self, weights: Iterable[tuple[str, Tensor]]) -> None:
        start = time.perf_counter()
        num_params = 0
        num_bytes = 0
        for name, tensor in weights:
            stored = tensor.to(device=self.device, dtype=self.dtype)
            self.weights[name] = stored
            num_params += stored.numel()
            num_bytes += stored.numel() * stored.element_size()
        if LM_HEAD not in self.weights and EMBED in self.weights:
            self.weights[LM_HEAD] = self.weights[EMBED]  # tied embeddings: same storage
        missing = [name for name in self.required_names() if name not in self.weights]
        if missing:
            raise KeyError(f"checkpoint is missing {len(missing)} weights: {missing}")
        log.info(
            "model.loaded",
            params=num_params,
            gb=round(num_bytes / 1e9, 3),
            ms=round((time.perf_counter() - start) * 1000, 1),
            num_layers=self.num_layers,
            dtype=str(self.dtype),
            device=str(self.device),
        )

    def forward(self, batch: ForwardBatch, kv: KVCacheManager, attn: AttentionBackend) -> Tensor:
        h = self.weights[EMBED][batch.input_ids]  # [T, C]
        cos, sin = rope_cos_sin(batch.positions, self.head_dim, self.theta)  # [T, D] fp32 each
        for i in range(self.num_layers):
            h = self._layer(i, h, cos, sin, batch, kv, attn)  # [T, C]
        return rms_norm(h, self.weights[FINAL_NORM], self.eps)  # [T, C]

    def compute_logits(self, hidden: Tensor, indices: Tensor) -> Tensor:
        """Logits for the rows in `indices`, as float32 for the sampler."""
        return F.linear(hidden[indices], self.weights[LM_HEAD]).float()  # [B, V]

    def _layer(
        self,
        i: int,
        h: Tensor,  # [T, C] residual stream
        cos: Tensor,  # [T, D]
        sin: Tensor,  # [T, D]
        batch: ForwardBatch,
        kv: KVCacheManager,
        attn: AttentionBackend,
    ) -> Tensor:
        w = self.weights
        prefix = f"model.layers.{i}."
        x = rms_norm(h, w[prefix + "input_layernorm.weight"], self.eps)  # [T, C]
        h = h + self._attention(i, x, cos, sin, batch, kv, attn)  # [T, C]
        x = rms_norm(h, w[prefix + "post_attention_layernorm.weight"], self.eps)  # [T, C]
        gate = F.linear(x, w[prefix + "mlp.gate_proj.weight"])  # [T, I]
        up = F.linear(x, w[prefix + "mlp.up_proj.weight"])  # [T, I]
        return h + F.linear(silu_and_mul(gate, up), w[prefix + "mlp.down_proj.weight"])  # [T, C]

    def _attention(
        self,
        i: int,
        x: Tensor,  # [T, C] normed input
        cos: Tensor,  # [T, D]
        sin: Tensor,  # [T, D]
        batch: ForwardBatch,
        kv: KVCacheManager,
        attn: AttentionBackend,
    ) -> Tensor:
        w = self.weights
        prefix = f"model.layers.{i}.self_attn."
        num_tokens, hq, hkv, d = x.shape[0], self.num_q_heads, self.num_kv_heads, self.head_dim
        q = F.linear(x, w[prefix + "q_proj.weight"]).view(num_tokens, hq, d)  # [T, Hq, D]
        k = F.linear(x, w[prefix + "k_proj.weight"]).view(num_tokens, hkv, d)  # [T, Hkv, D]
        v = F.linear(x, w[prefix + "v_proj.weight"]).view(num_tokens, hkv, d)  # [T, Hkv, D]
        # QK-Norm: per-head RMSNorm after the head split and before RoPE.
        q = apply_rope(rms_norm(q, w[prefix + "q_norm.weight"], self.eps), cos, sin)  # [T, Hq, D]
        k = apply_rope(rms_norm(k, w[prefix + "k_norm.weight"], self.eps), cos, sin)  # [T, Hkv, D]
        k_cache, v_cache = kv.layer_cache(i)  # [NB, bs, Hkv, D] each
        k_cache.view(-1, hkv, d)[batch.slot_mapping] = k  # slot = block_id * bs + offset
        v_cache.view(-1, hkv, d)[batch.slot_mapping] = v  # rows [T, Hkv, D]
        o = attn.forward(q, k_cache, v_cache, batch.attn)  # [T, Hq, D]
        return F.linear(o.reshape(num_tokens, hq * d), w[prefix + "o_proj.weight"])  # [T, C]
