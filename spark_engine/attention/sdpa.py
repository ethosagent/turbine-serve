"""Block 9: per-sequence gather from the paged cache + torch SDPA with GQA."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from spark_engine.log import get_logger
from spark_engine.types import AttentionMetadata, SchedulerOutput

log = get_logger("sdpa")


@dataclass(frozen=True)
class SDPAMetadata(AttentionMetadata):
    q_start: list[int]  # type: ignore[assignment]  # offset of each seq in the flat batch
    q_len: list[int]  # type: ignore[assignment]  # new tokens per seq
    ctx_len: list[int]  # type: ignore[assignment]  # num_computed + new tokens per seq
    block_tables: list[list[int]]  # block ids per seq


class SDPAAttention:
    def __init__(self, num_q_heads: int, num_kv_heads: int, head_dim: int) -> None:
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

    def build_metadata(self, sched: SchedulerOutput, block_size: int) -> SDPAMetadata:
        q_start: list[int] = []
        q_len: list[int] = []
        ctx_len: list[int] = []
        block_tables: list[list[int]] = []
        offset = 0
        for item in sched.seqs:
            q_start.append(offset)
            q_len.append(item.num_new_tokens)
            ctx_len.append(item.seq.num_computed + item.num_new_tokens)
            block_tables.append(list(item.seq.block_table))
            offset += item.num_new_tokens
        log.debug("attention.metadata", step_id=sched.step_id, num_seqs=len(q_len), tokens=offset)
        return SDPAMetadata(q_start, q_len, ctx_len, block_tables)

    def forward(
        self, q: Tensor, k_cache: Tensor, v_cache: Tensor, meta: AttentionMetadata
    ) -> Tensor:
        assert isinstance(meta, SDPAMetadata)
        out = torch.empty_like(q)  # [T, Hq, D]
        block_size = k_cache.shape[1]
        for start, n_q, ctx, table in zip(
            meta.q_start, meta.q_len, meta.ctx_len, meta.block_tables, strict=True
        ):
            assert n_q in (1, ctx), (
                f"q_len must be 1 (decode) or equal ctx_len (full prefill); "
                f"got q_len={n_q}, ctx_len={ctx}"
            )
            out[start : start + n_q] = self._attend_one(
                q[start : start + n_q], k_cache, v_cache, table, ctx, block_size
            )
        return out

    def _attend_one(
        self,
        q_i: Tensor,  # [q_len, Hq, D]
        k_cache: Tensor,  # [NB, bs, Hkv, D]
        v_cache: Tensor,  # [NB, bs, Hkv, D]
        table: list[int],
        ctx: int,
        block_size: int,
    ) -> Tensor:
        nblocks = -(-ctx // block_size)
        blocks = torch.tensor(table[:nblocks], device=k_cache.device)  # [nblocks]
        k = k_cache[blocks].reshape(-1, self.num_kv_heads, self.head_dim)[:ctx]  # [ctx, Hkv, D]
        v = v_cache[blocks].reshape(-1, self.num_kv_heads, self.head_dim)[:ctx]  # [ctx, Hkv, D]
        qh = q_i.transpose(0, 1)[None]  # [1, Hq, q_len, D]
        kh = k.transpose(0, 1)[None]  # [1, Hkv, ctx, D]
        vh = v.transpose(0, 1)[None]  # [1, Hkv, ctx, D]
        o = F.scaled_dot_product_attention(
            qh, kh, vh, is_causal=q_i.shape[0] > 1, enable_gqa=True
        )  # [1, Hq, q_len, D]
        return o[0].transpose(0, 1)  # [q_len, Hq, D]
