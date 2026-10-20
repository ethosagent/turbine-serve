"""Block 7: SchedulerOutput -> flat ForwardBatch -> model forward -> logits."""

import torch
from torch import Tensor

from turbine_serve.interfaces import AttentionBackend, KVCacheManager, Model
from turbine_serve.log import get_logger
from turbine_serve.types import ForwardBatch, SchedulerOutput

log = get_logger("model_runner")


class GPUModelRunner:
    """Flattens a step into [T] tensors and runs the model; works on any torch device."""

    def __init__(
        self, model: Model, kv: KVCacheManager, attn: AttentionBackend, device: str
    ) -> None:
        self.model = model
        self.kv = kv
        self.attn = attn
        self.device = torch.device(device)

    def prepare(self, sched: SchedulerOutput) -> ForwardBatch:
        """Prefill feeds the whole prompt; decode feeds the last sampled token."""
        block_size = self.kv.block_size
        input_ids: list[int] = []
        positions: list[int] = []
        slots: list[int] = []
        logits_indices: list[int] = []
        for item in sched.seqs:
            seq = item.seq
            tokens = seq.prompt_ids if seq.num_computed == 0 else seq.output_ids[-1:]
            assert len(tokens) == item.num_new_tokens, (
                f"{seq.seq_id}: scheduled {item.num_new_tokens} new tokens, have {len(tokens)}"
            )
            for pos in range(seq.num_computed, seq.num_computed + len(tokens)):
                positions.append(pos)
                slots.append(seq.block_table[pos // block_size] * block_size + pos % block_size)
            input_ids.extend(tokens)
            logits_indices.append(len(input_ids) - 1)
        log.debug(
            "batch.prepared",
            step_id=sched.step_id,
            num_seqs=len(sched.seqs),
            num_tokens=len(input_ids),
        )
        return ForwardBatch(
            input_ids=self._tensor(input_ids),  # [T]
            positions=self._tensor(positions),  # [T]
            slot_mapping=self._tensor(slots),  # [T]
            logits_indices=self._tensor(logits_indices),  # [B]
            attn=self.attn.build_metadata(sched, block_size),
        )

    def _tensor(self, values: list[int]) -> Tensor:
        return torch.tensor(values, dtype=torch.int64, device=self.device)  # [len(values)]

    def execute(self, sched: SchedulerOutput) -> Tensor:
        """Logits [B, vocab] in float32; the cuda sync makes the caller's forward_ms real."""
        try:
            with torch.inference_mode():
                batch = self.prepare(sched)
                hidden = self.model.forward(batch, self.kv, self.attn)  # [T, C]
                logits = self.model.compute_logits(hidden, batch.logits_indices)  # [B, V]
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
        except Exception as exc:
            log.error(
                "forward.failed",
                step_id=sched.step_id,
                seq_ids=[item.seq.seq_id for item in sched.seqs],
                error=repr(exc),
            )
            raise
        return logits
