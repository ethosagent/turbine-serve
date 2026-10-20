"""Data contracts exchanged between blocks (SPEC.md: Contracts). The only shapes blocks share."""

from dataclasses import dataclass, field
from enum import Enum

from torch import Tensor


@dataclass(frozen=True)
class SamplingParams:
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    seed: int | None = None


class SeqStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class Sequence:
    seq_id: str
    request_id: str
    prompt_ids: list[int]
    params: SamplingParams
    output_ids: list[int] = field(default_factory=list)
    status: SeqStatus = SeqStatus.WAITING
    num_computed: int = 0  # tokens whose KV is in cache
    block_table: list[int] = field(default_factory=list)
    finish_reason: str | None = None  # "stop" | "length" | "abort" | "error"
    arrival_ts: float = 0.0
    first_token_ts: float | None = None

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_ids) + len(self.output_ids)


@dataclass(frozen=True)
class ScheduledSeq:
    seq: Sequence
    num_new_tokens: int  # prompt length on prefill, 1 on decode
    num_lookahead: int = 0  # reserved for speculative decoding


@dataclass(frozen=True)
class SchedulerOutput:
    step_id: int
    seqs: list[ScheduledSeq]


@dataclass(frozen=True)
class AttentionMetadata:
    """Plain base; backend-specific subclasses override the fields with concrete types."""

    q_start: Tensor | None = None
    q_len: Tensor | None = None
    ctx_len: Tensor | None = None
    block_tables: list[list[int]] | None = None


@dataclass(frozen=True)
class ForwardBatch:
    input_ids: Tensor  # [T]
    positions: Tensor  # [T]
    slot_mapping: Tensor  # [T]  block_id * block_size + offset
    logits_indices: Tensor  # [B]  last token of each seq
    attn: AttentionMetadata


@dataclass(frozen=True)
class TokenOutput:
    request_id: str
    seq_id: str
    token_id: int
    finished: bool
    finish_reason: str | None = None
