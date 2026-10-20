"""Protocols for every swappable block (SPEC.md: Contracts). Imports only types."""

from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

from torch import Tensor

from spark_engine.types import (
    AttentionMetadata,
    ForwardBatch,
    SchedulerOutput,
    Sequence,
    TokenOutput,
)


@runtime_checkable
class Detokenizer(Protocol):
    def step(self, token_id: int) -> str: ...


@runtime_checkable
class Tokenizer(Protocol):
    eos_token_ids: tuple[int, ...]

    def encode_chat(self, messages: list[dict[str, Any]], **template_kwargs: Any) -> list[int]: ...
    def new_detokenizer(self) -> Detokenizer: ...


@runtime_checkable
class KVCacheManager(Protocol):
    block_size: int

    @property
    def num_free_blocks(self) -> int: ...

    def can_allocate(self, num_tokens: int) -> bool: ...
    def allocate(self, seq: Sequence, num_tokens: int) -> None: ...
    def free(self, seq: Sequence) -> None: ...
    def layer_cache(self, layer: int) -> tuple[Tensor, Tensor]: ...


@runtime_checkable
class Scheduler(Protocol):
    def add(self, seq: Sequence) -> None: ...
    def schedule(self) -> SchedulerOutput: ...
    def update(self, out: SchedulerOutput, tokens: list[int]) -> list[TokenOutput]: ...
    def abort(self, seq_id: str) -> None: ...
    def fail(self, seq_id: str, error: str) -> TokenOutput | None: ...
    def has_work(self) -> bool: ...


@runtime_checkable
class AttentionBackend(Protocol):
    def build_metadata(self, sched: SchedulerOutput, block_size: int) -> AttentionMetadata: ...
    def forward(
        self, q: Tensor, k_cache: Tensor, v_cache: Tensor, meta: AttentionMetadata
    ) -> Tensor: ...


@runtime_checkable
class Model(Protocol):
    def load_weights(self, weights: Iterable[tuple[str, Tensor]]) -> None: ...
    def forward(
        self, batch: ForwardBatch, kv: KVCacheManager, attn: AttentionBackend
    ) -> Tensor: ...
    def compute_logits(self, hidden: Tensor, indices: Tensor) -> Tensor: ...


@runtime_checkable
class ModelRunner(Protocol):
    def execute(self, sched: SchedulerOutput) -> Tensor: ...


@runtime_checkable
class Sampler(Protocol):
    def sample(self, logits: Tensor, seqs: list[Sequence]) -> list[int]: ...
