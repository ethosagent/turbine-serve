"""Block 6: paged K/V pools per layer plus a free-block list."""

import torch
from torch import Tensor

from turbine_serve.log import get_logger
from turbine_serve.types import Sequence

log = get_logger("kv_cache")


def compute_num_blocks(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    kv_gb: float,
    itemsize: int,
) -> int:
    """How many blocks fit in kv_gb GiB (K and V, all layers)."""
    bytes_per_block = num_layers * 2 * block_size * num_kv_heads * head_dim * itemsize
    return int(kv_gb * 2**30 // bytes_per_block)


class PagedKVCache:
    """One K and one V pool, each [num_layers, num_blocks, block_size, num_kv_heads, head_dim]."""

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int,
        kv_gb: float,
        device: str | torch.device,
        dtype: torch.dtype = torch.bfloat16,
        num_blocks: int | None = None,
    ) -> None:
        if num_blocks is None:
            itemsize = torch.empty((), dtype=dtype).element_size()
            num_blocks = compute_num_blocks(
                num_layers, num_kv_heads, head_dim, block_size, kv_gb, itemsize
            )
        self.block_size = block_size
        self.num_blocks = num_blocks
        shape = (num_layers, num_blocks, block_size, num_kv_heads, head_dim)
        self._k = torch.zeros(shape, dtype=dtype, device=device)  # [L, NB, bs, Hkv, D]
        self._v = torch.zeros(shape, dtype=dtype, device=device)  # [L, NB, bs, Hkv, D]
        # stack: pop() hands out the lowest block id first
        self._free: list[int] = list(range(num_blocks - 1, -1, -1))
        log.info(
            "kv.initialized",
            num_blocks=num_blocks,
            gb=kv_gb,
            block_size=block_size,
            num_layers=num_layers,
            dtype=str(dtype),
            device=str(device),
        )

    @property
    def num_free_blocks(self) -> int:
        return len(self._free)

    def blocks_needed(self, num_tokens: int) -> int:
        return -(-num_tokens // self.block_size)

    def can_allocate(self, num_tokens: int) -> bool:
        return self.blocks_needed(num_tokens) <= len(self._free)

    def allocate(self, seq: Sequence, num_tokens: int) -> None:
        """Append enough fresh blocks for num_tokens to seq.block_table."""
        needed = self.blocks_needed(num_tokens)
        if needed > len(self._free):
            raise ValueError(f"need {needed} blocks, only {len(self._free)} free")
        blocks = [self._free.pop() for _ in range(needed)]
        seq.block_table.extend(blocks)
        log.debug("blocks.allocated", seq_id=seq.seq_id, blocks=blocks, free=len(self._free))

    def free(self, seq: Sequence) -> None:
        """Return all of seq's blocks to the free list; a second call is a no-op."""
        if not seq.block_table:
            return
        count = len(seq.block_table)
        self._free.extend(reversed(seq.block_table))
        seq.block_table.clear()
        log.debug("blocks.freed", seq_id=seq.seq_id, count=count, free=len(self._free))

    def layer_cache(self, layer: int) -> tuple[Tensor, Tensor]:
        """Views (k, v), each [num_blocks, block_size, num_kv_heads, head_dim]."""
        return self._k[layer], self._v[layer]
