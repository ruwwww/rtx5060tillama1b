"""
KV Cache with paged memory management.

Memory layout
─────────────
k_pools[layer] : (num_blocks * block_size, num_kv_heads, head_dim)  ← flat view
                 stored as bfloat16, contiguous, pre-zeroed

Block table
───────────
Each request owns a list of physical block indices (integers).
Logical position p maps to:
  physical_block = block_table[p // block_size]
  block_offset   = p  % block_size
  flat_index     = physical_block * block_size + block_offset

Future extensibility
────────────────────
• Radix Attention / Prefix Caching
    BlockAllocator becomes ref-counted; blocks whose content hash matches
    a cached prefix are reused across requests (zero-copy sharing).
• FlashInfer backend
    Replace gather_kv() + SDPA with flashinfer.single_decode_with_kv_cache()
    or flashinfer.BatchPrefillWithPagedKVCacheWrapper — these kernels operate
    directly on the (num_blocks, num_kv_heads, block_size, head_dim) paged
    layout without materialising a gathered tensor.
"""

from __future__ import annotations

from typing import List, Tuple

import torch


class BlockAllocator:
    """
    Simple free-list allocator.

    Future: add ref-counting for prefix-cache block sharing.
    A block with ref_count > 1 must be copy-on-write before mutation.
    """

    def __init__(self, num_blocks: int, block_size: int) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free: List[int] = list(range(num_blocks))

    # ── allocation ──────────────────────────────────────────────────────────

    def alloc(self, n: int) -> List[int]:
        if len(self._free) < n:
            raise MemoryError(
                f"KV Cache OOM: requested {n} blocks, only {len(self._free)} free"
            )
        blocks, self._free = self._free[:n], self._free[n:]
        return blocks

    def free(self, blocks: List[int]) -> None:
        self._free.extend(blocks)

    # ── helpers ──────────────────────────────────────────────────────────────

    @property
    def num_free(self) -> int:
        return len(self._free)

    def blocks_needed(self, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size


class KVCachePool:
    """
    Pre-allocated KV cache for all layers.

    Internal storage is a flat 1-D token layout so scatter/gather reduce
    to simple advanced-index ops on contiguous tensors — no reshape copies.

    Shape per layer:
        k_pools[l] : (total_slots, num_kv_heads, head_dim)
        where total_slots = num_blocks * block_size
    """

    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        num_kv_heads: int,
        block_size: int,
        head_dim: int,
        device: str,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.num_kv_heads = num_kv_heads
        self.block_size = block_size
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        total_slots = num_blocks * block_size
        pool_shape = (total_slots, num_kv_heads, head_dim)
        self.k_pools = [
            torch.zeros(pool_shape, device=device, dtype=dtype)
            for _ in range(num_layers)
        ]
        self.v_pools = [
            torch.zeros(pool_shape, device=device, dtype=dtype)
            for _ in range(num_layers)
        ]

    # ── private helpers ──────────────────────────────────────────────────────

    def _flat_indices(
        self,
        block_table: List[int],
        seq_len: int,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """
        Return flat pool indices for positions [start_pos, start_pos+seq_len).
        Shape: (seq_len,)
        """
        positions = torch.arange(
            start_pos, start_pos + seq_len, device=self.device, dtype=torch.long
        )
        logical_blocks = positions // self.block_size        # which block in table
        block_offsets  = positions  % self.block_size        # offset within block

        # Map logical → physical blocks (CPU list → CUDA tensor)
        physical = torch.tensor(
            [block_table[i] for i in logical_blocks.tolist()],
            device=self.device, dtype=torch.long,
        )
        return physical * self.block_size + block_offsets    # flat slot index

    # ── write ────────────────────────────────────────────────────────────────

    def write_one(
        self,
        layer_idx: int,
        block_table: List[int],
        position: int,          # absolute token position
        k: torch.Tensor,        # (num_kv_heads, head_dim)
        v: torch.Tensor,
    ) -> None:
        """Write a single token's KV at the given position."""
        slot = block_table[position // self.block_size] * self.block_size \
               + position % self.block_size
        self.k_pools[layer_idx][slot] = k
        self.v_pools[layer_idx][slot] = v

    def write_many(
        self,
        layer_idx: int,
        block_table: List[int],
        k: torch.Tensor,        # (seq_len, num_kv_heads, head_dim)
        v: torch.Tensor,
        start_pos: int = 0,
    ) -> None:
        """
        Vectorised scatter-write for prefill (multiple tokens at once).

        Future (FlashInfer): this whole call is replaced by a fused
        prefill kernel that writes the paged KV while computing attention.
        """
        seq_len = k.size(0)
        if seq_len == 0:
            return
        idx = self._flat_indices(block_table, seq_len, start_pos)   # (seq_len,)
        self.k_pools[layer_idx][idx] = k
        self.v_pools[layer_idx][idx] = v

    # ── gather ───────────────────────────────────────────────────────────────

    def gather_kv(
        self,
        layer_idx: int,
        block_table: List[int],
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gather KV for a single sequence from the paged pool.

        Returns
        ───────
        k : (num_kv_heads, seq_len, head_dim)
        v : (num_kv_heads, seq_len, head_dim)

        Future (FlashInfer): replace with a kernel that reads directly from
        the paged layout — no gather materialisation, lower memory bandwidth.
        """
        idx = self._flat_indices(block_table, seq_len)   # (seq_len,)
        k = self.k_pools[layer_idx][idx]                 # (seq_len, kv_heads, head_dim)
        v = self.v_pools[layer_idx][idx]
        return k.permute(1, 0, 2).contiguous(), v.permute(1, 0, 2).contiguous()
