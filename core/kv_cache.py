"""
KV Cache Storage Abstraction.

Defines KVCacheBackend and the ContiguousKVBackend implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

import torch


class KVCacheBackend(ABC):
    """
    Abstract storage backend for KV cache memory management and reading/writing.
    """

    @abstractmethod
    def blocks_needed(self, num_tokens: int) -> int:
        """Calculate how many blocks are required for the given number of tokens."""
        pass

    @property
    @abstractmethod
    def num_free_blocks(self) -> int:
        """Number of free blocks remaining in the allocator."""
        pass

    @abstractmethod
    def alloc(self, n: int) -> List[int]:
        """Allocate n blocks and return their physical indices."""
        pass

    @abstractmethod
    def free(self, block_table: List[int]) -> None:
        """Free a list of blocks back to the allocator."""
        pass

    @abstractmethod
    def write_one(
        self,
        layer_idx: int,
        block_table: List[int],
        position: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """Write a single token's KV projection at the given position."""
        pass

    @abstractmethod
    def write_many(
        self,
        layer_idx: int,
        block_table: List[int],
        k: torch.Tensor,
        v: torch.Tensor,
        start_pos: int = 0,
    ) -> None:
        """Vectorized scatter-write for multiple prefill tokens at once."""
        pass

    @abstractmethod
    def gather_kv(
        self,
        layer_idx: int,
        block_table: List[int],
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather KV for a single sequence from the storage backend."""
        pass

    @abstractmethod
    def gather_kv_batch(
        self,
        layer_idx: int,
        block_tables: List[List[int]],
        seq_lens: List[int],
        flat_indices: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather KV for a batch of sequences, padded to max(seq_lens) with an attention mask."""
        pass

    @abstractmethod
    def precompute_decode_indices(
        self,
        block_tables: List[List[int]],
        positions: List[int],
    ) -> List[torch.Tensor]:
        """Precompute flat lookup indices for decoding requests once per step."""
        pass


class BlockAllocator:
    def __init__(self, num_blocks: int, block_size: int) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free: List[int] = list(range(num_blocks))

    def alloc(self, n: int) -> List[int]:
        if len(self._free) < n:
            raise MemoryError(
                f"KV Cache OOM: requested {n} blocks, only {len(self._free)} free"
            )
        blocks, self._free = self._free[:n], self._free[n:]
        return blocks

    def free(self, blocks: List[int]) -> None:
        self._free.extend(blocks)

    @property
    def num_free(self) -> int:
        return len(self._free)

    def blocks_needed(self, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size


class KVCachePool:
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

    def _flat_indices(
        self,
        block_table: List[int],
        seq_len: int,
        start_pos: int = 0,
    ) -> torch.Tensor:
        positions = torch.arange(
            start_pos, start_pos + seq_len, device=self.device, dtype=torch.long
        )
        logical_blocks = positions // self.block_size
        block_offsets  = positions  % self.block_size
        physical = torch.tensor(
            [block_table[i] for i in logical_blocks.tolist()],
            device=self.device, dtype=torch.long,
        )
        return physical * self.block_size + block_offsets

    def write_one(
        self,
        layer_idx: int,
        block_table: List[int],
        position: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        slot = block_table[position // self.block_size] * self.block_size \
               + position % self.block_size
        self.k_pools[layer_idx][slot] = k
        self.v_pools[layer_idx][slot] = v

    def write_many(
        self,
        layer_idx: int,
        block_table: List[int],
        k: torch.Tensor,
        v: torch.Tensor,
        start_pos: int = 0,
    ) -> None:
        seq_len = k.size(0)
        if seq_len == 0:
            return
        idx = self._flat_indices(block_table, seq_len, start_pos)
        self.k_pools[layer_idx][idx] = k
        self.v_pools[layer_idx][idx] = v

    def gather_kv(
        self,
        layer_idx: int,
        block_table: List[int],
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = self._flat_indices(block_table, seq_len)
        k = self.k_pools[layer_idx][idx]
        v = self.v_pools[layer_idx][idx]
        return k.permute(1, 0, 2).contiguous(), v.permute(1, 0, 2).contiguous()


class ContiguousKVBackend(KVCacheBackend):
    """
    Standard implementation wrapping KVCachePool and BlockAllocator.
    Keeps the exact contiguous gathering behaviour as the original codebase.
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
        self.allocator = BlockAllocator(num_blocks, block_size)
        self.pool = KVCachePool(
            num_layers, num_blocks, num_kv_heads, block_size, head_dim, device, dtype
        )

    def blocks_needed(self, num_tokens: int) -> int:
        return self.allocator.blocks_needed(num_tokens)

    @property
    def num_free_blocks(self) -> int:
        return self.allocator.num_free

    def alloc(self, n: int) -> List[int]:
        return self.allocator.alloc(n)

    def free(self, block_table: List[int]) -> None:
        self.allocator.free(block_table)

    def write_one(
        self,
        layer_idx: int,
        block_table: List[int],
        position: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        self.pool.write_one(layer_idx, block_table, position, k, v)

    def write_many(
        self,
        layer_idx: int,
        block_table: List[int],
        k: torch.Tensor,
        v: torch.Tensor,
        start_pos: int = 0,
    ) -> None:
        self.pool.write_many(layer_idx, block_table, k, v, start_pos)

    def gather_kv(
        self,
        layer_idx: int,
        block_table: List[int],
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.pool.gather_kv(layer_idx, block_table, seq_len)

    def gather_kv_batch(
        self,
        layer_idx: int,
        block_tables: List[List[int]],
        seq_lens: List[int],
        flat_indices: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B       = len(seq_lens)
        max_len = max(seq_lens)

        k_out = torch.zeros(
            B, self.pool.num_kv_heads, max_len, self.pool.head_dim,
            device=self.pool.device, dtype=self.pool.dtype,
        )
        v_out  = torch.zeros_like(k_out)
        mask = torch.full(
            (B, 1, 1, max_len), float("-inf"),
            device=self.pool.device, dtype=torch.float32,
        )

        for i, slen in enumerate(seq_lens):
            idx = (
                flat_indices[i]
                if flat_indices is not None
                else self.pool._flat_indices(block_tables[i], slen)
            )
            k = self.pool.k_pools[layer_idx][idx]
            v = self.pool.v_pools[layer_idx][idx]
            k_out[i, :, :slen, :] = k.permute(1, 0, 2)
            v_out[i, :, :slen, :] = v.permute(1, 0, 2)
            mask[i, 0, 0, :slen] = 0.0

        return k_out, v_out, mask

    def precompute_decode_indices(
        self,
        block_tables: List[List[int]],
        positions: List[int],
    ) -> List[torch.Tensor]:
        return [
            self.pool._flat_indices(block_tables[i], positions[i] + 1)
            for i in range(len(positions))
        ]


class PagedKVBackend(KVCacheBackend):
    """
    True paged layout KV cache storage backend.
    
    Stores key/value pools as 4D tensors:
      (num_blocks, block_size, num_kv_heads, head_dim)
    
    This matches the exact physical layout that FlashInfer expects, but
    materialises contiguous views for standard SDPA during execution to 
    allow validating correctness.
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

        self.allocator = BlockAllocator(num_blocks, block_size)

        # 4D flat storage pools per layer: (num_blocks, block_size, num_kv_heads, head_dim)
        pool_shape = (num_blocks, block_size, num_kv_heads, head_dim)
        self.k_pools = [
            torch.zeros(pool_shape, device=device, dtype=dtype)
            for _ in range(num_layers)
        ]
        self.v_pools = [
            torch.zeros(pool_shape, device=device, dtype=dtype)
            for _ in range(num_layers)
        ]

    def blocks_needed(self, num_tokens: int) -> int:
        return self.allocator.blocks_needed(num_tokens)

    @property
    def num_free_blocks(self) -> int:
        return self.allocator.num_free

    def alloc(self, n: int) -> List[int]:
        return self.allocator.alloc(n)

    def free(self, block_table: List[int]) -> None:
        self.allocator.free(block_table)

    def _flat_indices(self, block_table: List[int], seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, device=self.device, dtype=torch.long)
        logical_blocks = positions // self.block_size
        offsets = positions % self.block_size
        physical_blocks = torch.tensor(
            [block_table[i] for i in logical_blocks.tolist()],
            device=self.device, dtype=torch.long,
        )
        return physical_blocks, offsets

    def write_one(
        self,
        layer_idx: int,
        block_table: List[int],
        position: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        block_idx = block_table[position // self.block_size]
        offset = position % self.block_size
        self.k_pools[layer_idx][block_idx, offset] = k
        self.v_pools[layer_idx][block_idx, offset] = v

    def write_many(
        self,
        layer_idx: int,
        block_table: List[int],
        k: torch.Tensor,
        v: torch.Tensor,
        start_pos: int = 0,
    ) -> None:
        seq_len = k.size(0)
        if seq_len == 0:
            return
        blocks, offsets = self._flat_indices(block_table, seq_len)
        self.k_pools[layer_idx][blocks, offsets] = k
        self.v_pools[layer_idx][blocks, offsets] = v

    def gather_kv(
        self,
        layer_idx: int,
        block_table: List[int],
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        blocks, offsets = self._flat_indices(block_table, seq_len)
        k = self.k_pools[layer_idx][blocks, offsets]
        v = self.v_pools[layer_idx][blocks, offsets]
        return k.permute(1, 0, 2).contiguous(), v.permute(1, 0, 2).contiguous()

    def gather_kv_batch(
        self,
        layer_idx: int,
        block_tables: List[List[int]],
        seq_lens: List[int],
        flat_indices: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = len(seq_lens)
        max_len = max(seq_lens)

        k_out = torch.zeros(
            B, self.num_kv_heads, max_len, self.head_dim,
            device=self.device, dtype=self.dtype,
        )
        v_out = torch.zeros_like(k_out)
        mask = torch.full(
            (B, 1, 1, max_len), float("-inf"),
            device=self.device, dtype=torch.float32,
        )

        for i, slen in enumerate(seq_lens):
            if flat_indices is not None:
                blocks, offsets = flat_indices[i]
            else:
                blocks, offsets = self._flat_indices(block_tables[i], slen)
            k = self.k_pools[layer_idx][blocks, offsets]
            v = self.v_pools[layer_idx][blocks, offsets]
            k_out[i, :, :slen, :] = k.permute(1, 0, 2)
            v_out[i, :, :slen, :] = v.permute(1, 0, 2)
            mask[i, 0, 0, :slen] = 0.0

        return k_out, v_out, mask

    def precompute_decode_indices(
        self,
        block_tables: List[List[int]],
        positions: List[int],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        return [
            self._flat_indices(block_tables[i], positions[i] + 1)
            for i in range(len(positions))
        ]

