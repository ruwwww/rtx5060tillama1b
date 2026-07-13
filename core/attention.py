"""
Attention Backend Abstraction.

Decouples attention computation (SDPA, FlashInfer, etc.) from model code
and KV cache storage backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from core.kv_cache import KVCacheBackend


from dataclasses import dataclass

@dataclass
class FlashInferMetadata:
    block_table: torch.Tensor       # (B, max_blocks_per_seq)
    seq_lens: torch.Tensor          # (B,)
    block_size: int


class AttentionBackend(ABC):
    """
    Abstract interface for executing attention.
    """

    @abstractmethod
    def forward_prefill(
        self,
        q: torch.Tensor,                # (1, num_heads, seq_len, head_dim)
        k: torch.Tensor,                # (1, num_kv_heads, seq_len, head_dim)
        v: torch.Tensor,                # (1, num_kv_heads, seq_len, head_dim)
        layer_idx: int,
        kv_backend: KVCacheBackend,
        block_table: List[int],
        groups: int,
    ) -> torch.Tensor:                  # (1, num_heads, seq_len, head_dim)
        pass

    @abstractmethod
    def forward_decode_batch(
        self,
        q: torch.Tensor,                # (B, num_heads, 1, head_dim)
        k: torch.Tensor,                # (B, num_kv_heads, 1, head_dim)
        v: torch.Tensor,                # (B, num_kv_heads, 1, head_dim)
        layer_idx: int,
        kv_backend: KVCacheBackend,
        block_tables: List[List[int]],
        positions: List[int],
        flat_indices: Optional[List[torch.Tensor]],
        groups: int,
        metadata: FlashInferMetadata,
    ) -> torch.Tensor:                  # (B, num_heads, 1, head_dim)
        pass


class PyTorchAttention(AttentionBackend):
    """
    Standard PyTorch SDPA backend.
    Materializes contiguous gathered representations from KVCacheBackend.
    """

    def forward_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_idx: int,
        kv_backend: KVCacheBackend,
        block_table: List[int],
        groups: int,
    ) -> torch.Tensor:
        S = q.size(2)

        # Write prompt KV projection to storage
        k_write = k.squeeze(0).permute(1, 0, 2).contiguous()
        v_write = v.squeeze(0).permute(1, 0, 2).contiguous()
        kv_backend.write_many(layer_idx, block_table, k_write, v_write, start_pos=0)

        # Standard SDPA with causal masking
        k_e = k.repeat_interleave(groups, dim=1)
        v_e = v.repeat_interleave(groups, dim=1)
        return F.scaled_dot_product_attention(q, k_e, v_e, is_causal=True)

    def forward_decode_batch(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_idx: int,
        kv_backend: KVCacheBackend,
        block_tables: List[List[int]],
        positions: List[int],
        flat_indices: Optional[List[torch.Tensor]],
        groups: int,
        metadata: FlashInferMetadata,
    ) -> torch.Tensor:
        B = q.size(0)

        # Write each sequence's new single-token KV to storage
        for i in range(B):
            kv_backend.write_one(
                layer_idx, block_tables[i], positions[i],
                k[i, :, 0, :],
                v[i, :, 0, :],
            )

        # Gather padded batch context from storage
        seq_lens = [p + 1 for p in positions]
        k_ctx, v_ctx, mask = kv_backend.gather_kv_batch(
            layer_idx, block_tables, seq_lens, flat_indices
        )

        # Expand KV heads to match query heads (GQA)
        k_e = k_ctx.repeat_interleave(groups, dim=1)
        v_e = v_ctx.repeat_interleave(groups, dim=1)

        # Run batched attention
        return F.scaled_dot_product_attention(
            q, k_e, v_e, attn_mask=mask.to(q.dtype)
        )
