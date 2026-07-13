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
import flashinfer

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


class FlashInferAttention(AttentionBackend):
    """
    Highly optimized FlashInfer attention backend.
    
    Uses FlashInfer's Paged Prefill and Paged Decode Batch wrappers
    to compute attention directly over the 4D paged KV layout, entirely
    skipping intermediate contiguous gathers and padding.
    """

    def __init__(self, device: str) -> None:
        # Allocate 128 MB workspace buffer on target device
        self.workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
        self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            self.workspace, kv_layout="NHD"
        )
        self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self.workspace, kv_layout="NHD"
        )

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
        device = q.device
        S = q.size(2)

        # 1. Write prompt KV projection to 4D storage
        k_write = k.squeeze(0).permute(1, 0, 2).contiguous()
        v_write = v.squeeze(0).permute(1, 0, 2).contiguous()
        kv_backend.write_many(layer_idx, block_table, k_write, v_write, start_pos=0)

        # FlashInfer expects input q layout: [seq_len, num_heads, head_dim]
        q_fi = q.transpose(1, 2).contiguous().view(S, q.size(1), q.size(-1))

        # Build prefill metadata
        qo_indptr = torch.tensor([0, S], device=device, dtype=torch.int32)
        paged_kv_indptr = torch.tensor([0, len(block_table)], device=device, dtype=torch.int32)
        paged_kv_indices = torch.tensor(block_table, device=device, dtype=torch.int32)

        block_size = kv_backend.block_size
        last_page_len = torch.tensor([((S - 1) % block_size) + 1], device=device, dtype=torch.int32)

        self.prefill_wrapper.plan(
            qo_indptr=qo_indptr,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=last_page_len,
            num_qo_heads=q.size(1),
            num_kv_heads=k.size(1),
            head_dim_qk=q.size(-1),
            page_size=block_size,
            causal=True,
            q_data_type=q.dtype,
        )

        k_pool = kv_backend.k_pools[layer_idx]
        v_pool = kv_backend.v_pools[layer_idx]

        out = self.prefill_wrapper.run(q_fi, (k_pool, v_pool))
        return out.view(1, S, q.size(1), q.size(-1)).transpose(1, 2).contiguous()

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
        device = q.device
        block_size = kv_backend.block_size

        # 1. Write each sequence's new single-token KV to 4D storage
        for i in range(B):
            kv_backend.write_one(
                layer_idx, block_tables[i], positions[i],
                k[i, :, 0, :],
                v[i, :, 0, :],
            )

        # FlashInfer expects input q layout: [batch_size, num_heads, head_dim]
        q_fi = q.squeeze(2)

        # Build decode metadata
        lengths = [len(tbl) for tbl in block_tables]
        indptr_list = [0]
        for l in lengths:
            indptr_list.append(indptr_list[-1] + l)
        indptr = torch.tensor(indptr_list, device=device, dtype=torch.int32)

        flat_indices_list = []
        for tbl in block_tables:
            flat_indices_list.extend(tbl)
        indices = torch.tensor(flat_indices_list, device=device, dtype=torch.int32)

        seq_lens = [pos + 1 for pos in positions]
        last_page_len = torch.tensor([((slen - 1) % block_size) + 1 for slen in seq_lens], device=device, dtype=torch.int32)

        self.decode_wrapper.plan(
            indptr=indptr,
            indices=indices,
            last_page_len=last_page_len,
            num_qo_heads=q.size(1),
            num_kv_heads=k.size(1),
            head_dim=q.size(-1),
            page_size=block_size,
            q_data_type=q.dtype,
        )

        k_pool = kv_backend.k_pools[layer_idx]
        v_pool = kv_backend.v_pools[layer_idx]

        out = self.decode_wrapper.run(q_fi, (k_pool, v_pool))
        return out.unsqueeze(2)

