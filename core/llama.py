"""
Llama 3.2 architecture — all model definitions live here.

Attention has two execution paths:

  forward_prefill  — full causal SDPA over prompt + scatter-write KV to pool
  forward_decode   — read from paged KV pool + SDPA for single new token

GQA groups: 32 query heads, 8 KV heads  (group_size = 4)

Future (FlashInfer)
───────────────────
Replace the two attention paths with:
  flashinfer.BatchPrefillWithPagedKVCacheWrapper   (prefill)
  flashinfer.BatchDecodeWithPagedKVCacheWrapper    (decode)
Both kernels operate directly on the paged pool layout — no gather copy.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig

if TYPE_CHECKING:
    from core.kv_cache import KVCachePool


# ── RMS Norm ─────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(orig)


# ── Rotary Embedding (Llama 3 scaled) ────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    def __init__(self, model_cfg: ModelConfig) -> None:
        super().__init__()
        self.head_dim = model_cfg.head_dim

        inv_freq = 1.0 / (
            model_cfg.rope_theta
            ** (torch.arange(0, model_cfg.head_dim, 2, dtype=torch.float32) / model_cfg.head_dim)
        )

        if model_cfg.rope_scaling_type == "llama3":
            inv_freq = self._apply_llama3_scaling(
                inv_freq,
                factor=model_cfg.rope_scaling_factor,
                low_freq_factor=model_cfg.rope_scaling_low_freq_factor,
                high_freq_factor=model_cfg.rope_scaling_high_freq_factor,
                old_ctx=model_cfg.rope_scaling_original_context,
            )

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @staticmethod
    def _apply_llama3_scaling(
        inv_freq: torch.Tensor,
        factor: float,
        low_freq_factor: float,
        high_freq_factor: float,
        old_ctx: int,
    ) -> torch.Tensor:
        wavelen          = 2 * math.pi / inv_freq
        low_freq_wavelen = old_ctx / low_freq_factor
        high_freq_wavelen = old_ctx / high_freq_factor

        scaled = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
        smooth = (old_ctx / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
        smooth = smooth.clamp(0.0, 1.0)
        smoothed = (1 - smooth) * (scaled / factor) + smooth * scaled
        is_medium = ~(wavelen < high_freq_wavelen) & ~(wavelen > low_freq_wavelen)
        return torch.where(is_medium, smoothed, scaled)

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inv = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, -1)
        pos = position_ids.float()[:, None, :]
        freqs = (inv @ pos).transpose(1, 2)
        emb   = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    @staticmethod
    def apply_rotary(
        q: torch.Tensor, k: torch.Tensor,
        cos: torch.Tensor, sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        return (
            q * cos + RotaryEmbedding.rotate_half(q) * sin,
            k * cos + RotaryEmbedding.rotate_half(k) * sin,
        )


# ── Grouped-Query Attention ───────────────────────────────────────────────────

class GQA(nn.Module):
    def __init__(self, model_cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx  = layer_idx
        self.num_heads  = model_cfg.num_attention_heads
        self.num_kv     = model_cfg.num_key_value_heads
        self.head_dim   = model_cfg.head_dim
        self.groups     = self.num_heads // self.num_kv
        H  = model_cfg.hidden_size
        Hq = self.num_heads * self.head_dim
        Hk = self.num_kv   * self.head_dim

        self.q_proj = nn.Linear(H, Hq, bias=False)
        self.k_proj = nn.Linear(H, Hk, bias=False)
        self.v_proj = nn.Linear(H, Hk, bias=False)
        self.o_proj = nn.Linear(Hq, H, bias=False)

    # ── shared helper ─────────────────────────────────────────────────────────

    def _qkv(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project + RoPE.  Returns q,k,v in (B, heads, S, D) layout."""
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_kv,   self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_kv,   self.head_dim).transpose(1, 2)
        cos = cos[:, :, :self.head_dim]
        sin = sin[:, :, :self.head_dim]
        q, k = RotaryEmbedding.apply_rotary(q, k, cos, sin)
        return q, k, v

    # ── standard path (no KV cache) ───────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S, _ = x.shape
        q, k, v = self._qkv(x, cos, sin)
        k = k.repeat_interleave(self.groups, dim=1)
        v = v.repeat_interleave(self.groups, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=(S > 1))
        return self.o_proj(out.transpose(1, 2).contiguous().view(B, S, -1))

    # ── prefill path ──────────────────────────────────────────────────────────

    def forward_prefill(
        self,
        x: torch.Tensor,            # (1, seq_len, hidden)
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: "KVCachePool",
        block_table: List[int],
    ) -> torch.Tensor:
        """
        Causal SDPA over full prompt then scatter-write KV into the pool.

        Future (FlashInfer): fused prefill kernel writes KV while computing
        attention — replace this method body with one kernel call.
        """
        _, S, _ = x.shape
        q, k, v = self._qkv(x, cos, sin)
        # k,v: (1, num_kv, S, head_dim)

        # ── write KV to pool ─────────────────────────────────────────────────
        # k_write: (S, num_kv, head_dim)
        k_write = k.squeeze(0).permute(1, 0, 2).contiguous()
        v_write = v.squeeze(0).permute(1, 0, 2).contiguous()
        kv_cache.write_many(self.layer_idx, block_table, k_write, v_write, start_pos=0)

        # ── causal attention ─────────────────────────────────────────────────
        k_e = k.repeat_interleave(self.groups, dim=1)
        v_e = v.repeat_interleave(self.groups, dim=1)
        out = F.scaled_dot_product_attention(q, k_e, v_e, is_causal=True)
        return self.o_proj(out.transpose(1, 2).contiguous().view(1, S, -1))

    # ── decode path ───────────────────────────────────────────────────────────

    def forward_decode(
        self,
        x: torch.Tensor,            # (1, 1, hidden)
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: "KVCachePool",
        block_table: List[int],
        position: int,              # absolute position of this new token
    ) -> torch.Tensor:
        """
        Single-token decode: write new KV to pool, gather full context, attend.

        Steps:
          1. q, k, v = project + RoPE
          2. write_one → pool at position
          3. gather_kv  → (num_kv, pos+1, head_dim)
          4. SDPA (no causal mask — single query, all keys are in the past)

        Future (FlashInfer): replace steps 2-4 with
          flashinfer.single_decode_with_kv_cache(q, kv_cache, block_table)
        """
        q, k, v = self._qkv(x, cos, sin)
        # k,v: (1, num_kv, 1, head_dim)

        # ── write new token's KV ─────────────────────────────────────────────
        kv_cache.write_one(
            self.layer_idx, block_table, position,
            k.squeeze(0).squeeze(1),   # (num_kv, head_dim)
            v.squeeze(0).squeeze(1),
        )

        # ── gather full context (position+1 tokens) ──────────────────────────
        k_ctx, v_ctx = kv_cache.gather_kv(self.layer_idx, block_table, position + 1)
        # k_ctx: (num_kv, pos+1, head_dim)

        # Expand for GQA: (1, num_heads, pos+1, head_dim)
        k_e = k_ctx.unsqueeze(0).repeat_interleave(self.groups, dim=1)
        v_e = v_ctx.unsqueeze(0).repeat_interleave(self.groups, dim=1)

        # Single query → no causal mask needed
        out = F.scaled_dot_product_attention(q, k_e, v_e, is_causal=False)
        return self.o_proj(out.transpose(1, 2).contiguous().view(1, 1, -1))


# ── SwiGLU FFN ───────────────────────────────────────────────────────────────

class SwiGLUFFN(nn.Module):
    def __init__(self, model_cfg: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(model_cfg.hidden_size, model_cfg.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(model_cfg.hidden_size, model_cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(model_cfg.intermediate_size, model_cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ── Transformer Layer ─────────────────────────────────────────────────────────

class TransformerLayer(nn.Module):
    def __init__(self, model_cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn  = GQA(model_cfg, layer_idx)
        self.mlp        = SwiGLUFFN(model_cfg)
        self.input_layernorm          = RMSNorm(model_cfg.hidden_size, model_cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(model_cfg.hidden_size, model_cfg.rms_norm_eps)

    def _ffn(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.post_attention_layernorm(x))

    # ── standard path ────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, mask)
        return self._ffn(x)

    # ── cached paths ──────────────────────────────────────────────────────────

    def forward_prefill(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: "KVCachePool",
        block_table: List[int],
    ) -> torch.Tensor:
        x = x + self.self_attn.forward_prefill(self.input_layernorm(x), cos, sin, kv_cache, block_table)
        return self._ffn(x)

    def forward_decode(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: "KVCachePool",
        block_table: List[int],
        position: int,
    ) -> torch.Tensor:
        x = x + self.self_attn.forward_decode(self.input_layernorm(x), cos, sin, kv_cache, block_table, position)
        return self._ffn(x)


# ── Llama Model ───────────────────────────────────────────────────────────────

class LlamaModel(nn.Module):
    def __init__(self, model_cfg: ModelConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(model_cfg.vocab_size, model_cfg.hidden_size)
        self.layers       = nn.ModuleList(
            [TransformerLayer(model_cfg, i) for i in range(model_cfg.num_hidden_layers)]
        )
        self.norm       = RMSNorm(model_cfg.hidden_size, model_cfg.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(model_cfg)
        self.lm_head    = nn.Linear(model_cfg.hidden_size, model_cfg.vocab_size, bias=False)

    # ── weight loading ────────────────────────────────────────────────────────

    def load_hf_weights(self, state_dict: dict) -> None:
        clean = {}
        for k, v in state_dict.items():
            clean[k[6:] if k.startswith("model.") else k] = v
        if "lm_head.weight" not in clean and "embed_tokens.weight" in clean:
            clean["lm_head.weight"] = clean["embed_tokens.weight"]
        missing, unexpected = self.load_state_dict(clean, strict=False)
        if missing:
            print(f"[warn] missing keys : {missing}")
        if unexpected:
            print(f"[warn] unexpected   : {unexpected}")

    # ── no-cache forward (testing / small sequences) ──────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(h, position_ids)
        for layer in self.layers:
            h = layer(h, cos, sin, mask)
        return self.lm_head(self.norm(h))

    # ── prefill (writes KV cache) ─────────────────────────────────────────────

    def prefill(
        self,
        tokens: List[int],
        kv_cache: "KVCachePool",
        block_table: List[int],
    ) -> torch.Tensor:
        """
        Full prompt forward. Fills KV cache for all prompt positions.

        Returns
        ───────
        logits : (vocab_size,)  — logits for the LAST token (used to sample first output)
        """
        device = next(self.parameters()).device
        ids  = torch.tensor(tokens, device=device).unsqueeze(0)
        pos  = torch.arange(len(tokens), device=device).unsqueeze(0)
        h    = self.embed_tokens(ids)
        cos, sin = self.rotary_emb(h, pos)
        for layer in self.layers:
            h = layer.forward_prefill(h, cos, sin, kv_cache, block_table)
        return self.lm_head(self.norm(h))[0, -1, :]   # (vocab_size,)

    # ── decode (single token, reads+writes KV cache) ──────────────────────────

    def decode_one(
        self,
        token: int,
        kv_cache: "KVCachePool",
        block_table: List[int],
        position: int,
    ) -> torch.Tensor:
        """
        Single decode step.

        `position` = number of tokens already in the KV cache before this call.
        We write the new token's KV at this position and attend over [0, position].

        Returns
        ───────
        logits : (vocab_size,)
        """
        device = next(self.parameters()).device
        ids = torch.tensor([[token]], device=device)
        pos = torch.tensor([[position]], device=device)
        h   = self.embed_tokens(ids)
        cos, sin = self.rotary_emb(h, pos)
        for layer in self.layers:
            h = layer.forward_decode(h, cos, sin, kv_cache, block_table, position)
        return self.lm_head(self.norm(h))[0, 0, :]    # (vocab_size,)
