from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x_normed).to(input_dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, model_cfg: ModelConfig):
        super().__init__()
        self.head_dim = model_cfg.head_dim
        self.max_position_embeddings = model_cfg.max_position_embeddings
        self.theta = model_cfg.rope_theta

        # Compute base frequencies
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))

        scaling_type = model_cfg.rope_scaling_type
        factor = model_cfg.rope_scaling_factor
        low_freq_factor = model_cfg.rope_scaling_low_freq_factor
        high_freq_factor = model_cfg.rope_scaling_high_freq_factor
        old_context_len = model_cfg.rope_scaling_original_context

        if scaling_type == "llama3":
            wavelen = 2 * math.pi / inv_freq
            low_freq_wavelen = old_context_len / low_freq_factor
            high_freq_wavelen = old_context_len / high_freq_factor

            inv_freq_scaled = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)

            smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            smooth = torch.clamp(smooth, 0.0, 1.0)
            smoothed_inv_freq = (1 - smooth) * (inv_freq_scaled / factor) + smooth * inv_freq_scaled
            
            is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
            inv_freq = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_scaled)

        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, -1)
        pos = position_ids.float()[:, None, :]
        freqs = (inv_freq @ pos).transpose(1, 2)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    @staticmethod
    def apply_rotary_pos_emb(
        q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        q_embed = (q * cos) + (RotaryEmbedding.rotate_half(q) * sin)
        k_embed = (k * cos) + (RotaryEmbedding.rotate_half(k) * sin)
        return q_embed, k_embed


class GQA(nn.Module):
    def __init__(self, model_cfg: ModelConfig):
        super().__init__()
        self.hidden_size = model_cfg.hidden_size
        self.num_heads = model_cfg.num_attention_heads
        self.num_kv_heads = model_cfg.num_key_value_heads
        self.head_dim = model_cfg.head_dim
        self.num_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos = cos[:, :, :self.head_dim]
        sin = sin[:, :, :self.head_dim]
        q, k = RotaryEmbedding.apply_rotary_pos_emb(q, k, cos, sin)

        k = k.repeat_interleave(self.num_groups, dim=1)
        v = v.repeat_interleave(self.num_groups, dim=1)

        output = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, is_causal=(q_len > 1))
        output = output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return self.o_proj(output)


class SwiGLUFFN(nn.Module):
    def __init__(self, model_cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(model_cfg.hidden_size, model_cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(model_cfg.hidden_size, model_cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(model_cfg.intermediate_size, model_cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerLayer(nn.Module):
    def __init__(self, model_cfg: ModelConfig, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = GQA(model_cfg)
        self.mlp = SwiGLUFFN(model_cfg)
        self.input_layernorm = RMSNorm(model_cfg.hidden_size, model_cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(model_cfg.hidden_size, model_cfg.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cos, sin, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class LlamaModel(nn.Module):
    def __init__(self, model_cfg: ModelConfig):
        super().__init__()
        self.model_cfg = model_cfg
        self.embed_tokens = nn.Embedding(model_cfg.vocab_size, model_cfg.hidden_size)
        self.layers = nn.ModuleList([
            TransformerLayer(model_cfg, i) for i in range(model_cfg.num_hidden_layers)
        ])
        self.norm = RMSNorm(model_cfg.hidden_size, model_cfg.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(model_cfg)
        self.lm_head = nn.Linear(model_cfg.hidden_size, model_cfg.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(hidden_states, position_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin, attention_mask)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

    def load_hf_weights(self, state_dict: dict) -> None:
        clean_state = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                clean_state[k[6:]] = v
            else:
                clean_state[k] = v
        if 'lm_head.weight' not in clean_state and 'embed_tokens.weight' in clean_state:
            clean_state['lm_head.weight'] = clean_state['embed_tokens.weight']
        missing, unexpected = self.load_state_dict(clean_state, strict=False)
        if missing:
            print(f'Missing keys: {missing}')
        if unexpected:
            print(f'Unexpected keys: {unexpected}')
