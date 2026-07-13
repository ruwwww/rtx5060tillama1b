from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    hidden_size: int = 2048
    num_hidden_layers: int = 16
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    intermediate_size: int = 8192
    vocab_size: int = 128256
    max_position_embeddings: int = 131072
    rope_theta: float = 500000.0
    rms_norm_eps: float = 1e-5
    head_dim: int = 64
    dtype: str = 'bfloat16'
    
    hf_path: Optional[str] = None
    
    # RoPE scaling properties for Llama 3.2
    rope_scaling_factor: float = 32.0
    rope_scaling_low_freq_factor: float = 1.0
    rope_scaling_high_freq_factor: float = 4.0
    rope_scaling_original_context: int = 8192
    rope_scaling_type: Optional[str] = "llama3"


@dataclass
class EngineConfig:
    max_batch_size: int = 4
    max_seq_len: int = 2048
    device: str = 'cuda'

    # torch.compile settings
    use_torch_compile: bool = False
    compile_mode: str = "reduce-overhead"   # or "default" / "max-autotune"

    # CUDA Graph settings (Phase 4)
    # Capture one graph per bucket; replay the smallest bucket >= actual B.
    # Buckets empirically justified by Q3 audit: B=1,4,8,16 cover 2.5x step time range.
    use_cuda_graphs: bool = False
    graph_batch_buckets: tuple = (1, 4, 8, 16)   # must be sorted ascending

