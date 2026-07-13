# RTX 5060 Ti Llama 1B

A minimalistic pure Python/PyTorch inference engine for Llama 3.2 1B (safetensors format), with a stunning glassmorphic chat interface served locally.

## Features & Optimizations
- **FlashInfer Attention**: Direct attention on paged 4D KV cache layout.
- **Zero-Allocation Decode**: Replaces hot-path dynamic tensor creations with stable pre-allocated GPU buffers.
- **`torch.compile` Integration**: Optimizes Linear/FFN operations using TorchInductor.
- **CUDA Graph Manager (`CudaGraphManager`)**: Decoupled manager supporting lazy graph capture and replay for power-of-two batch size buckets (`1, 2, 4, 8, 12, 16`), achieving up to **830 tok/s** (+70% throughput, -58% p50 latency).
- **Graceful Fallbacks**: Automatically falls back to the eager buffered decode path if runtime conditions change.

## Setup

Ensure your local conda environment is active and dependencies are met:
```bash
pip install -e .
```

## Running the Server

Run the local inference server:
```bash
python -m rtx5060tillama1b.server --port 8000
```

Open `http://localhost:8000` in your browser to interact with the model using the premium UI.

