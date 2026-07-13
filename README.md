# RTX 5060 Ti Llama 1B

A local conversational application powered by Llama 3.2 1B, featuring a premium glassmorphic chat interface and a high-performance custom PyTorch inference engine.

## What is this?
This project provides a self-hosted, lightweight playground to interact with the Llama 3.2 1B model directly on your local system. It combines a modern, visually stunning web chat interface with an optimized back-end that manages text generation efficiently.

## Key Features
- **Web Chat Interface**: Local glassmorphic web UI for model interaction.
- **Streaming Generation**: Real-time word-by-word generation output.
- **Continuous Batching**: Custom request scheduler supporting concurrent requests.

## Performance Milestones & Architecture
Below is the generation throughput progression (measured on batch serving at batch size 16):

- **Baseline PyTorch**: ~70 tokens/sec.
- **Paged Attention (FlashInfer)**: ~470 tokens/sec (executes directly over paged 4D KV cache pools).
- **Bucketed CUDA Graphs & `torch.compile`**: **~830 tokens/sec** (replays execution paths in batch-size buckets, reducing CPU launch overhead to ~10 microseconds).
- **Zero-Allocation Decode**: Replaces dynamic runtime tensor allocations with pre-allocated GPU buffers.


## Setup

Ensure your local conda environment is active and dependencies are met:
```bash
pip install -e .
```

## Running the Server

Start the local server:
```bash
python -m rtx5060tillama1b.server --port 8000
```

Once running, open **`http://localhost:8000`** in your web browser to start chatting with the model.



