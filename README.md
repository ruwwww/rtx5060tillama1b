# RTX 5060 Ti Llama 1B

A minimalistic pure Python/PyTorch inference engine for Llama 3.2 1B (safetensors format), with a stunning glassmorphic chat interface served locally.

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
