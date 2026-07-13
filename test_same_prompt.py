import asyncio
from config import EngineConfig, ModelConfig
from engine import LlamaEngine

HF_PATH = (
    "/home/kuroko/.cache/huggingface/hub/"
    "models--meta-llama--Llama-3.2-1B-Instruct/snapshots/"
    "9213176726f574b556790deb65791e0c5aa438b6"
)

async def main():
    model_cfg = ModelConfig(hf_path=HF_PATH)
    engine_cfg = EngineConfig()
    engine = LlamaEngine(model_cfg, engine_cfg)
    await engine.start()

    prompt = (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
        "What is the capital of France?<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )

    print("\n--- RUN 1 ---")
    async for chunk in engine.generate(prompt, max_new_tokens=16, temperature=0.0):
        print(chunk.text, end="", flush=True)
    print("\n")

    print("--- RUN 2 (Same Prompt) ---")
    async for chunk in engine.generate(prompt, max_new_tokens=16, temperature=0.0):
        print(chunk.text, end="", flush=True)
    print("\n")

    await engine.stop()

if __name__ == "__main__":
    asyncio.run(main())
