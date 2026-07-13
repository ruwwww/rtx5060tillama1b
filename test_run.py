import asyncio
from config import EngineConfig, ModelConfig
from engine import LlamaEngine


async def test_inference():
    model_cfg = ModelConfig(
        hf_path="/home/kuroko/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6"
    )
    engine_cfg = EngineConfig()
    
    print("Loading engine...")
    engine = LlamaEngine(model_cfg, engine_cfg)
    
    print("\nRunning inference...")
    async for chunk in engine.generate("User: Hello, introduce yourself in one short sentence.\nAssistant:", max_new_tokens=32):
        print(chunk.text, end="", flush=True)
    print("\n\nSuccess!")


if __name__ == "__main__":
    asyncio.run(test_inference())
