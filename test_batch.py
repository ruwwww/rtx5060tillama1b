import asyncio
from config import EngineConfig, ModelConfig
from engine import LlamaEngine
HF_PATH = "/home/kuroko/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6"
async def run_request(engine, req_id, prompt):
    print(f"[Req {req_id}] Started")
    async for chunk in engine.generate(prompt, max_new_tokens=32):
        if chunk.text:
            print(f"[Req {req_id}]: {chunk.text.strip()}")
    print(f"[Req {req_id}] Finished")
async def main():
    model_cfg = ModelConfig(hf_path=HF_PATH)
    engine_cfg = EngineConfig()
    engine = LlamaEngine(model_cfg, engine_cfg)
    await engine.start()
    # Send multiple requests at once to trigger continuous batching
    await asyncio.gather(
        run_request(engine, 1, "User: Hello! Tell me a joke.\nAssistant:"),
        run_request(engine, 2, "User: What is 2+2?\nAssistant:"),
        run_request(engine, 3, "User: Define gravity.\nAssistant:"),
        run_request(engine, 4, "User: What is the capital of France?\nAssistant:"),
        run_request(engine, 5, "User: Write a short poem about the sea.\nAssistant:"),
    )
    await engine.stop()
if __name__ == "__main__":
    asyncio.run(main())