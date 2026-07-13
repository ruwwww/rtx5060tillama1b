"""
Concurrent batch test — 4 requests submitted simultaneously.
Verifies that:
  1. Continuous batching scheduler admits all 4 requests
  2. decode_batch() handles heterogeneous context lengths correctly
  3. Streaming output arrives independently per request
"""
import asyncio
import time

from config import EngineConfig, ModelConfig
from engine import LlamaEngine

HF_PATH = (
    "/home/kuroko/.cache/huggingface/hub/"
    "models--meta-llama--Llama-3.2-1B-Instruct/snapshots/"
    "9213176726f574b556790deb65791e0c5aa438b6"
)

PROMPTS = [
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nWhat is the capital of France?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nExplain what a GPU is in one sentence.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nWhat is 17 multiplied by 6?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nName three planets in the solar system.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
]


async def collect(label: str, gen) -> tuple[str, float]:
    t0 = time.perf_counter()
    parts = []
    async for chunk in gen:
        if chunk.text:
            parts.append(chunk.text)
    elapsed = time.perf_counter() - t0
    return "".join(parts), elapsed


async def main() -> None:
    model_cfg  = ModelConfig(hf_path=HF_PATH)
    engine_cfg = EngineConfig()

    engine = LlamaEngine(model_cfg, engine_cfg)
    await engine.start()

    print(f"\n[batch test] submitting {len(PROMPTS)} requests concurrently\n")
    t_start = time.perf_counter()

    # Submit all requests at once — scheduler will batch decode steps
    tasks = [
        asyncio.create_task(
            collect(f"Q{i}", engine.generate(p, max_new_tokens=32))
        )
        for i, p in enumerate(PROMPTS)
    ]
    results = await asyncio.gather(*tasks)
    total   = time.perf_counter() - t_start

    for i, (text, elapsed) in enumerate(results):
        short_prompt = PROMPTS[i].split("\n")[1]
        print(f"[Q{i}] {short_prompt}")
        print(f"  → {text.strip()}")
        print(f"  ({elapsed:.2f}s)\n")

    print(f"[batch test] total wall time: {total:.2f}s")
    await engine.stop()


if __name__ == "__main__":
    asyncio.run(main())