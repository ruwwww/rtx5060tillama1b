"""
Heavy‑load simulator with constant concurrency (4 requests in flight).
- Maintains exactly 4 simultaneous requests at all times.
- As soon as one request finishes, a new one is launched.
- Runs until a total of N requests have been processed (N = 4 * 8 = 32).
Measures throughput, average latency, token generation rate, and tail latencies.
"""
import asyncio
import time
import random
import statistics
from typing import List, Tuple

from config import EngineConfig, ModelConfig
from engine import LlamaEngine

HF_PATH = (
    "/home/kuroko/.cache/huggingface/hub/"
    "models--meta-llama--Llama-3.2-1B-Instruct/snapshots/"
    "9213176726f574b556790deb65791e0c5aa438b6"
)

# Larger pool of prompts (all follow the same chat template)
PROMPT_POOL = [
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nWhat is the capital of France?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nExplain what a GPU is in one sentence.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nWhat is 17 multiplied by 6?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nName three planets in the solar system.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nWrite a haiku about autumn.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nWhat is the boiling point of water in Celsius?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nWho wrote '1984'?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nList three colors of the rainbow.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nWhat is the square root of 144?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nWhat is the tallest mountain on Earth?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nTranslate 'hello' into Spanish.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nWhat is the chemical symbol for gold?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nHow many continents are there?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nWhat is the speed of light roughly?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nWhat is the capital of Japan?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nName a famous scientist.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
]

async def collect(gen) -> Tuple[str, float]:
    """Collect streaming output and measure elapsed time for one request."""
    t0 = time.perf_counter()
    parts = []
    async for chunk in gen:
        if chunk.text:
            parts.append(chunk.text)
    elapsed = time.perf_counter() - t0
    return "".join(parts), elapsed

async def worker(engine, semaphore, request_id: int, prompt: str, max_tokens: int = 32):
    """
    Single request worker: acquires semaphore (to limit concurrency),
    runs the generation, and returns (request_id, elapsed, generated_text).
    """
    async with semaphore:
        start_time = time.perf_counter()
        text, elapsed = await collect(engine.generate(prompt, max_new_tokens=max_tokens))
        return request_id, elapsed, text

async def main():
    # ---- Configuration ----
    CONCURRENT_REQUESTS = 16      # fixed concurrency
    TOTAL_REQUESTS = 16 * 16       # 256 requests total (16 "rounds" of 16)
    MAX_NEW_TOKENS = 128

    model_cfg = ModelConfig(hf_path=HF_PATH)
    engine_cfg = EngineConfig()
    engine = LlamaEngine(model_cfg, engine_cfg)
    await engine.start()

    print(f"\n=== Heavy‑Load Simulator (Constant Concurrency) ===")
    print(f"Concurrent requests: {CONCURRENT_REQUESTS}")
    print(f"Total requests to process: {TOTAL_REQUESTS}")
    print(f"Max new tokens per request: {MAX_NEW_TOKENS}")
    print("Starting...\n")

    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

    # Prepare a list of prompts (randomly chosen, with replacement if pool too small)
    prompts = [random.choice(PROMPT_POOL) for _ in range(TOTAL_REQUESTS)]

    # We'll create all tasks upfront, but the semaphore will limit how many run concurrently.
    # However, the typical pattern is to launch tasks but have them wait on semaphore.
    # To truly maintain "as soon as one finishes, next starts", we can launch all tasks immediately,
    # but they will block on semaphore acquisition. This works because asyncio will schedule
    # the next waiter as soon as a semaphore is released.
    tasks = [
        asyncio.create_task(worker(engine, semaphore, i, prompts[i], max_tokens=MAX_NEW_TOKENS))
        for i in range(TOTAL_REQUESTS)
    ]

    # Wait for all tasks to complete and collect results
    t_start = time.perf_counter()
    results = await asyncio.gather(*tasks)
    total_time = time.perf_counter() - t_start

    # Process results
    latencies = [elapsed for _, elapsed, _ in results]
    texts = [text for _, _, text in results]
    total_tokens = sum(len(text.split()) for text in texts)   # approximate

    # Print some sample outputs (first few)
    print("Sample outputs (first 4 requests):")
    for i in range(min(4, len(results))):
        print(f"[Req {i}] {texts[i][:80]}...")
        print(f"  latency: {latencies[i]:.2f}s\n")

    # Statistics
    avg_lat = statistics.mean(latencies) if latencies else 0
    p50 = statistics.median(latencies) if latencies else 0
    p95 = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies) if latencies else 0
    p99 = statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else max(latencies) if latencies else 0

    throughput_tps = total_tokens / total_time if total_time > 0 else 0
    throughput_rps = TOTAL_REQUESTS / total_time if total_time > 0 else 0

    print("\n=== Summary ===")
    print(f"Total requests:        {TOTAL_REQUESTS}")
    print(f"Total wall time:       {total_time:.2f}s")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Average latency:       {avg_lat:.2f}s")
    print(f"Median latency (p50):  {p50:.2f}s")
    print(f"p95 latency:           {p95:.2f}s")
    print(f"p99 latency:           {p99:.2f}s")
    print(f"Throughput (tokens/s): {throughput_tps:.2f}")
    print(f"Throughput (req/s):    {throughput_rps:.2f}")

    await engine.stop()

if __name__ == "__main__":
    asyncio.run(main())