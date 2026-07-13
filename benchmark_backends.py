"""
Benchmark comparison between PyTorch (Gather + SDPA) and FlashInfer (Paged direct attention) backends.
Ensures a completely fair comparison by:
  1. Using the exact same PagedKVBackend for both.
  2. Disabling EOS early termination so every request generates exactly 64 tokens (decode-dominated).
  3. Running with identical concurrency (16 requests in flight, 128 requests total).
"""
import asyncio
import time
import random
import statistics
import torch

from config import EngineConfig, ModelConfig
from engine import LlamaEngine
from core.attention import PyTorchAttention, FlashInferAttention

HF_PATH = (
    "/home/kuroko/.cache/huggingface/hub/"
    "models--meta-llama--Llama-3.2-1B-Instruct/snapshots/"
    "9213176726f574b556790deb65791e0c5aa438b6"
)

PROMPT_POOL = [
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nWhat is the capital of France?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nExplain what a GPU is in one sentence.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nWhat is 17 multiplied by 6?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nName three planets in the solar system.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
]


def set_attention_backend(model, backend):
    model.attn_backend = backend
    for layer in model.layers:
        layer.self_attn.attn_backend = backend


async def collect(gen) -> int:
    tokens = 0
    async for chunk in gen:
        if chunk.text:
            tokens += 1
    return tokens


async def worker(engine, semaphore, prompt: str, max_tokens: int):
    async with semaphore:
        t0 = time.perf_counter()
        tokens = await collect(engine.generate(prompt, max_new_tokens=max_tokens))
        elapsed = time.perf_counter() - t0
        return tokens, elapsed


async def run_benchmark(engine, backend_name, backend_obj, num_requests=128, concurrency=16, tokens_per_req=64):
    set_attention_backend(engine.model, backend_obj)

    # Disable EOS checking to ensure constant decoding workload
    original_is_done = engine._is_done
    engine._is_done = lambda req, tok: len(req.output_tokens) >= req.max_new_tokens

    # Warmup step to eliminate JIT compilation latency from the measurement
    print(f"[{backend_name}] Warming up...")
    warmup_tasks = [
        asyncio.create_task(worker(engine, asyncio.Semaphore(4), random.choice(PROMPT_POOL), max_tokens=16))
        for _ in range(4)
    ]
    await asyncio.gather(*warmup_tasks)

    print(f"[{backend_name}] Running benchmark...")
    semaphore = asyncio.Semaphore(concurrency)
    prompts = [random.choice(PROMPT_POOL) for _ in range(num_requests)]

    t_start = time.perf_counter()
    tasks = [
        asyncio.create_task(worker(engine, semaphore, prompts[i], max_tokens=tokens_per_req))
        for i in range(num_requests)
    ]
    results = await asyncio.gather(*tasks)
    total_time = time.perf_counter() - t_start

    # Restore EOS checking
    engine._is_done = original_is_done

    total_tokens = sum(r[0] for r in results)
    latencies = [r[1] for r in results]

    return {
        "wall_time": total_time,
        "tokens": total_tokens,
        "tps": total_tokens / total_time,
        "req_s": num_requests / total_time,
        "avg_latency": statistics.mean(latencies),
        "p99_latency": statistics.quantiles(latencies, n=100)[98],
    }


async def main():
    model_cfg = ModelConfig(hf_path=HF_PATH)
    engine_cfg = EngineConfig()
    engine = LlamaEngine(model_cfg, engine_cfg)
    await engine.start()

    # 1. Benchmark PyTorch (SDPA + Gather)
    pytorch_res = await run_benchmark(
        engine, "PyTorch SDPA", PyTorchAttention(),
        num_requests=64, concurrency=16, tokens_per_req=64
    )

    # 2. Benchmark FlashInfer (Paged direct attention)
    flashinfer_res = await run_benchmark(
        engine, "FlashInfer", FlashInferAttention(engine.device),
        num_requests=64, concurrency=16, tokens_per_req=64
    )

    await engine.stop()

    print("\n" + "="*50)
    print("CONCISE BENCHMARK COMPARISON (64 tokens/req, 16 concurrency)")
    print("="*50)
    print(f"{'Metric':<25} | {'PyTorch SDPA':<15} | {'FlashInfer Paged':<18}")
    print("-"*50)
    print(f"{'Throughput (tokens/s)':<25} | {pytorch_res['tps']:<15.2f} | {flashinfer_res['tps']:<18.2f}")
    print(f"{'Throughput (req/s)':<25} | {pytorch_res['req_s']:<15.2f} | {flashinfer_res['req_s']:<18.2f}")
    print(f"{'Average Latency':<25} | {pytorch_res['avg_latency']:<15.2f}s | {flashinfer_res['avg_latency']:<18.2f}s")
    print(f"{'p99 Latency':<25} | {pytorch_res['p99_latency']:<15.2f}s | {flashinfer_res['p99_latency']:<18.2f}s")
    print("="*50)


if __name__ == "__main__":
    asyncio.run(main())
