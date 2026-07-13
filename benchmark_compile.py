"""
benchmark_compile.py — compare eager vs torch.compile throughput.

Usage:
    python benchmark_compile.py [--compile] [--mode reduce-overhead]

Both runs use 64 requests @ max_new_tokens=64, batch=16.
"""

import argparse
import asyncio
import statistics
import time

from config import EngineConfig, ModelConfig
from engine import LlamaEngine

MODEL_PATH = (
    "/home/kuroko/.cache/huggingface/hub/"
    "models--meta-llama--Llama-3.2-1B-Instruct/snapshots/"
    "9213176726f574b556790deb65791e0c5aa438b6"
)

PROMPTS = [
    "Tell me a short story about a fox.",
    "What is the capital of France?",
    "Explain quantum entanglement simply.",
    "Write a haiku about the sea.",
] * 16  # 64 requests total


async def run_bench(use_compile: bool, mode: str, max_batch: int = 16):
    model_cfg  = ModelConfig(hf_path=MODEL_PATH)
    engine_cfg = EngineConfig(
        max_batch_size=max_batch,
        max_seq_len=512,
        use_torch_compile=use_compile,
        compile_mode=mode,
    )

    engine = LlamaEngine(model_cfg, engine_cfg)
    await engine.start()          # warms up torch.compile if enabled

    latencies: list[float] = []
    total_tokens = 0

    t_wall_start = time.perf_counter()

    async def one_request(prompt: str):
        nonlocal total_tokens
        t0 = time.perf_counter()
        n  = 0
        async for chunk in engine.generate(prompt, max_new_tokens=64, temperature=0.0):
            if not chunk.finished:
                n += 1
        latencies.append(time.perf_counter() - t0)
        total_tokens += n

    await asyncio.gather(*[one_request(p) for p in PROMPTS])
    wall = time.perf_counter() - t_wall_start

    await engine.stop()

    n_req  = len(PROMPTS)
    lat_s  = sorted(latencies)
    p50    = statistics.median(lat_s)
    p95    = lat_s[int(0.95 * len(lat_s))]
    tps    = total_tokens / wall
    rps    = n_req / wall

    tag = f"compile({mode})" if use_compile else "eager"
    print(f"\n{'='*60}")
    print(f"  Mode             : {tag}")
    print(f"  Requests         : {n_req}")
    print(f"  Wall time        : {wall:.2f}s")
    print(f"  Tokens generated : {total_tokens}")
    print(f"  Throughput       : {tps:.1f} tok/s  |  {rps:.2f} req/s")
    print(f"  Avg latency      : {sum(latencies)/len(latencies)*1000:.0f}ms")
    print(f"  p50 latency      : {p50*1000:.0f}ms")
    print(f"  p95 latency      : {p95*1000:.0f}ms")
    print(f"{'='*60}")
    return tps


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--compile", action="store_true", help="Enable torch.compile")
    ap.add_argument("--mode", default="reduce-overhead",
                    choices=["default", "reduce-overhead", "max-autotune"])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--both", action="store_true", help="Run eager then compiled back-to-back")
    args = ap.parse_args()

    if args.both:
        print("\n>>> Running EAGER baseline ...")
        eager_tps = asyncio.run(run_bench(False, args.mode, args.batch))

        print("\n>>> Running COMPILED ...")
        comp_tps  = asyncio.run(run_bench(True, args.mode, args.batch))

        delta = (comp_tps - eager_tps) / eager_tps * 100
        print(f"\n  Speedup: {delta:+.1f}% ({eager_tps:.1f} → {comp_tps:.1f} tok/s)")
    else:
        asyncio.run(run_bench(args.compile, args.mode, args.batch))
