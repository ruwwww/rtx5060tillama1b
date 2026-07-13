import argparse
import asyncio
import time
import psutil
import torch
import numpy as np
from config import EngineConfig, ModelConfig
from engine import LlamaEngine

HF = ('/home/kuroko/.cache/huggingface/hub/'
      'models--meta-llama--Llama-3.2-1B-Instruct/snapshots/'
      '9213176726f574b556790deb65791e0c5aa438b6')

async def run_benchmark(use_graphs: bool, use_compile: bool, batch_size: int, num_tokens: int):
    # Setup configs
    cfg = EngineConfig(
        max_batch_size=batch_size,
        use_torch_compile=use_compile,
        use_cuda_graphs=use_graphs,
        graph_batch_buckets=(1, 2, 4, 8, 12, 16)
    )
    model_cfg = ModelConfig(hf_path=HF)
    
    engine = LlamaEngine(model_cfg, cfg)
    await engine.start()

    # Pre-generate prompts to ensure concurrent dispatch
    prompts = [f"Provide a brief summary of computer history. Prompt ID {i}." for i in range(batch_size)]
    
    # Warmup step to capture graphs or compile model
    print(f"--- Running warmup pass for batch size {batch_size} (Graphs: {use_graphs}, Compile: {use_compile}) ---")
    warmup_tasks = [
        engine.generate(prompts[i % len(prompts)], max_new_tokens=4, temperature=0.0)
        for i in range(batch_size)
    ]
    
    async def consume(gen):
        async for chunk in gen:
            pass
            
    # Consume warmup
    generators = []
    for t in warmup_tasks:
        generators.append(consume(t))
    await asyncio.gather(*generators)
    
    # Wait for execution thread to finish
    await asyncio.sleep(0.5)
    
    # Now start actual benchmark and track CPU/GPU
    print(f"--- Starting benchmark evaluation ---")
    start_time = time.perf_counter()
    cpu_usages = []
    
    # Monitor CPU usage in a simple background task
    async def cpu_monitor():
        while True:
            cpu_usages.append(psutil.cpu_percent(interval=None))
            await asyncio.sleep(0.1)
            
    monitor_task = asyncio.create_task(cpu_monitor())
    
    tasks = [
        engine.generate(prompts[i % len(prompts)], max_new_tokens=num_tokens, temperature=0.0)
        for i in range(batch_size)
    ]
    
    token_latencies = []
    
    async def consume_measured(gen):
        last_t = time.perf_counter()
        async for chunk in gen:
            if not chunk.finished:
                now = time.perf_counter()
                token_latencies.append((now - last_t) * 1000) # ms
                last_t = now
                
    generators_measured = [consume_measured(t) for t in tasks]
    await asyncio.gather(*generators_measured)
    
    monitor_task.cancel()
    end_time = time.perf_counter()
    duration = end_time - start_time
    
    total_tokens = batch_size * num_tokens
    throughput = total_tokens / duration
    avg_cpu = np.mean(cpu_usages) if cpu_usages else 0.0
    
    # Retrieve stats
    hit_rate = 0.0
    capture_count = 0
    stats = {}
    if engine.graph_manager:
        stats = engine.graph_manager.stats()
        hit_rate = stats.get("graph_hit_rate", 0.0)
        capture_count = stats.get("total_captures", 0)
        
    print("\n============================================================")
    print(f"Mode             : graphs={use_graphs}, compile={use_compile}")
    print(f"Requests / Batch : {batch_size}")
    print(f"Wall time        : {duration:.2f}s")
    print(f"Tokens generated : {total_tokens}")
    print(f"Throughput       : {throughput:.2f} tok/s")
    print(f"Avg latency      : {np.mean(token_latencies):.2f}ms")
    print(f"p50 latency      : {np.percentile(token_latencies, 50):.2f}ms")
    print(f"p95 latency      : {np.percentile(token_latencies, 95):.2f}ms")
    print(f"Avg CPU utilization: {avg_cpu:.1f}%")
    if use_graphs:
        print(f"Graph hit rate   : {hit_rate:.1f}%")
        print(f"Graph capture cnt: {capture_count}")
        print(f"Buckets stats    : {stats.get('buckets')}")
    print("============================================================\n")
    
    await engine.stop()
    return {
        "throughput": throughput,
        "avg": np.mean(token_latencies),
        "p50": np.percentile(token_latencies, 50),
        "p95": np.percentile(token_latencies, 95),
        "cpu": avg_cpu,
        "hit_rate": hit_rate,
        "captures": capture_count,
        "stats": stats
    }

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--graphs", action="store_true")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    
    await run_benchmark(args.graphs, args.compile, args.batch, args.tokens)

if __name__ == "__main__":
    asyncio.run(main())
