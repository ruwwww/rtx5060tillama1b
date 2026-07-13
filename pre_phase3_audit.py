"""
pre_phase3_audit.py
--------------------
Answers the 4 staff-engineer questions before Phase 3 implementation:

  Q1. Is req.block_table a Python list or GPU tensor?
  Q2. Can FlashInfer plan() be skipped when batch composition is unchanged?
      Measure its per-call CPU cost.
  Q3. Can CUDA Graphs be keyed by batch_size?
      Measure decode step time at B=1,2,4,8,16 to see if per-B graphs make sense.
  Q4. Profile plan() vs run() individually.

Run:
    python pre_phase3_audit.py
"""

import sys
import time
import statistics
import asyncio

import torch

from config import EngineConfig, ModelConfig
from core.kv_cache import PagedKVBackend
from core.attention import FlashInferAttention

HF_PATH = (
    "/home/kuroko/.cache/huggingface/hub/"
    "models--meta-llama--Llama-3.2-1B-Instruct/snapshots/"
    "9213176726f574b556790deb65791e0c5aa438b6"
)

BLOCK_SIZE = 16
NUM_LAYERS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 64
HIDDEN = 2048

DEVICE = "cuda"


def sep(title=""):
    print(f"\n{'─'*60}")
    if title:
        print(f"  {title}")
        print(f"{'─'*60}")


# ── Q1: What type is req.block_table? ────────────────────────────────────────

def q1_block_table_type():
    sep("Q1 — req.block_table type audit")
    from core.scheduler import Request
    from core.kv_cache import PagedKVBackend

    backend = PagedKVBackend(
        num_layers=NUM_LAYERS, num_blocks=128, num_kv_heads=NUM_KV_HEADS,
        block_size=BLOCK_SIZE, head_dim=HEAD_DIM, device=DEVICE,
    )
    req = Request(prompt_tokens=[1, 2, 3, 4])

    # Simulate what scheduler._admit_waiting does:
    req.block_table = backend.alloc(4)

    print(f"  req.block_table type : {type(req.block_table)}")
    print(f"  req.block_table value: {req.block_table[:4]}...")

    is_list = isinstance(req.block_table, list)
    print(f"\n  Is Python list?      : {is_list}")
    if is_list:
        print("  [BLOCKER] Per-step torch.tensor(block_table) conversion is NEEDED today.")
        print("  Fix: store block_table_tensor on Request, allocated once, extended lazily.")
    else:
        print("  [OK] Already a tensor — can use .copy_() directly.")


# ── Q2/Q4: Measure FlashInfer plan() and run() CPU cost ──────────────────────

def q2_q4_flashinfer_plan_cost():
    sep("Q2 / Q4 — FlashInfer plan() vs run() CPU cost")
    try:
        import flashinfer
    except ImportError:
        print("  [SKIP] flashinfer not installed")
        return

    REPS = 200
    B    = 16
    SEQ  = 64   # tokens already in cache (simulate mid-decode)

    # Build a fake paged KV backend to get realistic block tables
    num_blocks = 512
    backend = PagedKVBackend(
        num_layers=NUM_LAYERS, num_blocks=num_blocks, num_kv_heads=NUM_KV_HEADS,
        block_size=BLOCK_SIZE, head_dim=HEAD_DIM, device=DEVICE,
    )

    blocks_per_seq = (SEQ + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_tables = [backend.alloc(blocks_per_seq) for _ in range(B)]
    positions    = [SEQ - 1] * B   # each seq has SEQ tokens in cache

    # Build FlashInfer decode wrapper (same as FlashInferAttention does)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD")

    # Precompute indptr / indices / last_page_len exactly as attention.py does
    def make_plan_args():
        """Build the plan() arguments from block_tables + positions."""
        lengths = [len(tbl) for tbl in block_tables]
        indptr_list = [0]
        for l in lengths:
            indptr_list.append(indptr_list[-1] + l)
        indptr = torch.tensor(indptr_list, device=DEVICE, dtype=torch.int32)

        flat = []
        for tbl in block_tables:
            flat.extend(tbl)
        indices = torch.tensor(flat, device=DEVICE, dtype=torch.int32)

        seq_lens = [p + 1 for p in positions]
        last_page = torch.tensor(
            [((sl - 1) % BLOCK_SIZE) + 1 for sl in seq_lens],
            device=DEVICE, dtype=torch.int32,
        )
        return indptr, indices, last_page

    num_q_heads = 32
    indptr, indices, last_page = make_plan_args()

    # Dummy Q
    q = torch.randn(B, num_q_heads, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
    kv_data = (
        backend.k_pools[0],  # (num_blocks, block_size, num_kv_heads, head_dim)
        backend.v_pools[0],
    )

    torch.cuda.synchronize()

    # ── Measure plan() CPU cost ──────────────────────────────────────────────
    plan_times = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        wrapper.plan(
            indptr=indptr,
            indices=indices,
            last_page_len=last_page,
            num_qo_heads=num_q_heads,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            page_size=BLOCK_SIZE,
            q_data_type=torch.bfloat16,
        )
        torch.cuda.synchronize()
        plan_times.append((time.perf_counter() - t0) * 1e6)  # µs

    plan_p50 = statistics.median(plan_times)
    plan_p95 = sorted(plan_times)[int(0.95 * len(plan_times))]
    print(f"\n  plan() @ B={B}, seq_len={SEQ}")
    print(f"    p50 = {plan_p50:.1f} µs")
    print(f"    p95 = {plan_p95:.1f} µs")
    print(f"    min = {min(plan_times):.1f} µs")
    print(f"    max = {max(plan_times):.1f} µs")

    if plan_p50 < 100:
        verdict = "CHEAP (<100µs) — not a bottleneck, but still worth skipping when batch is unchanged."
    elif plan_p50 < 500:
        verdict = "MODERATE (100–500µs) — worth caching/skipping when batch composition unchanged."
    else:
        verdict = "EXPENSIVE (>500µs) — must be optimized."
    print(f"  Verdict: {verdict}")

    # ── Measure run() GPU latency ─────────────────────────────────────────────
    # Re-plan once before measuring run()
    wrapper.plan(
        indptr=indptr,
        indices=indices,
        last_page_len=last_page,
        num_qo_heads=num_q_heads,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        page_size=BLOCK_SIZE,
        q_data_type=torch.bfloat16,
    )

    # Warmup
    for _ in range(5):
        _ = wrapper.run(q, kv_data)
    torch.cuda.synchronize()

    run_times = []
    for _ in range(REPS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = wrapper.run(q, kv_data)
        torch.cuda.synchronize()
        run_times.append((time.perf_counter() - t0) * 1e6)

    run_p50 = statistics.median(run_times)
    run_p95 = sorted(run_times)[int(0.95 * len(run_times))]
    print(f"\n  run() @ B={B}, seq_len={SEQ}")
    print(f"    p50 = {run_p50:.1f} µs")
    print(f"    p95 = {run_p95:.1f} µs")

    print(f"\n  plan/run ratio: plan is {plan_p50/run_p50:.1f}x the GPU kernel time")

    # ── Q2: Can plan() be skipped when batch unchanged? ─────────────────────
    sep("Q2 — Can plan() be skipped when batch composition is unchanged?")
    print("  FlashInfer plan() sets up metadata (indptr, paged indices).")
    print("  It MUST be called when ANY of these change:")
    print("    - batch size B")
    print("    - seq_lens (change every decode step: +1 each)")
    print("    - block_tables (change when new blocks allocated)")
    print()
    print("  seq_lens change EVERY step (each request grows by 1 token).")
    print("  Therefore plan() cannot be fully skipped.")
    print()
    print("  However: plan() CAN be avoided inside the CUDA graph itself.")
    print("  The graph only captures run(). plan() runs on CPU before replay.")
    print("  This is exactly the architecture we already proposed — confirmed correct.")


# ── Q3: Per-batch-size graph cache analysis ───────────────────────────────────

def q3_per_batchsize_graph_cache():
    sep("Q3 — Per-batch-size graph cache: does B matter for step time?")

    # We'll time a real decode_batch call at different B values
    # to see if per-B graphs would save meaningful compute vs MAX_B padding.
    from core.llama import LlamaModel
    from core.attention import FlashInferAttention, PyTorchAttention
    from safetensors.torch import load_file
    import os

    print("  Loading model for timing...")
    state = load_file(os.path.join(HF_PATH, "model.safetensors"), device="cpu")
    from config import ModelConfig
    model_cfg = ModelConfig(hf_path=HF_PATH)

    attn_backend = PyTorchAttention()  # use SDPA to avoid FlashInfer plan() overhead here
    model = LlamaModel(model_cfg, attn_backend=attn_backend).to(dtype=torch.bfloat16, device=DEVICE)
    model.load_hf_weights({k: v.to(torch.bfloat16) for k, v in state.items()})
    del state
    model.eval()
    torch.cuda.synchronize()

    backend = PagedKVBackend(
        num_layers=NUM_LAYERS, num_blocks=1024, num_kv_heads=NUM_KV_HEADS,
        block_size=BLOCK_SIZE, head_dim=HEAD_DIM, device=DEVICE,
    )

    SEQ_LEN = 32   # simulate mid-decode
    REPS    = 50
    batch_sizes = [1, 2, 4, 8, 16]
    results = {}

    print(f"  Measuring decode step latency vs batch size (seq_len={SEQ_LEN})...")
    print()

    for B in batch_sizes:
        # Allocate fake block tables for B sequences
        blocks_per_seq = (SEQ_LEN + BLOCK_SIZE - 1) // BLOCK_SIZE
        block_tables = [backend.alloc(blocks_per_seq) for _ in range(B)]
        positions    = [SEQ_LEN - 1] * B
        tokens       = [42] * B

        # Prefill fake KV entries so attention has something to read
        with torch.inference_mode():
            for layer_idx in range(NUM_LAYERS):
                for i in range(B):
                    k_fake = torch.randn(SEQ_LEN, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
                    v_fake = torch.randn_like(k_fake)
                    backend.write_many(layer_idx, block_tables[i], k_fake, v_fake)

        # Warmup
        with torch.inference_mode():
            for _ in range(3):
                _ = model.decode_batch(tokens, backend, block_tables, positions)
        torch.cuda.synchronize()

        # Timed
        times = []
        with torch.inference_mode():
            for _ in range(REPS):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = model.decode_batch(tokens, backend, block_tables, positions)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000)

        p50 = statistics.median(times)
        results[B] = p50
        print(f"    B={B:2d}  →  {p50:.2f} ms/step  ({1000/p50:.0f} steps/s)")

        # Free blocks before next iteration
        for t in block_tables:
            backend.free(t)

    print()
    # The key question: does B=16 take proportionally more than B=1?
    ratio_16_1 = results[16] / results[1]
    print(f"  B=16 / B=1 step time ratio: {ratio_16_1:.1f}x")
    if ratio_16_1 > 4:
        print("  CONCLUSION: Step time scales significantly with B.")
        print("  → Per-B graph cache is WORTH IT.")
        print("    Suggested buckets: [1, 2, 4, 8, 16]")
    elif ratio_16_1 > 2:
        print("  CONCLUSION: Moderate scaling. Per-B cache helps, especially at low load.")
        print("    Suggested buckets: [1, 4, 8, 16] or [2, 4, 8, 16]")
    else:
        print("  CONCLUSION: Step time doesn't scale much with B (GPU is saturated even at B=1).")
        print("  → A single MAX_B graph might be fine; per-B cache is optional.")


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Pre-Phase3 Audit  —  Answering 4 Staff Engineer Questions")
    print("=" * 60)

    q1_block_table_type()
    q2_q4_flashinfer_plan_cost()
    q3_per_batchsize_graph_cache()

    sep("Summary")
    print("  See output above for per-question conclusions.")
    print("  Use these to finalise Phase 3 design before implementation.")
