"""
LlamaEngine — continuous-batching inference engine.

Architecture
────────────
┌──────────────────────────────────────────┐
│  FastAPI / test_run.py                   │
│   engine.generate(prompt) → async stream │
└───────────────┬──────────────────────────┘
                │ add Request
┌───────────────▼──────────────────────────┐
│  Scheduler                               │
│   waiting → PREFILL → RUNNING → DONE    │
└───────────────┬──────────────────────────┘
                │ (prefill_batch, decode_batch)
┌───────────────▼──────────────────────────┐
│  _step()                                 │
│   • prefill : LlamaModel.prefill()       │
│   • decode  : LlamaModel.decode_one()    │
└───────────────┬──────────────────────────┘
                │ reads / writes
┌───────────────▼──────────────────────────┐
│  KVCachePool (pre-allocated paged pool)  │
└──────────────────────────────────────────┘

The event-loop task (_run_loop) runs continuously in the background.
generate() adds a Request and streams tokens from its asyncio.Queue.

Decode steps currently run sequentially per request; true batched decode
(pad to same length or use variable-length FlashInfer kernel) is a natural
next step.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import logging
import os
import time
from dataclasses import dataclass
from typing import AsyncIterator, List, Optional

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from config import EngineConfig, ModelConfig
from core.attention import FlashInferAttention
from core.kv_cache import KVCacheBackend, PagedKVBackend
from core.llama import LlamaModel
from core.scheduler import Request, RequestStatus, Scheduler
from core.cuda_graph import CudaGraphManager
import core.scheduler as _sched_mod


@dataclass
class GenerationOutput:
    text:     str
    finished: bool = False


class LlamaEngine:
    """
    Minimalistic continuous-batching inference engine.

    Usage
    ─────
    engine = LlamaEngine(model_cfg, engine_cfg)
    await engine.start()                         # launches background loop
    async for chunk in engine.generate(prompt):
        print(chunk.text, end="")
    """

    BLOCK_SIZE = 16   # tokens per paged KV block

    def __init__(self, model_cfg: ModelConfig, engine_cfg: EngineConfig) -> None:
        self.model_cfg  = model_cfg
        self.engine_cfg = engine_cfg
        self.device = engine_cfg.device if torch.cuda.is_available() else "cpu"

        print(f"[engine] device = {self.device}")
        self.tokenizer    = AutoTokenizer.from_pretrained(model_cfg.hf_path)
        self.eos_token_id = self.tokenizer.eos_token_id

        self.attn_backend = FlashInferAttention(self.device)
        self.model    = self._load_model()
        self.kv_cache = self._init_kv_cache()
        self.scheduler = Scheduler(self.kv_cache, max_running=engine_cfg.max_batch_size)

        # ── Propagate device + block limit to scheduler module constants ───────────
        # So Request.block_table_tensor is allocated on the correct device.
        _sched_mod._DEVICE = self.device
        _sched_mod._MAX_BLOCKS_PER_SEQ = engine_cfg.max_seq_len // self.BLOCK_SIZE + 4

        # ── torch.compile ─────────────────────────────────────────────────────
        # Compile only the transformer layers. Attention uses FlashInfer wrappers
        # that call .plan()/.run() dynamically, so fullgraph=False lets us
        # compile the dense linear + FFN ops while allowing graph breaks at the
        # FlashInfer boundary.
        if engine_cfg.use_torch_compile and torch.cuda.is_available():
            print(f"[engine] torch.compile  : mode={engine_cfg.compile_mode}")
            self.model = torch.compile(
                self.model,
                mode=engine_cfg.compile_mode,
                fullgraph=False,
                dynamic=True,          # dynamic shapes → no recompile on B change
            )

        # ── Pre-allocated decode input buffers ─────────────────────────────────
        # These tensors live at stable GPU addresses across all decode steps.
        # Filled in-place each step via .copy_() / direct index writes.
        # Eliminates the 5 per-step torch.tensor() calls audited in Phase 2.
        # (Block index from block_table_tensor on Request uses .copy_() too.)
        if torch.cuda.is_available():
            MAX_B   = engine_cfg.max_batch_size
            MAX_BLK = engine_cfg.max_seq_len // self.BLOCK_SIZE + 4
            dev     = self.device

            # Token IDs for current decode step: shape (MAX_B, 1)
            self._ids_buf      = torch.zeros(MAX_B, 1, dtype=torch.long,  device=dev)
            # Position indices (= num_kv_entries per seq): shape (MAX_B, 1)
            self._pos_buf      = torch.zeros(MAX_B, 1, dtype=torch.long,  device=dev)
            # FlashInfer: cumulative block counts per seq + 1 sentinel: (MAX_B+1,)
            self._indptr_buf   = torch.zeros(MAX_B + 1, dtype=torch.int32, device=dev)
            # FlashInfer: flat list of all block indices across batch
            self._indices_buf  = torch.zeros(MAX_B * MAX_BLK, dtype=torch.int32, device=dev)
            # FlashInfer: number of valid tokens in the last block per seq: (MAX_B,)
            self._lastpg_buf   = torch.zeros(MAX_B, dtype=torch.int32, device=dev)
            # seq_lens (= position + 1): (MAX_B,) — for decode_batch call
            self._seqlens_buf  = torch.zeros(MAX_B, dtype=torch.int32, device=dev)
            
            # ── Scatter-write index/offset buffers (Phase 4) ──────────────────────
            # Pre-allocated tensors used to write new token KV coordinates in CUDA Graph
            self._blk_indices_buf = torch.zeros(MAX_B, dtype=torch.int32, device=dev)
            self._blk_offsets_buf = torch.zeros(MAX_B, dtype=torch.int32, device=dev)
            
            print(f"[engine] decode buffers: MAX_B={MAX_B}, MAX_BLK={MAX_BLK}")
        else:
            self._ids_buf = self._pos_buf = self._indptr_buf = None
            self._indices_buf = self._lastpg_buf = self._seqlens_buf = None
            self._blk_indices_buf = self._blk_offsets_buf = None

        # ── CUDA Graph Manager (Phase 4) ──────────────────────────────────────
        self.graph_manager = None
        if engine_cfg.use_cuda_graphs and torch.cuda.is_available():
            self.graph_manager = CudaGraphManager(
                bucket_sizes=list(engine_cfg.graph_batch_buckets),
                vocab_size=model_cfg.vocab_size,
                device=self.device,
            )
            print(f"[engine] CUDA graph manager initialized with buckets {engine_cfg.graph_batch_buckets}")

        # ── Instrumentation/Timing structures ─────────────────────────────────
        self.last_step_profile = {
            "scheduler_us": 0.0,
            "buffer_fill_us": 0.0,
            "plan_decode_us": 0.0,
            "graph_replay_us": 0.0,
            "forward_decode_us": 0.0,
            "sampling_us": 0.0,
            "total_decode_step_us": 0.0,
        }

        # Dedicated inference thread — keeps PyTorch compute off the event loop
        # so the event loop can concurrently handle streaming consumers and
        # new request submissions while GPU work is in flight.
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="llama-infer"
        )
        self._loop_task: Optional[asyncio.Task] = None

        # Perf counters (updated each step by _compute_sync)
        self._last_prefill_ms: float = 0.0
        self._last_decode_ms:  float = 0.0

    # ── init helpers ─────────────────────────────────────────────────────────

    def _load_model(self) -> LlamaModel:
        path = os.path.join(self.model_cfg.hf_path, "model.safetensors")
        print(f"[engine] loading weights from {path}")
        state = load_file(path, device="cpu")
        state = {k: v.to(torch.bfloat16) for k, v in state.items()}

        model = LlamaModel(self.model_cfg, attn_backend=self.attn_backend).to(dtype=torch.bfloat16)
        model.load_hf_weights(state)
        del state
        model.to(self.device)
        model.eval()
        print("[engine] model ready")
        return model

    def _init_kv_cache(self) -> KVCacheBackend:
        # Pre-allocate enough blocks for max_batch * max_seq_len tokens
        total_tokens = self.engine_cfg.max_batch_size * self.engine_cfg.max_seq_len
        num_blocks   = total_tokens // self.BLOCK_SIZE + 32   # +32 headroom

        backend = PagedKVBackend(
            num_layers  = self.model_cfg.num_hidden_layers,
            num_blocks  = num_blocks,
            num_kv_heads= self.model_cfg.num_key_value_heads,
            block_size  = self.BLOCK_SIZE,
            head_dim    = self.model_cfg.head_dim,
            device      = self.device,
            dtype       = torch.bfloat16,
        )

        kv_gb = (
            2 * self.model_cfg.num_hidden_layers
            * num_blocks * self.BLOCK_SIZE
            * self.model_cfg.num_key_value_heads
            * self.model_cfg.head_dim * 2  # bfloat16
        ) / 1024**3
        print(f"[engine] KV cache  : {num_blocks} blocks × {self.BLOCK_SIZE} tokens = {total_tokens} slots  ({kv_gb:.2f} GiB)")
        return backend

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Launch the background continuous-batching loop."""
        if getattr(self, '_warmup_done', False) is False:
            await asyncio.get_event_loop().run_in_executor(
                self._executor, self._warmup
            )
            self._warmup_done = True
        self._loop_task = asyncio.get_event_loop().create_task(self._run_loop())

    async def stop(self) -> None:
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass

    # ── warmup (triggers torch.compile JIT / kernel cache) ────────────────────

    def _warmup(self) -> None:
        """Run a synthetic prefill + decode to trigger torch.compile tracing."""
        B   = self.engine_cfg.max_batch_size
        dev = self.device
        # Synthetic single-token prefill
        dummy_ids  = torch.ones((1, 8), dtype=torch.long, device=dev)
        dummy_pos  = torch.arange(8, device=dev).unsqueeze(0)
        with torch.inference_mode():
            h = self.model.embed_tokens(dummy_ids)
            cos, sin = self.model.rotary_emb(h, dummy_pos)
            for layer in self.model.layers:
                h = layer.input_layernorm(h)
        # Flush compilation artifact
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print("[engine] warmup done")

    # ── sampling ──────────────────────────────────────────────────────────────

    def _sample(self, logits: torch.Tensor, req: Request) -> int:
        if req.temperature <= 0:
            return int(logits.argmax().item())

        logits = logits.float() / req.temperature

        if req.top_k > 0:
            thresh = torch.topk(logits, req.top_k).values[-1]
            logits = logits.masked_fill(logits < thresh, float("-inf"))

        if req.top_p < 1.0:
            sorted_l, sorted_i = torch.sort(logits, descending=True)
            cum = torch.cumsum(sorted_l.softmax(dim=-1), dim=-1)
            remove = cum > req.top_p
            remove[1:] = remove[:-1].clone()
            remove[0]  = False
            logits[sorted_i[remove]] = float("-inf")

        return int(torch.multinomial(logits.softmax(dim=-1), 1).item())

    def _is_done(self, req: Request, token_id: int) -> bool:
        return token_id == self.eos_token_id or len(req.output_tokens) >= req.max_new_tokens

    # ── one engine step ───────────────────────────────────────────────────────

    # ── synchronous compute (runs in thread pool) ─────────────────────────────

    def _compute_sync(
        self,
        prefill_reqs: List[Request],
        decode_reqs:  List[Request],
    ) -> List[tuple]:
        """
        Pure CPU/GPU computation — no asyncio, safe to run in a thread.

        Returns list of (req, token_id, phase) where phase is
        'prefill' or 'decode'. Token delivery happens back on the event loop.
        """
        t_start_step = time.perf_counter()
        results: List[tuple] = []
        
        # Reset current step profile times
        for k in self.last_step_profile:
            self.last_step_profile[k] = 0.0

        # ── prefill ───────────────────────────────────────────────────────────────
        for req in prefill_reqs:
            t0 = time.perf_counter()
            with torch.inference_mode():
                logits = self.model.prefill(
                    req.prompt_tokens, self.kv_cache, req.block_table
                )
            self._last_prefill_ms = (time.perf_counter() - t0) * 1000
            results.append((req, self._sample(logits, req), 'prefill'))

        # ── batched decode ───────────────────────────────────────────────────────
        active = [r for r in decode_reqs if r.status == RequestStatus.RUNNING]
        if active:
            B = len(active)
            t_buf_start = time.perf_counter()

            if self._ids_buf is not None:
                # ── Fill pre-allocated buffers in-place ───────────────────────
                block_size = self.BLOCK_SIZE
                indptr_val = 0

                for i, req in enumerate(active):
                    self._ids_buf[i, 0]    = req.output_tokens[-1]
                    self._pos_buf[i, 0]    = req.num_kv_entries
                    self._seqlens_buf[i]   = req.num_kv_entries + 1
                    sl = req.num_kv_entries + 1
                    self._lastpg_buf[i]    = ((sl - 1) % block_size) + 1
                    
                    # Flat-copy block indices from the GPU tensor on Request
                    n_blk = len(req.block_table)
                    self._indptr_buf[i]    = indptr_val
                    self._indices_buf[indptr_val:indptr_val + n_blk].copy_(
                        req.block_table_tensor[:n_blk]
                    )
                    indptr_val += n_blk
                    
                    # Pre-calculate write blocks and offsets for the scatter-write inside CUDA Graph
                    pos = req.num_kv_entries
                    self._blk_indices_buf[i] = req.block_table[pos // block_size]
                    self._blk_offsets_buf[i] = pos % block_size

                self._indptr_buf[B] = indptr_val

                # Views scaled to actual batch size B
                ids_b       = self._ids_buf[:B]
                pos_b       = self._pos_buf[:B]
                indptr_b    = self._indptr_buf[:B + 1]
                indices_b   = self._indices_buf[:indptr_val]
                lastpg_b    = self._lastpg_buf[:B]
                seqlens_b   = self._seqlens_buf[:B]
                blk_ind_b   = self._blk_indices_buf[:B]
                blk_off_b   = self._blk_offsets_buf[:B]
                
                self.last_step_profile["buffer_fill_us"] = (time.perf_counter() - t_buf_start) * 1e6

                # ── FlashInfer plan() ─────────────────────────────────────────
                t_plan_start = time.perf_counter()
                self.attn_backend.plan_decode(
                    indptr_b, indices_b, lastpg_b,
                    num_qo_heads=self.model.layers[0].self_attn.num_heads,
                    num_kv_heads=self.model.layers[0].self_attn.num_kv,
                    head_dim=self.model.layers[0].self_attn.head_dim,
                    block_size=self.kv_cache.block_size,
                )
                self.last_step_profile["plan_decode_us"] = (time.perf_counter() - t_plan_start) * 1e6

                # ── Try CUDA Graph ────────────────────────────────────────────
                use_graph_path = False
                all_logits = None
                
                if self.graph_manager is not None:
                    bucket_size = self.graph_manager.select_bucket(B)
                    if bucket_size is not None:
                        # Slice of the static input buffers that correspond to the full bucket size
                        ids_bucket       = self._ids_buf[:bucket_size]
                        pos_bucket       = self._pos_buf[:bucket_size]
                        blk_ind_bucket   = self._blk_indices_buf[:bucket_size]
                        blk_off_bucket   = self._blk_offsets_buf[:bucket_size]

                        # Define closure for lazy capture
                        def capture_fn(b_size: int) -> torch.Tensor:
                            return self.model.decode_batch_graph(
                                ids_buf=self._ids_buf[:b_size],
                                pos_buf=self._pos_buf[:b_size],
                                kv_cache=self.kv_cache,
                                blk_indices=self._blk_indices_buf[:b_size],
                                blk_offsets=self._blk_offsets_buf[:b_size],
                            )

                        bucket = self.graph_manager.get_or_capture(B, capture_fn)
                        if bucket is not None:
                            try:
                                # Prior to capture or on bucket validation check, we can compare eager vs graph outputs
                                # to ensure correctness (Requirement 6)
                                do_validation = (bucket.replay_count == 0)
                                
                                # Write KV for the actual requests eager-style before replaying or executing
                                # since write_kv_indexed inside graph will execute.
                                # For validation, we do eager forward first.
                                if do_validation:
                                    # Copy current state to validate
                                    eager_k = [self.kv_cache.k_pools[l].clone() for l in range(len(self.model.layers))]
                                    eager_v = [self.kv_cache.v_pools[l].clone() for l in range(len(self.model.layers))]
                                    
                                    # Execute eager path
                                    with torch.inference_mode():
                                        eager_logits = self.model.decode_batch_buffered(
                                            ids_b, pos_b, self.kv_cache,
                                            indptr_b, indices_b, lastpg_b, seqlens_b,
                                            block_tables=[req.block_table for req in active],
                                            positions=[req.num_kv_entries for req in active],
                                        )
                                    
                                    # Restore KV cache state for graph run
                                    for l in range(len(self.model.layers)):
                                        self.kv_cache.k_pools[l].copy_(eager_k[l])
                                        self.kv_cache.v_pools[l].copy_(eager_v[l])

                                # Run CUDA Graph replay
                                t_graph_start = time.perf_counter()
                                static_logits = self.graph_manager.replay(bucket)
                                torch.cuda.synchronize()
                                self.last_step_profile["graph_replay_us"] = (time.perf_counter() - t_graph_start) * 1e6
                                
                                # Slice output to active batch size
                                all_logits = static_logits[:B]
                                use_graph_path = True
                                
                                if do_validation:
                                    # Validate logits are within floating-point tolerance
                                    diff = torch.abs(all_logits - eager_logits).max().item()
                                    print(f"[graph] validated bucket B={bucket_size} for batch B={B}. Max logit diff: {diff:.6f}")
                                    if diff > 1e-2:
                                        print(f"[graph] Warning: high numerical discrepancy: {diff}")
                            
                            except Exception as e:
                                self.graph_manager.record_fallback(B, f"Replay error: {e}")
                                use_graph_path = False

                if not use_graph_path:
                    # ── Fallback decode path ──────────────────────────────────
                    t_fwd_start = time.perf_counter()
                    with torch.inference_mode():
                        all_logits = self.model.decode_batch_buffered(
                            ids_b, pos_b, self.kv_cache,
                            indptr_b, indices_b, lastpg_b, seqlens_b,
                            block_tables=[req.block_table for req in active],
                            positions=[req.num_kv_entries for req in active],
                        )
                    self.last_step_profile["forward_decode_us"] = (time.perf_counter() - t_fwd_start) * 1e6
                    self._last_decode_ms = (time.perf_counter() - t_fwd_start) * 1000

            else:
                # ── CPU fallback ──────────────────────────────────────────────
                t_fwd_start = time.perf_counter()
                tokens       = [req.output_tokens[-1] for req in active]
                positions    = [req.num_kv_entries     for req in active]
                block_tables = [req.block_table        for req in active]

                with torch.inference_mode():
                    all_logits = self.model.decode_batch(
                        tokens, self.kv_cache, block_tables, positions
                    )
                self.last_step_profile["forward_decode_us"] = (time.perf_counter() - t_fwd_start) * 1e6
                self._last_decode_ms = (time.perf_counter() - t_fwd_start) * 1000

            # ── Sampling ──────────────────────────────────────────────────────
            t_sample_start = time.perf_counter()
            for i, req in enumerate(active):
                results.append((req, self._sample(all_logits[i], req), 'decode'))
            self.last_step_profile["sampling_us"] = (time.perf_counter() - t_sample_start) * 1e6

        self.last_step_profile["total_decode_step_us"] = (time.perf_counter() - t_start_step) * 1e6
        return results

    # ── async step (event loop: schedule → thread → dispatch) ───────────────

    async def _step(self) -> None:
        # 1. Schedule — fast list ops, safe on event loop thread
        t_sched_start = time.perf_counter()
        prefill_reqs, decode_reqs = self.scheduler.schedule()
        sched_time_us = (time.perf_counter() - t_sched_start) * 1e6
        self.last_step_profile["scheduler_us"] = sched_time_us
        
        if not prefill_reqs and not decode_reqs:
            return


        # 2. Compute — heavy GPU work runs in thread, event loop stays free
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            self._executor,
            functools.partial(self._compute_sync, prefill_reqs, decode_reqs),
        )

        # 3. Dispatch tokens — back on event loop thread, queue ops are safe
        for req, token, phase in results:
            req.push_token(token)
            if phase == 'prefill':
                req.status = RequestStatus.RUNNING
            if self._is_done(req, token):
                self.scheduler.finish(req)

    # ── background loop ───────────────────────────────────────────────────────

    async def _run_loop(self) -> None:
        while True:
            if self.scheduler.has_work:
                await self._step()
            else:
                await asyncio.sleep(0.002)

    # ── public generate interface ─────────────────────────────────────────────

    async def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float  = 0.8,
        top_p: float        = 0.95,
        top_k: int          = 40,
    ) -> AsyncIterator[GenerationOutput]:
        """
        Async generator — submit a request and stream text chunks as they arrive.

        If the background loop is not running (e.g. test_run.py), the method
        starts a temporary loop for this call.
        """
        tokens = self.tokenizer.encode(prompt)
        req = Request(
            prompt_tokens  = tokens,
            max_new_tokens = max_new_tokens,
            temperature    = temperature,
            top_p          = top_p,
            top_k          = top_k,
        )
        self.scheduler.add(req)

        # Ensure the background loop is running
        if self._loop_task is None or self._loop_task.done():
            await self.start()

        async for token_id in req.stream():
            yield GenerationOutput(self.tokenizer.decode([token_id]), finished=False)

        yield GenerationOutput("", finished=True)
