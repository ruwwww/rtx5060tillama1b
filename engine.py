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
import os
from dataclasses import dataclass
from typing import AsyncIterator, List, Optional

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from config import EngineConfig, ModelConfig
from core.kv_cache import BlockAllocator, KVCachePool
from core.llama import LlamaModel
from core.scheduler import Request, RequestStatus, Scheduler


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

        self.model    = self._load_model()
        self.kv_cache, self.allocator = self._init_kv_cache()
        self.scheduler = Scheduler(self.allocator, max_running=engine_cfg.max_batch_size)

        # Dedicated inference thread — keeps PyTorch compute off the event loop
        # so the event loop can concurrently handle streaming consumers and
        # new request submissions while GPU work is in flight.
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="llama-infer"
        )
        self._loop_task: Optional[asyncio.Task] = None

    # ── init helpers ─────────────────────────────────────────────────────────

    def _load_model(self) -> LlamaModel:
        path = os.path.join(self.model_cfg.hf_path, "model.safetensors")
        print(f"[engine] loading weights from {path}")
        state = load_file(path, device="cpu")
        state = {k: v.to(torch.bfloat16) for k, v in state.items()}

        model = LlamaModel(self.model_cfg).to(dtype=torch.bfloat16)
        model.load_hf_weights(state)
        del state
        model.to(self.device)
        model.eval()
        print("[engine] model ready")
        return model

    def _init_kv_cache(self) -> tuple[KVCachePool, BlockAllocator]:
        # Pre-allocate enough blocks for max_batch * max_seq_len tokens
        total_tokens = self.engine_cfg.max_batch_size * self.engine_cfg.max_seq_len
        num_blocks   = total_tokens // self.BLOCK_SIZE + 32   # +32 headroom

        allocator = BlockAllocator(num_blocks, self.BLOCK_SIZE)
        pool = KVCachePool(
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
        return pool, allocator

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Launch the background continuous-batching loop."""
        self._loop_task = asyncio.get_event_loop().create_task(self._run_loop())

    async def stop(self) -> None:
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass

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
        results: List[tuple] = []

        # ── prefill ───────────────────────────────────────────────────────────────
        for req in prefill_reqs:
            with torch.inference_mode():
                logits = self.model.prefill(
                    req.prompt_tokens, self.kv_cache, req.block_table
                )
            results.append((req, self._sample(logits, req), 'prefill'))

        # ── batched decode ───────────────────────────────────────────────────────
        active = [r for r in decode_reqs if r.status == RequestStatus.RUNNING]
        if active:
            tokens       = [req.output_tokens[-1] for req in active]
            positions    = [req.num_kv_entries     for req in active]
            block_tables = [req.block_table        for req in active]

            with torch.inference_mode():
                all_logits = self.model.decode_batch(
                    tokens, self.kv_cache, block_tables, positions
                )

            for i, req in enumerate(active):
                results.append((req, self._sample(all_logits[i], req), 'decode'))

        return results

    # ── async step (event loop: schedule → thread → dispatch) ───────────────

    async def _step(self) -> None:
        # 1. Schedule — fast list ops, safe on event loop thread
        prefill_reqs, decode_reqs = self.scheduler.schedule()
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
