"""
Continuous-batching scheduler.

How it works
────────────
Each request goes through these states:

  WAITING → PREFILL → RUNNING → DONE

  WAITING  : queued, blocks not yet allocated
  PREFILL  : admitted this step; engine will run its full prompt forward
  RUNNING  : decode phase; one new token generated per engine step
  DONE     : hit EOS or max_new_tokens; blocks freed

The Scheduler fills slots greedily from the waiting queue on each call
to schedule(), up to max_running.

Future extensibility
────────────────────
• Preemption
    When memory is tight, evict low-priority RUNNING requests (swap their
    blocks to CPU), freeing GPU blocks for high-priority requests.
• Chunked Prefill
    Split long prompts into fixed-size chunks processed over multiple steps,
    interleaved with decode steps to reduce time-to-first-token.
• Prefix / Radix Caching
    Before allocating fresh blocks, walk a RadixTree of cached prefixes.
    If a prefix matches, reuse its blocks (ref-count++) and only prefill
    the suffix — dramatically cutting both latency and memory.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import AsyncIterator, Deque, List, Optional

from core.kv_cache import KVCacheBackend


class RequestStatus(Enum):
    WAITING = auto()
    PREFILL = auto()
    RUNNING = auto()
    DONE    = auto()


@dataclass
class Request:
    prompt_tokens:  List[int]
    max_new_tokens: int   = 128
    temperature:    float = 0.8
    top_p:          float = 0.95
    top_k:          int   = 40

    # ── set by scheduler ────────────────────────────────────────────────────
    request_id:    str           = field(default_factory=lambda: uuid.uuid4().hex[:8])
    status:        RequestStatus = field(default=RequestStatus.WAITING)
    output_tokens: List[int]     = field(default_factory=list)
    block_table:   List[int]     = field(default_factory=list)

    # ── streaming ────────────────────────────────────────────────────────────
    # Token IDs are pushed here; None is the end sentinel.
    _queue: asyncio.Queue = field(default_factory=asyncio.Queue, repr=False)

    def __post_init__(self) -> None:
        self._num_prompt_tokens = len(self.prompt_tokens)

    # ── properties ──────────────────────────────────────────────────────────

    @property
    def num_prompt_tokens(self) -> int:
        return self._num_prompt_tokens

    @property
    def num_kv_entries(self) -> int:
        """
        Number of token KV pairs currently written in the cache.

        During decode, output_tokens[-1] is the *current input* token
        that will be written this step — it is NOT in the cache yet.
        So entries = prompt_tokens + (output_tokens generated so far - 1).

        Timeline:
          after prefill : output=[t0]        → num_kv_entries = P
          after step 1  : output=[t0,t1]     → num_kv_entries = P+1
          after step 2  : output=[t0,t1,t2]  → num_kv_entries = P+2
        """
        return self._num_prompt_tokens + max(0, len(self.output_tokens) - 1)

    # ── output helpers ──────────────────────────────────────────────────────

    def push_token(self, token_id: int) -> None:
        self.output_tokens.append(token_id)
        self._queue.put_nowait(token_id)

    def mark_done(self) -> None:
        self.status = RequestStatus.DONE
        self._queue.put_nowait(None)   # end sentinel

    async def stream(self) -> AsyncIterator[int]:
        """Async generator — yields token IDs as they are generated."""
        while True:
            tok = await self._queue.get()
            if tok is None:
                return
            yield tok


class Scheduler:
    """
    Greedy continuous-batching scheduler.

    On every call to schedule():
    1. Admit WAITING requests into running slots (if blocks available)
    2. Return (prefill_batch, decode_batch) for this engine step
    """

    def __init__(
        self,
        kv_backend: KVCacheBackend,
        max_running: int = 4,
    ) -> None:
        self.kv_backend  = kv_backend
        self.max_running = max_running

        self.waiting: Deque[Request] = deque()
        self.running: List[Request]  = []

    # ── public API ──────────────────────────────────────────────────────────

    def add(self, request: Request) -> None:
        self.waiting.append(request)

    def schedule(self) -> tuple[List[Request], List[Request]]:
        """
        Returns (prefill_batch, decode_batch).

        Admission policy: fill slots greedily; stop if blocks are exhausted.
        Future: add priority, preemption, chunked-prefill logic here.
        """
        self._admit_waiting()
        prefill = [r for r in self.running if r.status == RequestStatus.PREFILL]
        decode  = [r for r in self.running if r.status == RequestStatus.RUNNING]
        return prefill, decode

    def finish(self, req: Request) -> None:
        req.mark_done()
        self.kv_backend.free(req.block_table)
        if req in self.running:
            self.running.remove(req)

    @property
    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    # ── internal ─────────────────────────────────────────────────────────────

    def _blocks_for(self, req: Request) -> int:
        total = req.num_prompt_tokens + req.max_new_tokens
        return self.kv_backend.blocks_needed(total)

    def _admit_waiting(self) -> None:
        while self.waiting and len(self.running) < self.max_running:
            req = self.waiting[0]
            needed = self._blocks_for(req)
            if self.kv_backend.num_free_blocks < needed:
                break     # not enough memory — stop admitting
            self.waiting.popleft()
            req.block_table = self.kv_backend.alloc(needed)
            req.status = RequestStatus.PREFILL
            self.running.append(req)
