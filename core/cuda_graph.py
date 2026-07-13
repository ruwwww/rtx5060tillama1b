"""
core/cuda_graph.py — CudaGraphManager

Responsibility: graph capture, replay, bucket lifecycle, stats.

LlamaEngine calls:
    bucket = graph_manager.get_or_capture(B, capture_fn)
    if bucket:
        graph_manager.replay(bucket)
        logits = bucket.output_buf[:B]
    else:
        <fallback>

Design constraints:
  • Does not assume FlashInfer, SDPA, or any specific attention backend.
  • Does not know about requests, scheduling, or KV cache internals.
  • Captures lazily — only on first use of a bucket.
  • Always falls back gracefully; never raises through to the caller.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch

log = logging.getLogger(__name__)


@dataclass
class GraphBucket:
    """One captured CUDA graph for a fixed batch size."""
    bucket_size:    int
    graph:          torch.cuda.CUDAGraph
    output_buf:     torch.Tensor        # (bucket_size, vocab_size) — fixed address
    captured_at:    float = field(default_factory=time.time)
    replay_count:   int = 0
    fallback_count: int = 0             # times fallback was used FOR this bucket
    last_replay_us: float = 0.0


class CudaGraphManager:
    """
    Manages a pool of CUDA graphs, one per batch-size bucket.

    Parameters
    ----------
    bucket_sizes : iterable of int
        Batch sizes to capture graphs for (e.g. [1, 4, 8, 16]).
        Must all be <= max_batch_size.
    vocab_size : int
        Needed only for bookkeeping / output buffer shape validation.
    device : str
        CUDA device string.
    warmup_steps : int
        Number of warmup runs before each graph capture.
    """

    def __init__(
        self,
        bucket_sizes: List[int],
        vocab_size:   int,
        device:       str,
        warmup_steps: int = 3,
    ) -> None:
        self.bucket_sizes  = sorted(bucket_sizes)
        self.vocab_size    = vocab_size
        self.device        = device
        self.warmup_steps  = warmup_steps

        self._buckets: Dict[int, GraphBucket] = {}
        self._fallback_log: deque = deque(maxlen=256)
        self._total_replay   = 0
        self._total_fallback = 0
        self._total_capture  = 0

    # ── public API ────────────────────────────────────────────────────────────

    def select_bucket(self, B: int) -> Optional[int]:
        """Return the smallest bucket size >= B, or None if B > all buckets."""
        for bkt in self.bucket_sizes:
            if bkt >= B:
                return bkt
        return None

    def get_or_capture(
        self,
        B:          int,
        capture_fn: Callable[[int], torch.Tensor],
    ) -> Optional[GraphBucket]:
        """
        Return the GraphBucket for the appropriate bucket size.

        If the bucket hasn't been captured yet, capture it now (lazy).
        Returns None if B exceeds all bucket sizes or capture fails.

        Parameters
        ----------
        B          : actual batch size for this step
        capture_fn : callable(bucket_size) -> logits_tensor
                     Must be called inside torch.inference_mode().
                     Must return the logits tensor (bucket_size, vocab_size)
                     whose address becomes the stable output buffer.
        """
        bkt = self.select_bucket(B)
        if bkt is None:
            reason = f"B={B} exceeds max bucket {self.bucket_sizes[-1] if self.bucket_sizes else '?'}"
            self._log_fallback(reason)
            return None

        if bkt not in self._buckets:
            ok = self._lazy_capture(bkt, capture_fn)
            if not ok:
                return None

        return self._buckets.get(bkt)

    def replay(self, bucket: GraphBucket) -> torch.Tensor:
        """
        Replay the captured graph.

        NOTE: Caller must have already:
          1. Filled input buffers (ids, pos, write_blk, write_off)
          2. Called plan_decode() (FlashInfer metadata update)
        before calling this method.

        Returns the output_buf tensor — caller slices [:actual_B].
        """
        t0 = time.perf_counter()
        bucket.graph.replay()
        bucket.last_replay_us = (time.perf_counter() - t0) * 1e6
        bucket.replay_count  += 1
        self._total_replay   += 1
        return bucket.output_buf

    def stats(self) -> dict:
        """Return per-bucket and aggregate statistics."""
        total_ops = self._total_replay + self._total_fallback
        buckets = {}
        for bkt, b in self._buckets.items():
            total_bkt = b.replay_count + b.fallback_count
            buckets[bkt] = {
                "replay_count":   b.replay_count,
                "fallback_count": b.fallback_count,
                "utilization_pct": (b.replay_count / total_bkt * 100) if total_bkt else 0.0,
                "last_replay_us": b.last_replay_us,
            }
        return {
            "total_replays":    self._total_replay,
            "total_fallbacks":  self._total_fallback,
            "total_captures":   self._total_capture,
            "graph_hit_rate":   (self._total_replay / total_ops * 100) if total_ops else 0.0,
            "buckets":          buckets,
            "bucket_sizes":     self.bucket_sizes,
            "captured_buckets": list(self._buckets.keys()),
            "fallback_log":     list(self._fallback_log),
        }

    def record_fallback(self, B: int, reason: str) -> None:
        """Called by engine when it falls back despite a bucket being available."""
        bkt = self.select_bucket(B)
        if bkt and bkt in self._buckets:
            self._buckets[bkt].fallback_count += 1
        self._total_fallback += 1
        self._log_fallback(reason)

    # ── internals ─────────────────────────────────────────────────────────────

    def _lazy_capture(self, bucket_size: int, capture_fn: Callable) -> bool:
        """
        Warmup + capture for a single bucket size.

        Returns True on success, False on any failure (logs the reason).
        """
        try:
            log.info("[graph] warming up bucket B=%d (%d steps)...",
                     bucket_size, self.warmup_steps)

            # ── warmup ────────────────────────────────────────────────────────
            with torch.inference_mode():
                for _ in range(self.warmup_steps):
                    capture_fn(bucket_size)
            torch.cuda.synchronize()

            # ── capture ───────────────────────────────────────────────────────
            log.info("[graph] capturing bucket B=%d...", bucket_size)
            g = torch.cuda.CUDAGraph()
            with torch.inference_mode(), torch.cuda.graph(g):
                static_output = capture_fn(bucket_size)

            # static_output is at a stable GPU address for the lifetime of g
            bucket = GraphBucket(
                bucket_size=bucket_size,
                graph=g,
                output_buf=static_output,
            )
            self._buckets[bucket_size] = bucket
            self._total_capture += 1
            print(f"[graph] captured B={bucket_size}  "
                  f"output_buf={tuple(static_output.shape)}")
            return True

        except Exception as exc:
            reason = f"capture failed B={bucket_size}: {exc}"
            log.warning("[graph] %s", reason)
            self._log_fallback(reason)
            return False

    def _log_fallback(self, reason: str) -> None:
        self._fallback_log.append(reason)
        log.debug("[graph] fallback: %s", reason)
