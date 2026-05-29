"""Decode-regime attention kernel — Triton skeleton.

Phase 3 task. Decode is M = 1 (one new query token per sequence) attending
to a KV cache of length T. The arithmetic intensity is O(1) flops per byte,
so the kernel is HBM-bandwidth-bound and the optimization story is entirely
about reading the KV cache as few times as possible and as coalesced as
possible.

Plan:
    Inputs:
        q:       (B, H, D)         single new query per (batch, head)
        k_cache: (B, H_kv, T, D)
        v_cache: (B, H_kv, T, D)
    Parallelization:
        - Outer grid: (B, H). One program instance per (batch, head).
        - Inner: split T into N_KV_BLOCKS tiles of length BLOCK_T. Each
          program iterates over its assigned tiles and accumulates the
          online-softmax (m, l, acc) state.
        - Optionally split the T axis across program instances ("split-k"
          for decode), with a second reduction kernel to merge per-block
          (m, l, acc) states. This is the FlashDecoding/FlashAttention-2
          decode trick. The merge is the same online-softmax reduction we
          already use intra-block, so it's natural to extend.
    Considerations:
        - For very long contexts (T >> 4k), split-k is a 2-5x speedup over
          a single-program decode. Defer this to phase 3.
        - GQA-aware: when `num_kv_heads < H`, share the KV reads across the
          group (broadcast q across the group's heads in registers).
"""

from __future__ import annotations

from typing import Optional

import torch


def triton_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    num_kv_heads: Optional[int] = None,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """Single-query attention against a KV cache. Skeleton — see module docstring.

    Args:
        q:       `(B, H, D)` or `(B, H, 1, D)`.
        k_cache: `(B, H_kv, T, D)`.
        v_cache: `(B, H_kv, T, D)`.
        num_kv_heads: `H_kv`. Defaults to `H` (no GQA).
        sm_scale: Softmax scale; defaults to `1/sqrt(D)`.

    Returns:
        `(B, H, 1, D)`.
    """
    raise NotImplementedError("Phase 3 — see DESIGN_NOTES.md")
