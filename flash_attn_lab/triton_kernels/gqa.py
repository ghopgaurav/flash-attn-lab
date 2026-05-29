"""Grouped-Query Attention (GQA) prefill kernel — Triton skeleton.

Phase 3 task. The GQA pattern shares K and V across `num_q_heads / num_kv_heads`
query heads, reducing the KV-cache footprint at the cost of a small accuracy
hit. Modern open-weight LLMs (Llama 3.x, Qwen 2/3, Mistral) all use GQA.

Plan:
    Inputs:
        q: (B, H_q, M, D)
        k: (B, H_kv, N, D)
        v: (B, H_kv, N, D)
        num_kv_heads: H_kv. Must divide H_q evenly.
    Mapping:
        Each query head h maps to kv head h // (H_q / H_kv). Inside the
        Triton kernel we replace the existing `off_h` derivation with a
        `kv_off_h = off_h // group_size` so the kv pointers index the
        smaller H_kv axis. The query axis is otherwise unchanged.
    Considerations:
        - The grid over (B, H_q) is unchanged, so SM occupancy is the same
          as MHA prefill.
        - KV reads are reused `group_size` times per Q head; on architectures
          with persistent kernel scheduling we'd want to prefer to launch
          all Q-heads sharing a KV-head on the same SM to maximize L2 reuse,
          but the simple per-(B, H_q) launch is the correct first cut.

See DESIGN_NOTES.md for the chosen autotune space and dtype matrix.
"""

from __future__ import annotations

import torch


def triton_gqa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_kv_heads: int,
    causal: bool = True,
) -> torch.Tensor:
    """GQA prefill attention. Skeleton — see module docstring.

    Args:
        q: `(B, H_q, M, D)`.
        k: `(B, H_kv, N, D)` where `H_kv == num_kv_heads`.
        v: `(B, H_kv, N, D)`.
        num_kv_heads: `H_kv`. Must divide `H_q` evenly.
        causal: Lower-triangular mask if True.

    Returns:
        `(B, H_q, M, D)`.
    """
    raise NotImplementedError("Phase 3 — see DESIGN_NOTES.md")
