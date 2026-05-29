"""Pure PyTorch reference implementations.

These are the ground-truth oracles that every fused kernel is checked against
in the test suite. They are deliberately written in the most boring,
mathematically transparent way possible: `softmax(QK^T / sqrt(d)) V`. No
tricks, no fusion, no online softmax. If a fused kernel disagrees with these
beyond the configured tolerance, the fused kernel is wrong.
"""

from __future__ import annotations

import math
from typing import Optional

import torch


def reference_attention_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """Reference scaled-dot-product attention for the prefill regime.

    Args:
        q: Query, shape `(B, H, M, D)`.
        k: Key,   shape `(B, H, N, D)`.
        v: Value, shape `(B, H, N, D)`.
        causal: If True, mask `j > i`.
        sm_scale: Softmax scale; defaults to `1/sqrt(D)`.

    Returns:
        `(B, H, M, D)` tensor in the same dtype as `q`.

    Math is performed in fp32 internally for numerical stability; the result
    is cast back to `q.dtype`. This makes the reference suitable as a target
    for both fp16/bf16 fused kernels.
    """
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q, k, v must all be 4D: (B, H, S, D)")
    if q.shape[-1] != k.shape[-1] or k.shape[-1] != v.shape[-1]:
        raise ValueError("head dim mismatch")
    if k.shape[-2] != v.shape[-2]:
        raise ValueError("kv seq mismatch")

    orig_dtype = q.dtype
    d = q.shape[-1]
    scale = sm_scale if sm_scale is not None else 1.0 / math.sqrt(d)

    qf = q.float()
    kf = k.float()
    vf = v.float()

    # (B, H, M, N)
    scores = torch.matmul(qf, kf.transpose(-2, -1)) * scale

    if causal:
        m, n = scores.shape[-2], scores.shape[-1]
        # Bottom-right alignment when M != N is unusual; we use top-left
        # alignment which is the convention used by SDPA when M == N.
        mask = torch.ones(m, n, dtype=torch.bool, device=scores.device).triu(1)
        scores = scores.masked_fill(mask, float("-inf"))

    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, vf)
    return out.to(orig_dtype)


def reference_attention_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """Reference single-query attention against a KV cache.

    Args:
        q:       Single-query tensor, shape `(B, H, 1, D)` or `(B, H, D)`.
        k_cache: Cached keys,   shape `(B, H, T, D)`.
        v_cache: Cached values, shape `(B, H, T, D)`.
        sm_scale: Softmax scale; defaults to `1/sqrt(D)`.

    Returns:
        `(B, H, 1, D)` tensor in the same dtype as `q`. The decode output
        always carries the M=1 axis so callers can chain prefill + decode
        results without reshape gymnastics.
    """
    if q.dim() == 3:
        q = q.unsqueeze(-2)
    if q.shape[-2] != 1:
        raise ValueError(f"decode expects M=1, got M={q.shape[-2]}")

    return reference_attention_prefill(
        q,
        k_cache,
        v_cache,
        causal=False,  # decode attends to all of [0, T); causality is implicit.
        sm_scale=sm_scale,
    )


def reference_softmax_rowwise(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable row-wise softmax in fp32, cast back to input dtype."""
    orig = x.dtype
    xf = x.float()
    m = xf.max(dim=-1, keepdim=True).values
    e = (xf - m).exp()
    s = e.sum(dim=-1, keepdim=True)
    return (e / s).to(orig)


def reference_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Reference fp32 matmul. Cast inputs to fp32, multiply, cast back to a.dtype."""
    return torch.matmul(a.float(), b.float()).to(a.dtype)


def reference_block_sum(x: torch.Tensor) -> torch.Tensor:
    """Reference block-sum: sum each row of `x` (last dim) into a single scalar.

    Returned shape is `x.shape[:-1]`. Computed in fp32.
    """
    return x.float().sum(dim=-1).to(x.dtype)
