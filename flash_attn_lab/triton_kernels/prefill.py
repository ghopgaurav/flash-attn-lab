"""Causal fused attention forward in Triton (FlashAttention-style).

Tiling strategy:
    For each (batch, head, query-block) tile of size BLOCK_M, we hold Q in
    on-chip SRAM and stream K, V in BLOCK_N-sized tiles along the sequence
    axis. Inside the inner loop we compute the BLOCK_M-by-BLOCK_N score
    tile, fuse the (max, sum-of-exp, weighted-V) accumulators into running
    state, and never materialize the full M-by-N attention matrix in HBM.
    For causal=True, we early-exit the inner loop once the leftmost column
    of the next K tile is past the rightmost row of the current Q tile.

Online softmax recurrence (numerically stable, associative across KV tiles):
    For each new score block S of shape (BLOCK_M, BLOCK_N) at scale sm_scale:
        m_new = max(m_prev, rowmax(S))
        alpha = exp(m_prev - m_new)
        p     = exp(S - m_new[:, None])
        l_new = alpha * l_prev + rowsum(p)
        acc   = alpha[:, None] * acc + p @ V_block
    After the loop, divide acc by l (broadcast) to get the final softmax-
    weighted V. This is the same recurrence used in FlashAttention v1/v2;
    the associativity of the (m, l, acc) merge is what makes the kernel
    correct under arbitrary BLOCK_N tiling.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - triton may be missing on CPU-only boxes
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_AVAILABLE = False


def _autotune_configs():
    if not _TRITON_AVAILABLE:
        return []
    # A small but useful sweep. Real production code would have many more
    # configs; we keep the search space tight so first-call autotune doesn't
    # dominate notebook time.
    configs = []
    for bm in (64, 128):
        for bn in (32, 64, 128):
            for nw in (4, 8):
                for ns in (2, 3):
                    configs.append(
                        triton.Config(
                            {"BLOCK_M": bm, "BLOCK_N": bn},
                            num_warps=nw,
                            num_stages=ns,
                        )
                    )
    return configs


if _TRITON_AVAILABLE:

    @triton.autotune(configs=_autotune_configs(), key=["N_CTX", "HEAD_DIM"])
    @triton.jit
    def _attn_fwd_kernel(
        Q,
        K,
        V,
        Out,
        sm_scale,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_om, stride_od,
        Z, H, N_CTX,
        HEAD_DIM: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        start_m = tl.program_id(0)
        off_zh = tl.program_id(1)
        off_z = off_zh // H
        off_h = off_zh % H

        q_base = Q + off_z * stride_qb + off_h * stride_qh
        k_base = K + off_z * stride_kb + off_h * stride_kh
        v_base = V + off_z * stride_vb + off_h * stride_vh
        o_base = Out + off_z * stride_ob + off_h * stride_oh

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)

        # Load Q tile once.
        q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

        # Running softmax state.
        m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        # Effective sm_scale, applied to QK^T pre-softmax.
        qk_scale = sm_scale * 1.44269504089  # log2(e), so we can use exp2

        # Determine inner-loop range.
        if IS_CAUSAL:
            # Only attend to keys at positions <= the rightmost query in this tile.
            hi = tl.minimum(N_CTX, (start_m + 1) * BLOCK_M)
        else:
            hi = N_CTX

        for start_n in range(0, hi, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)

            k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
            v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd

            k_mask = offs_n[:, None] < N_CTX
            k = tl.load(k_ptrs, mask=k_mask, other=0.0)
            v = tl.load(v_ptrs, mask=k_mask, other=0.0)

            qk = tl.dot(q, tl.trans(k))  # (BLOCK_M, BLOCK_N)
            qk = qk * qk_scale

            if IS_CAUSAL:
                causal_mask = offs_m[:, None] >= offs_n[None, :]
                qk = tl.where(causal_mask, qk, -float("inf"))
            # Mask out-of-range KV positions (bottom of the matrix).
            kv_mask = offs_n[None, :] < N_CTX
            qk = tl.where(kv_mask, qk, -float("inf"))

            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            alpha = tl.exp2(m_i - m_ij)
            p = tl.exp2(qk - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)

            # Update accumulator.
            acc = acc * alpha[:, None]
            # p is fp32; v may be fp16/bf16. tl.dot accepts mixed via cast.
            acc = tl.dot(p.to(v.dtype), v, acc=acc)

            l_i = l_i * alpha + l_ij
            m_i = m_ij

        # Final normalization.
        acc = acc / l_i[:, None]

        o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
        tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N_CTX)


def triton_attention_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """Causal fused attention forward in Triton.

    Args:
        q: `(B, H, M, D)`, fp16 or bf16.
        k: `(B, H, N, D)`, same dtype as `q`.
        v: `(B, H, N, D)`, same dtype as `q`.
        causal: Apply lower-triangular mask if True.
        sm_scale: Softmax scale; defaults to `1/sqrt(D)`.

    Returns:
        `(B, H, M, D)` in `q.dtype`.

    Currently supports M == N (square attention). HEAD_DIM must be a power
    of two in {16, 32, 64, 128, 256}.
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError(
            "Triton is not available. Install PyTorch 2.5+ with CUDA to get Triton."
        )
    if not q.is_cuda:
        raise RuntimeError("triton_attention_prefill requires CUDA tensors")
    if q.shape != k.shape or k.shape != v.shape:
        raise ValueError(
            "this scaffold supports M == N square attention; "
            f"got q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}"
        )
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"q.dtype must be fp16 or bf16, got {q.dtype}")
    if q.dtype != k.dtype or k.dtype != v.dtype:
        raise ValueError("q, k, v must share dtype")

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    B, H, N_CTX, D = q.shape
    if D not in (16, 32, 64, 128, 256):
        raise ValueError(f"HEAD_DIM must be in {{16,32,64,128,256}}, got {D}")

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    out = torch.empty_like(q)

    grid = lambda meta: (triton.cdiv(N_CTX, meta["BLOCK_M"]), B * H)
    _attn_fwd_kernel[grid](
        q, k, v, out,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, N_CTX,
        HEAD_DIM=D,
        IS_CAUSAL=causal,
    )
    return out
