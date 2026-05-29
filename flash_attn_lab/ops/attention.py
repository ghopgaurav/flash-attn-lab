"""Public `attention` op, registered via `torch.library.custom_op`.

Why `torch.library.custom_op` and not `torch.autograd.Function`:
    - `custom_op` is the modern recommended path (PyTorch 2.4+). It plays
      nicely with `torch.compile`: the fake-tensor implementation lets
      Inductor reason about shapes/dtypes without ever launching the kernel.
    - It cleanly separates the dispatcher contract (signature, layout,
      device) from the implementation; the autograd story is registered
      separately if/when we have a backward.

Public surface:
    attention(q, k, v, causal=True, sm_scale=None) -> Tensor

Behavior:
    - For now, dispatches unconditionally to the Triton prefill kernel.
    - Decode (M=1) and GQA dispatch are extension points; the function
      branches on shape/regime to pick a kernel.
    - Backward is a stub: registering as `register_autograd` with a
      function that raises `NotImplementedError` makes it explicit instead
      of silently zero-ing grads.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from flash_attn_lab.triton_kernels.prefill import triton_attention_prefill

# Library namespace. Picked deliberately to avoid colliding with PyTorch's own
# "aten" or "prims" libraries.
_LIB_NS = "flash_attn_lab"
_OP_NAME = "attention"
_QUALNAME = f"{_LIB_NS}::{_OP_NAME}"


def _select_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    sm_scale: Optional[float],
) -> torch.Tensor:
    """Pick the right kernel for the (regime, dtype, device) tuple.

    Today: prefill -> Triton. Decode (M=1) and GQA (H_q != H_kv) raise
    NotImplementedError; the harness can branch on shape and call a
    specific implementation directly until those kernels land.
    """
    H_q = q.shape[1]
    H_kv = k.shape[1]
    M = q.shape[2]

    if H_q != H_kv:
        raise NotImplementedError(
            "GQA dispatch (H_q != H_kv) is a Phase 3 task; "
            "use triton_kernels.gqa.triton_gqa_attention when implemented."
        )
    if M == 1 and k.shape[2] > 1:
        raise NotImplementedError(
            "Decode (M=1) dispatch is a Phase 3 task; "
            "use triton_kernels.decode.triton_decode_attention when implemented."
        )
    return triton_attention_prefill(q, k, v, causal=causal, sm_scale=sm_scale)


# `mutates_args=()` declares the op is purely functional; required by the
# new custom_op API.
@torch.library.custom_op(
    _QUALNAME,
    mutates_args=(),
)
def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """Public fused attention op.

    Args:
        q: `(B, H, M, D)`.
        k: `(B, H_kv, N, D)`. For MHA, `H_kv == H`.
        v: `(B, H_kv, N, D)`.
        causal: Lower-triangular mask if True.
        sm_scale: Softmax scale; defaults to `1/sqrt(D)`.

    Returns:
        `(B, H, M, D)` in `q.dtype`.
    """
    return _select_kernel(q, k, v, causal, sm_scale)


@attention.register_fake
def _attention_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """Shape/dtype-only implementation for `torch.compile` and meta tensors."""
    return torch.empty_like(q)


def _attention_setup_context(ctx, inputs, output):  # pragma: no cover - stub
    q, k, v, causal, sm_scale = inputs
    ctx.save_for_backward(q, k, v, output)
    ctx.causal = causal
    ctx.sm_scale = sm_scale if sm_scale is not None else 1.0 / math.sqrt(q.shape[-1])


def _attention_backward(ctx, grad_output):  # pragma: no cover - stub
    # TODO(phase4): implement FlashAttention v2 backward (dQ, dK, dV).
    # The forward kernel keeps (m, l) but does not currently save them; the
    # backward needs the saved softmax LSE row stats to recompute exact P.
    raise NotImplementedError(
        "flash_attn_lab.attention backward is not implemented yet "
        "(Phase 4). Use torch.nn.functional.scaled_dot_product_attention "
        "if you need a backward pass today."
    )


# Register the autograd stub so callers get a clear error instead of a
# silent miscompile when someone tries `.backward()` on a result.
attention.register_autograd(
    _attention_backward,
    setup_context=_attention_setup_context,
)


__all__ = ["attention"]
