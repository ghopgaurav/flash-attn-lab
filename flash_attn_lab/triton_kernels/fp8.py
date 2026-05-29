"""FP8 attention kernel — Triton skeleton (SM89+).

Phase 5 task. FP8 attention requires (a) tensor-core paths that accept the
fp8 dtypes and (b) per-tensor (or per-head) scaling factors so the limited
dynamic range of fp8 doesn't catastrophically clip activations.

Dtype choice:
    - e4m3 (4-bit exp, 3-bit mantissa) has higher precision but a tighter
      dynamic range; preferred for Q, K, V and attention probabilities.
    - e5m2 (5-bit exp, 2-bit mantissa) has wider range, lower precision;
      useful for gradients (not relevant here, forward only).
    For prefill attention, e4m3 for Q, K, V and an e4m3 (or bf16)
    accumulator-side cast for the softmax output is the typical recipe.

Scaling strategy:
    - Per-tensor scales (q_scale, k_scale, v_scale) are the simplest. The
      kernel multiplies QK^T by q_scale * k_scale to get back to fp32-ish
      logits for the softmax, then re-scales the V matmul by v_scale.
    - Per-head scales reduce clipping for outlier heads (Llama-style
      heads-with-attention-sinks); revisit in DESIGN_NOTES.md.

SM89+ guard:
    The fp8 dtypes (`torch.float8_e4m3fn`, `torch.float8_e5m2`) and the
    PTX mma.fp8 instructions are only supported on SM89 (Ada) and newer.
    On SM80 (Ampere) we fall back to bf16 (or raise — caller's choice).
"""

from __future__ import annotations

from typing import Optional

import torch

from flash_attn_lab.utils.device import get_device_info


def _ensure_sm89():
    info = get_device_info()
    if not info.is_cuda:
        raise RuntimeError("triton_fp8_attention requires a CUDA GPU")
    if not info.supports_fp8:
        raise RuntimeError(
            f"triton_fp8_attention requires SM89+ (Ada) or newer; "
            f"current device is {info.name} ({info.sm_str})"
        )


def triton_fp8_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    causal: bool = True,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """FP8 prefill attention. Skeleton — see module docstring.

    Args:
        q:       `(B, H, M, D)`, dtype `torch.float8_e4m3fn`.
        k:       `(B, H, N, D)`, dtype `torch.float8_e4m3fn`.
        v:       `(B, H, N, D)`, dtype `torch.float8_e4m3fn`.
        q_scale: scalar or per-head scale tensor.
        k_scale: scalar or per-head scale tensor.
        v_scale: scalar or per-head scale tensor.
        causal: Lower-triangular mask if True.
        sm_scale: Softmax scale; defaults to `1/sqrt(D)`.

    Returns:
        `(B, H, M, D)` in bf16 (output is dequantized).
    """
    _ensure_sm89()
    raise NotImplementedError("Phase 5")
