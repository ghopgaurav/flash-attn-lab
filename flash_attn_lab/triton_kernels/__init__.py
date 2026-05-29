"""Triton kernel implementations.

Working:
    prefill.triton_attention_prefill

Skeletons (raise NotImplementedError):
    decode.triton_decode_attention
    gqa.triton_gqa_attention
    fp8.triton_fp8_attention
"""

from __future__ import annotations

from flash_attn_lab.triton_kernels.prefill import triton_attention_prefill

__all__ = ["triton_attention_prefill"]
