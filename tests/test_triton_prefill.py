"""Correctness tests for the Triton prefill kernel."""

from __future__ import annotations

import pytest
import torch

from flash_attn_lab.utils.reference import reference_attention_prefill


@pytest.mark.triton
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("seqlen", [128, 512])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_triton_prefill_matches_reference(
    cuda_device, triton_available, dtype, causal, seqlen, head_dim
):
    from flash_attn_lab.triton_kernels.prefill import triton_attention_prefill

    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bf16 not supported on this device")

    B, H = 2, 4
    g = torch.Generator(device=cuda_device).manual_seed(123)
    q = torch.randn((B, H, seqlen, head_dim), generator=g, device=cuda_device, dtype=dtype) * 0.5
    k = torch.randn((B, H, seqlen, head_dim), generator=g, device=cuda_device, dtype=dtype) * 0.5
    v = torch.randn((B, H, seqlen, head_dim), generator=g, device=cuda_device, dtype=dtype) * 0.5

    out = triton_attention_prefill(q, k, v, causal=causal)
    ref = reference_attention_prefill(q, k, v, causal=causal)

    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.triton
def test_triton_prefill_rejects_mismatched_shapes(cuda_device, triton_available):
    from flash_attn_lab.triton_kernels.prefill import triton_attention_prefill

    q = torch.randn(1, 1, 64, 64, device=cuda_device, dtype=torch.float16)
    k = torch.randn(1, 1, 32, 64, device=cuda_device, dtype=torch.float16)
    v = torch.randn(1, 1, 32, 64, device=cuda_device, dtype=torch.float16)
    with pytest.raises(ValueError):
        triton_attention_prefill(q, k, v)
