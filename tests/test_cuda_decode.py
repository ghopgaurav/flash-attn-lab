"""Correctness tests for the naive fused decode-attention CUDA kernel."""

from __future__ import annotations

import math

import pytest
import torch

from flash_attn_lab.utils.reference import reference_attention_decode


@pytest.mark.cuda
@pytest.mark.parametrize("B, H, T, D", [(1, 1, 16, 32), (2, 4, 257, 64), (3, 8, 1024, 128)])
def test_decode_attention_matches_reference(cuda_device, cuda_module, B, H, T, D):
    g = torch.Generator(device=cuda_device).manual_seed(0)
    q = torch.randn((B, H, D), generator=g, device=cuda_device, dtype=torch.float32) * 0.5
    k = torch.randn((B, H, T, D), generator=g, device=cuda_device, dtype=torch.float32) * 0.5
    v = torch.randn((B, H, T, D), generator=g, device=cuda_device, dtype=torch.float32) * 0.5

    sm_scale = 1.0 / math.sqrt(D)
    out = cuda_module.decode_attention(q, k, v, sm_scale)

    ref = reference_attention_decode(q.unsqueeze(-2), k, v, sm_scale=sm_scale).squeeze(-2)
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)
