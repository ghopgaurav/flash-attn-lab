"""Correctness tests for the online softmax kernel."""

from __future__ import annotations

import pytest
import torch

from flash_attn_lab.utils.reference import reference_softmax_rowwise


@pytest.mark.cuda
@pytest.mark.parametrize("B, N", [(1, 64), (8, 1024), (3, 257)])
def test_softmax_online_matches_reference(cuda_device, cuda_module, B, N):
    g = torch.Generator(device=cuda_device).manual_seed(0)
    x = torch.randn((B, N), generator=g, device=cuda_device, dtype=torch.float32) * 4.0

    y = cuda_module.softmax_online(x)
    ref = reference_softmax_rowwise(x)
    torch.testing.assert_close(y, ref, atol=1e-5, rtol=1e-4)


@pytest.mark.cuda
def test_softmax_online_handles_large_logits(cuda_device, cuda_module):
    """Naive exp(x) on x=200 overflows fp32; the online version must not."""
    x = torch.full((2, 16), 200.0, device=cuda_device, dtype=torch.float32)
    y = cuda_module.softmax_online(x)
    expected = torch.full_like(x, 1.0 / 16)
    torch.testing.assert_close(y, expected, atol=1e-5, rtol=1e-5)
