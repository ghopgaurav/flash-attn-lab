"""Correctness tests for the warp-shuffle row-sum reduction."""

from __future__ import annotations

import pytest
import torch

from flash_attn_lab.utils.reference import reference_block_sum


@pytest.mark.cuda
@pytest.mark.parametrize("B, N", [(1, 32), (4, 257), (16, 4096), (3, 1)])
def test_row_sum_matches_reference(cuda_device, cuda_module, B, N):
    g = torch.Generator(device=cuda_device).manual_seed(0)
    x = torch.randn((B, N), generator=g, device=cuda_device, dtype=torch.float32)

    y = cuda_module.row_sum(x)
    ref = reference_block_sum(x)
    torch.testing.assert_close(y, ref, atol=1e-3, rtol=1e-3)
