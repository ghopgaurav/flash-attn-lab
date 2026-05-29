"""Correctness tests for the tiled CUDA GEMM kernel."""

from __future__ import annotations

import pytest
import torch


@pytest.mark.cuda
@pytest.mark.parametrize("M, K, N", [(64, 64, 64), (96, 33, 130), (256, 128, 96)])
def test_matmul_tiled_matches_torch(cuda_device, cuda_module, M, K, N):
    g = torch.Generator(device=cuda_device).manual_seed(0)
    A = torch.randn((M, K), generator=g, device=cuda_device, dtype=torch.float32)
    B = torch.randn((K, N), generator=g, device=cuda_device, dtype=torch.float32)

    C = cuda_module.matmul_tiled(A, B)
    ref = A @ B
    torch.testing.assert_close(C, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.cuda
def test_matmul_tiled_rejects_bad_shape(cuda_device, cuda_module):
    A = torch.randn(8, 4, device=cuda_device, dtype=torch.float32)
    B = torch.randn(8, 4, device=cuda_device, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        cuda_module.matmul_tiled(A, B)
