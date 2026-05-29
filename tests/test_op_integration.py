"""Integration tests for the public `flash_attn_lab.attention` op.

These tests are deliberately split: some run on CPU-only environments
(import the module, exercise the fake/meta path) and some require a CUDA
GPU + Triton.
"""

from __future__ import annotations

import pytest
import torch

from flash_attn_lab.ops.attention import attention
from flash_attn_lab.utils.device import get_device_info, select_dtype


def test_attention_module_imports_on_cpu():
    """Importing the op surface must work on CPU-only machines."""
    assert callable(attention)


def test_attention_fake_implementation_shapes():
    """The fake/meta impl must return a tensor with q's shape and dtype."""
    q = torch.empty((2, 4, 32, 64), device="meta", dtype=torch.float16)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    out = attention(q, k, v, causal=True)
    assert out.shape == q.shape
    assert out.dtype == q.dtype
    assert out.device.type == "meta"


def test_device_info_safe_on_cpu():
    info = get_device_info()
    assert isinstance(info.name, str)
    if not torch.cuda.is_available():
        assert info.is_cuda is False
        assert info.sm == 0


def test_select_dtype_falls_back_safely():
    info = get_device_info()
    dt = select_dtype("bf16", info)
    assert isinstance(dt, torch.dtype)
    if not info.is_cuda:
        assert dt == torch.float32


@pytest.mark.triton
def test_attention_dispatches_to_triton_prefill(cuda_device, triton_available):
    from flash_attn_lab.utils.reference import reference_attention_prefill

    dtype = select_dtype("bf16")
    if dtype == torch.float32:
        pytest.skip("device does not support fp16/bf16")

    q = torch.randn(2, 4, 128, 64, device=cuda_device, dtype=dtype) * 0.5
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    out = attention(q, k, v, causal=True)
    ref = reference_attention_prefill(q, k, v, causal=True)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


def test_attention_backward_is_stub():
    """The autograd path is registered but raises a clear error today."""
    if not torch.cuda.is_available():
        pytest.skip("backward stub test needs CUDA + Triton path")
    try:
        import triton  # noqa: F401
    except Exception:
        pytest.skip("triton not available")

    dtype = select_dtype("bf16")
    if dtype == torch.float32:
        pytest.skip("device does not support fp16/bf16")
    q = torch.randn(1, 1, 64, 64, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    out = attention(q, k, v, causal=True)
    with pytest.raises(NotImplementedError):
        out.sum().backward()
