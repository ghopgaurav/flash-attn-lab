"""pytest fixtures for flash-attn-lab.

Centralizes the CUDA / Triton / SM89 capability checks so individual test
modules can `pytest.importorskip("torch")` and then ask for a fixture.
"""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(scope="session")
def cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available; CUDA-only test skipped")
    return torch.device("cuda")


@pytest.fixture(scope="session")
def cuda_module():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available; CUDA-only test skipped")
    from flash_attn_lab.cuda_kernels.build import get_cuda_module

    mod = get_cuda_module()
    if mod is None:
        pytest.skip("CUDA module unavailable (build failed or no CUDA toolchain)")
    return mod


@pytest.fixture(scope="session")
def triton_available():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available; Triton tests require CUDA")
    try:
        import triton  # noqa: F401
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"triton not importable: {exc}")
    return True
