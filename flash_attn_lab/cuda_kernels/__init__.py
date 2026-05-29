"""Raw CUDA C++ kernels and the JIT loader.

Use `get_cuda_module()` from `build.py` to lazily compile the .cu sources
and obtain a Python module exposing the kernels via pybind11.
"""

from __future__ import annotations

from flash_attn_lab.cuda_kernels.build import get_cuda_module

__all__ = ["get_cuda_module"]
