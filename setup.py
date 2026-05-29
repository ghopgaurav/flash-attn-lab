"""AOT build script for the raw CUDA kernels.

Defaults: editable install via pyproject.toml + JIT compilation (no extension).

To force AOT compilation of the CUDA module, set BUILD_CUDA_EXT=1:

    BUILD_CUDA_EXT=1 pip install -e .

This is the right path for environments that should not pay the JIT cost on
first kernel call (e.g. CI on a self-hosted GPU runner).

If torch is not importable or no nvcc is on PATH, the extension is silently
skipped so `pip install -e .` still works on CPU-only machines.
"""

from __future__ import annotations

import os
from pathlib import Path

from setuptools import setup

HERE = Path(__file__).resolve().parent
CU_DIR = HERE / "flash_attn_lab" / "cuda_kernels"


def _maybe_cuda_extension():
    if os.environ.get("BUILD_CUDA_EXT", "0") not in ("1", "true", "yes"):
        return []
    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    except Exception:
        return []

    sources = [
        str(CU_DIR / "bindings.cpp"),
        str(CU_DIR / "matmul_tiled.cu"),
        str(CU_DIR / "reduce.cu"),
        str(CU_DIR / "softmax_online.cu"),
        str(CU_DIR / "decode_attention.cu"),
    ]
    ext = CUDAExtension(
        name="flash_attn_lab._cuda_ext",
        sources=sources,
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": ["-O3", "--use_fast_math", "-std=c++17"],
        },
    )
    return [ext], BuildExtension


_ext_info = _maybe_cuda_extension()
if _ext_info:
    ext_modules, build_ext = _ext_info
    setup(
        ext_modules=ext_modules,
        cmdclass={"build_ext": build_ext},
    )
else:
    setup()
