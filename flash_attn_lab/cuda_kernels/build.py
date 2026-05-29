"""JIT compile + load wrapper for the raw CUDA kernels.

`get_cuda_module()` returns a Python module exposing the kernels as defined
in `bindings.cpp`. The compile step is cached by torch's extension build
directory, so subsequent calls in the same Python process (and across
processes) are essentially free.

If CUDA is unavailable, returns `None` and logs a single warning. Importing
this module on a CPU-only box must not raise.

Environment knobs:
    FLASH_ATTN_LAB_FORCE_REBUILD=1  Force a fresh recompile (sets verbose=True too).
    FLASH_ATTN_LAB_BUILD_VERBOSE=1  Verbose compile output.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_MODULE_NAME = "flash_attn_lab_cuda"
_THIS_DIR = Path(__file__).resolve().parent

_CUDA_SOURCES = [
    "matmul_tiled.cu",
    "reduce.cu",
    "softmax_online.cu",
    "decode_attention.cu",
]
_CPP_SOURCES = ["bindings.cpp"]

_module_cache: Optional[object] = None
_load_lock = threading.Lock()


def _resolve_sources() -> list[str]:
    paths = []
    for fname in _CPP_SOURCES + _CUDA_SOURCES:
        p = _THIS_DIR / fname
        if not p.exists():  # pragma: no cover - defensive
            raise FileNotFoundError(f"missing CUDA source: {p}")
        paths.append(str(p))
    return paths


def get_cuda_module() -> Optional[object]:
    """Lazily JIT-compile and return the raw CUDA kernel module.

    Returns:
        The loaded module, or `None` if CUDA is unavailable or the build
        toolchain is missing. Never raises in the no-CUDA path; raises
        whatever `torch.utils.cpp_extension.load` raises if the build itself
        fails on a CUDA-capable machine.
    """
    global _module_cache
    if _module_cache is not None:
        return _module_cache

    if not torch.cuda.is_available():
        logger.warning(
            "CUDA is not available; raw CUDA kernels will not be loaded. "
            "Triton kernels and Python reference implementations remain usable."
        )
        return None

    with _load_lock:
        if _module_cache is not None:
            return _module_cache

        from torch.utils.cpp_extension import load  # local import; heavy

        verbose = bool(int(os.environ.get("FLASH_ATTN_LAB_BUILD_VERBOSE", "0")))
        force = bool(int(os.environ.get("FLASH_ATTN_LAB_FORCE_REBUILD", "0")))
        if force:
            verbose = True

        sources = _resolve_sources()
        logger.info("compiling %s with %d sources", _MODULE_NAME, len(sources))

        extra_cflags = ["-O3", "-std=c++17"]
        extra_cuda_cflags = ["-O3", "--use_fast_math", "-std=c++17"]

        try:
            mod = load(
                name=_MODULE_NAME,
                sources=sources,
                extra_cflags=extra_cflags,
                extra_cuda_cflags=extra_cuda_cflags,
                verbose=verbose,
            )
        except Exception:
            logger.exception("failed to JIT compile %s", _MODULE_NAME)
            raise

        _module_cache = mod
        return mod
