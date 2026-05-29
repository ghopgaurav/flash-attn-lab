"""flash-attn-lab: from-scratch fused attention in Triton and CUDA.

A multi-architecture study of fused attention written from scratch in Triton
and raw CUDA C++, covering both prefill and decode regimes, with rigorous
benchmarking and Nsight Compute profiling.

Public API:
    from flash_attn_lab.ops.attention import attention
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
