"""Single-invocation kernel runner — NCU profiling target.

Usage (run directly for a smoke check):
    python bench/profile_one.py

Usage (inside NCU):
    sudo ncu \\
        --set full \\
        --target-processes all \\
        --kernel-name regex:_attn_fwd_kernel \\
        --csv \\
        -o profiles/triton_prefill_a100_s4096_d128 \\
        python bench/profile_one.py

NCU targets both kernels by default (triton_prefill + torch_sdpa).
Pass --kernel-name to restrict to just one.

Config knobs via env vars (override for different shapes):
    PROF_B=1 PROF_H=16 PROF_S=4096 PROF_D=128 PROF_DTYPE=bf16
    PROF_KERNEL=triton_prefill   # or torch_sdpa
"""

from __future__ import annotations

import math
import os
import sys

import torch

_DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}

B = int(os.environ.get("PROF_B", 1))
H = int(os.environ.get("PROF_H", 16))
S = int(os.environ.get("PROF_S", 4096))
D = int(os.environ.get("PROF_D", 128))
DTYPE_NAME = os.environ.get("PROF_DTYPE", "bf16")
KERNEL = os.environ.get("PROF_KERNEL", "triton_prefill")
WARMUP = int(os.environ.get("PROF_WARMUP", 5))

if not torch.cuda.is_available():
    print("ERROR: no CUDA device; NCU profiling requires a GPU.", file=sys.stderr)
    sys.exit(1)

dtype = _DTYPE_MAP[DTYPE_NAME]
device = torch.device("cuda")

g = torch.Generator(device=device).manual_seed(0)
q = torch.randn((B, H, S, D), generator=g, device=device, dtype=dtype)
k = torch.randn((B, H, S, D), generator=g, device=device, dtype=dtype)
v = torch.randn((B, H, S, D), generator=g, device=device, dtype=dtype)

# ── kernel dispatch ────────────────────────────────────────────────────────────

def _run_triton():
    from flash_attn_lab.triton_kernels.prefill import triton_attention_prefill
    return triton_attention_prefill(q, k, v, causal=True)


def _run_sdpa():
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)


_DISPATCH = {"triton_prefill": _run_triton, "torch_sdpa": _run_sdpa}
if KERNEL not in _DISPATCH:
    print(f"ERROR: unknown kernel '{KERNEL}'. Choose: {list(_DISPATCH)}", file=sys.stderr)
    sys.exit(1)

run = _DISPATCH[KERNEL]

# ── warmup — lets autotune complete before NCU attaches ────────────────────────
for _ in range(WARMUP):
    run()
torch.cuda.synchronize()

# ── single profiled call — NCU attaches here ───────────────────────────────────
out = run()
torch.cuda.synchronize()

# Print a summary so the script is useful standalone too.
sm_scale = 1.0 / math.sqrt(D)
flops = 4 * B * H * S * S * D // 2  # causal halves score region
print(
    f"profile_one: kernel={KERNEL} B={B} H={H} S={S} D={D} dtype={DTYPE_NAME} "
    f"out_shape={tuple(out.shape)} analytical_GFLOP={flops/1e9:.3f}"
)
