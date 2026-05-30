# DESIGN_NOTES

Append-only log of design decisions, trade-offs, and TODOs. New entries go on top.

---

## 2026-05-29 — Phase 1 A100 baseline results and NCU bottleneck analysis

First real GPU run on A100-SXM4-40GB (SM80, CUDA 12.8, PyTorch 2.11). Sweep: `torch_sdpa` and `triton_prefill`, bf16, S∈{512,1024,2048,4096,8192}, D∈{64,128}, H=16, B∈{1,2}, causal.

**Autotune winner (S=4096, D=128):** BLOCK_M=128, num_warps=8. Grid=(32,16), block=256 threads.

**NCU bottleneck finding (S=4096, D=128, B=1, H=16):**
- SM Compute SOL: **29.9%**. Tensor-core pipe: **37.7% of active cycles** — pipeline is healthy but underutilized due to low occupancy.
- Achieved occupancy: **12.5%** (theoretical max for this config — shared memory tile consumes most of the 164 KB shared memory budget, limiting to 2 warps/scheduler).
- Primary stall: **fixed-latency execution dependency at 40.7% of CPI**. With only 2 active warps per scheduler, A100 cannot hide the 6-cycle tensor-core output dependency. Scheduler is idle 64% of cycles.
- Shared-memory bank conflicts on stores: **~20-22% excess wavefronts** — 1.2-way conflicts from the Q tile layout. Addressable with padding.
- Bottleneck classification: **latency-limited**, not bandwidth-limited or compute-saturated. NCU explicitly flags this ("both SOL below 60% → indicates latency issues").

**Key insight on NCU SOL vs analytical TFLOP/s:**
NCU reports 29.9% SM SOL while analytical FLOP counting gives 16.3% of peak (51 TFLOP/s / 312). The gap is the non-matmul work: online softmax recurrence, masking, index arithmetic. This delta directly quantifies the "non-matmul tax" — the thing asymmetric hardware scaling in FA4 is designed to reduce.

**Decisions to revisit (updated from Phase 1 observations):**

- **Occupancy improvement (Phase 2 priority).** The ~48 KB shared memory usage per block (BLOCK_M=128 × HEAD_DIM=128 × 2 bytes × Q+running-state) is the occupancy ceiling. Options: (a) halve BLOCK_M to 64 (reduces tiles, more K-loop iterations, but doubles occupancy), (b) reduce to bf16 accumulation for the running acc (risky numerics), (c) software-pipeline the K/V loads using async copy (`tl.load` with `eviction_policy=evict_first`).
- **Shared-memory bank conflict fix.** The 1.2-way conflict on shared stores is from writing fp16 Q tiles in 128-element rows without padding. Adding 8 elements of padding per row (i.e., making the SMEM stride `HEAD_DIM + 8`) should eliminate it. ~7% latency improvement expected.
- **Autotune config space widening.** BLOCK_M=128, num_warps=8 won at S=4096, D=128. But the autotune did not explore num_stages=4, which enables more async prefetch overlap. Worth adding for SM89+ where L2 latency is lower.
- **Triton prefill gap vs torch_sdpa.** At S=8192, D=128, B=1: triton=116 TFLOP/s, sdpa=148 TFLOP/s (1.27× gap). At S=4096 it's 1.38×. The gap narrows at longer sequences, confirming occupancy rather than algorithm is the bottleneck (sdpa/FA2 uses persistent kernels with better warp utilization).

---

## 2026-05-29 — Initial scaffold

Repo bootstrapped. Working components: Triton prefill kernel (causal, fp16/bf16, autotuned), four CUDA C++ kernels (tiled GEMM, warp-shuffle reduction, online softmax, naive fused decode attention), `torch.library.custom_op` public surface with a fake-tensor implementation, sweep harness with Drive-checkpoint support, roofline + plotting modules, full test suite against PyTorch references.

Decisions to revisit:

- **Triton autotune config space.** Current sweep is small (BLOCK_M ∈ {64, 128}, BLOCK_N ∈ {32, 64, 128}, num_warps ∈ {4, 8}, num_stages ∈ {2, 3}). Should be widened once we have per-GPU baseline numbers; in particular SM120 likely wants num_stages ≥ 4 and BLOCK_M = 256.
- **CUDA shared-memory tile sizes.** Tiled GEMM uses 32×32 fp32 tiles, which is the textbook size and not optimal on any Ampere+. Worth a follow-up that compares 64×64 with register blocking.
- **FP8 scaling strategy.** Per-tensor vs per-head: per-tensor is the simpler first cut, but Llama-style attention sinks have outlier heads that benefit from per-head scales. Keep both behind a flag in `triton_fp8_attention`.
- **Backward pass approach.** Two options: (1) save (m, l) row stats from the forward and recompute P in the backward (the FA2 strategy), or (2) recompute the forward inside the backward. Option (1) is the only one with reasonable memory; doing it requires extending the public op signature to optionally return the LSE. Defer to phase 4.
- **Decode regime in the public op.** Currently `attention()` raises NotImplementedError when M=1; once `triton_decode_attention` lands we should branch in `_select_kernel`.
- **GQA-aware harness.** The sweep config doesn't yet take `num_kv_heads`; add it when the GQA kernel lands.
- **Roofline peak table.** The Blackwell SM120 entry uses an estimated bf16 tensor-core figure; verify against the official RTX PRO 6000 spec sheet before we cite it in RESULTS.md.
