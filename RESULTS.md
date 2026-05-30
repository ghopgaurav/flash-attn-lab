# RESULTS

Cross-architecture benchmark results for flash-attn-lab. Numbers come from
`bench/harness.py` runs and Nsight Compute profiles. Nothing is hand-edited;
all latency/TFLOP/HBM figures are median over 200 iterations.

Status legend: `TBD` = not yet measured on that device.

---

## Methodology

- **Sweep harness**: `bench/harness.py`. Warmup 10 iters, measure 200 iters.
- **Latency**: median. p95 also stored in the per-GPU CSV under `bench/results/`.
- **Achieved TFLOP/s**: `attention_flops(B,H,M,N,D,causal)` / median wall time. Causal halves the score region; the FLOP count reflects this.
- **Achieved HBM GB/s**: bytes touched (Q read + K read + V read + Out write), no spill correction. Understates true HBM traffic.
- **% of peak**: achieved / hardware spec ceiling. Peak figures from NVIDIA datasheets (see `bench/roofline.py` header for citations).
- **Arithmetic intensity**: achieved TFLOP/s / achieved HBM GB/s → FLOP/byte.
- **NCU hardware counters**: `bench/ncu_extract.py` with `--set full`. The SOL (Speed of Light) compute and memory throughput percentages are the NCU ground truth and differ from the FLOP-count-derived % above.
- All shapes use `causal=True`, bf16 (or fp16 on SM75), square Q/K/V.
- Reference oracle: `utils.reference.reference_attention_prefill` (pure PyTorch fp32).

---

## Per-GPU results

### A100-SXM4-40GB (SM80) — 2026-05-29

Peak: **312 TFLOP/s** bf16 TC, **1555 GB/s** HBM2e.
CSV: `bench/results/a100.csv`

| seqlen | head_dim | B | kernel | median ms | TFLOP/s | % TC peak | HBM GB/s | % HBM peak | AI (F/B) | bound |
|---|---|---|---|---|---|---|---|---|---|---|
| 512 | 64 | 1 | torch_sdpa | 0.043 | 6.2 | 2.0% | 24.3 | 1.6% | 256 | Compute |
| 512 | 64 | 1 | triton_prefill | 0.099 | 2.7 | 0.9% | 10.6 | 0.7% | 256 | Compute |
| 1024 | 64 | 1 | torch_sdpa | 0.064 | 16.9 | 5.4% | 66.0 | 4.2% | 256 | Compute |
| 1024 | 64 | 1 | triton_prefill | 0.113 | 9.5 | 3.1% | 37.3 | 2.4% | 256 | Compute |
| 2048 | 64 | 1 | torch_sdpa | 0.118 | 36.6 | 11.7% | 71.6 | 4.6% | 512 | Compute |
| 2048 | 64 | 1 | triton_prefill | 0.178 | 24.3 | 7.8% | 47.5 | 3.1% | 512 | Compute |
| 4096 | 64 | 1 | torch_sdpa | 0.309 | 56.0 | 17.9% | 54.7 | 3.5% | 1024 | Compute |
| 4096 | 64 | 1 | triton_prefill | 0.399 | 43.3 | 13.9% | 42.3 | 2.7% | 1024 | Compute |
| 4096 | 128 | 1 | torch_sdpa | 0.491 | 70.5 | 22.6% | 68.9 | 4.4% | 1024 | Compute |
| 4096 | 128 | 1 | triton_prefill | 0.679 | 51.0 | 16.3% | 49.8 | 3.2% | 1024 | Compute |
| 8192 | 128 | 1 | torch_sdpa | 0.937 | 147.7 | 47.3% | 72.2 | 4.6% | 2048 | Compute |
| 8192 | 128 | 1 | triton_prefill | 1.188 | 116.4 | 37.3% | 56.9 | 3.7% | 2048 | Compute |
| 8192 | 128 | 2 | torch_sdpa | 2.886 | 96.1 | 30.8% | 47.0 | 3.0% | 2048 | Compute |
| 8192 | 128 | 2 | triton_prefill | 3.936 | 70.5 | 22.6% | 34.5 | 2.2% | 2048 | Compute |

> All numbers are median-over-200-iterations. H=16 heads for all rows.
> % TC peak = TFLOP/s / 312. % HBM peak = HBM GB/s / 1555.
> These are FLOP-count-derived; see Bottleneck Analysis for NCU hardware-counter ground truth.

### T4 (SM75) — not yet run

| seqlen | head_dim | dtype | kernel | median ms | TFLOP/s | % TC peak | HBM GB/s |
|---|---|---|---|---|---|---|---|
| 1024 | 64 | fp16 | torch_sdpa | TBD | TBD | TBD | TBD |
| 1024 | 64 | fp16 | triton_prefill | TBD | TBD | TBD | TBD |
| 4096 | 64 | fp16 | torch_sdpa | TBD | TBD | TBD | TBD |
| 4096 | 64 | fp16 | triton_prefill | TBD | TBD | TBD | TBD |

### L4 (SM89) — not yet run

| seqlen | head_dim | dtype | kernel | median ms | TFLOP/s | % TC peak | HBM GB/s |
|---|---|---|---|---|---|---|---|
| 1024 | 64 | bf16 | triton_prefill | TBD | TBD | TBD | TBD |
| 4096 | 128 | bf16 | triton_prefill | TBD | TBD | TBD | TBD |

### RTX PRO 6000 (Blackwell SM120) — not yet run

| seqlen | head_dim | dtype | kernel | median ms | TFLOP/s | % TC peak | HBM GB/s |
|---|---|---|---|---|---|---|---|
| 4096 | 128 | bf16 | triton_prefill | TBD | TBD | TBD | TBD |
| 8192 | 128 | bf16 | triton_prefill | TBD | TBD | TBD | TBD |

---

## Cross-architecture comparison (triton_prefill, S=4096, D=128, B=1, H=16, causal)

| GPU | SM | TFLOP/s | % TC peak | HBM GB/s | % HBM peak | AI (F/B) | bound |
|---|---|---|---|---|---|---|---|
| A100-SXM4-40GB | SM80 | 51.0 | 16.3% | 49.8 | 3.2% | 1024 | Compute |
| T4 | SM75 | TBD | TBD | TBD | TBD | TBD | TBD |
| L4 | SM89 | TBD | TBD | TBD | TBD | TBD | TBD |
| RTX PRO 6000 | SM120 | TBD | TBD | TBD | TBD | TBD | TBD |

> The cross-architecture comparison at fixed (S, D, B, H) isolates the tensor-core generation effect.
> The asymmetric hardware scaling narrative (FA4 motivation) predicts that the matmul fraction
> grows relative to non-matmul as we move from SM80 → SM89 → SM120 due to asymmetric
> tensor-core speedup. This will be visible as % TC peak rising faster than % HBM peak across generations.

---

## Bottleneck analysis — NCU hardware counters

Run `bench/ncu_extract.py` to populate this section. The NCU Speed of Light (SOL) metrics are
the credible source for "% of ceiling" — hardware cycle counters, not FLOP estimates.

```bash
# On Colab (requires sudo for perf counters):
sudo ncu \
    --set full \
    --target-processes all \
    --kernel-name regex:_attn_fwd_kernel \
    --csv \
    -o profiles/triton_prefill_a100_s4096_d128 \
    python bench/profile_one.py

# Or use the helper (runs NCU and parses stdout in one step):
sudo python bench/ncu_extract.py --kernel triton_prefill --profile-args "B=1,H=16,S=4096,D=128"

# Offline: compute % of peak from the bench CSV (no NCU needed):
python bench/ncu_extract.py --csv bench/results/a100.csv
```

### A100-SXM4-40GB — triton_prefill (S=4096, D=128, B=1, H=16, causal)

Profile: `profiles/triton_prefill_a100_s4096_d128.ncu-rep` (NCU 2025.1.1, 2026-05-29).
Autotune winner: BLOCK_M=128, num_warps=8 — block=(256 threads), grid=(32, 16).

| Metric | Value | NCU source |
|---|---|---|
| SM compute throughput (SOL) | **29.9% of peak** | `GPU Speed Of Light Throughput / Compute (SM) Throughput` |
| Tensor-core pipe utilization | **37.7% of active cycles** | `ComputeWorkloadAnalysis` — Tensor FP sub-pipeline, highest-utilized |
| Memory throughput (L1+L2+DRAM total) | **64.5 GB/s** | `Memory Workload Analysis / Memory Throughput` |
| Theoretical occupancy | **12.5%** (2 warps/scheduler) | `Occupancy / TheoreticalOccupancy` — limited by registers + shared memory |
| Achieved occupancy | **12.5%** | `Occupancy / Achieved Occupancy` |
| Warp issue slot utilization | **35.7%** (1 instr per 2.8 cycles) | `SchedulerStats / IssueSlotUtilization` |
| Primary stall | **Fixed-latency dep — 40.7% of CPI** | `WarpStateStats` — tensor-core result not ready |
| Shared memory bank conflicts | **~20-22%** excess wavefronts on stores | `MemoryWorkloadAnalysis_Tables / SharedMemoryConflicts` |
| **Bottleneck** | **Latency-limited** (occupancy-constrained) | NCU SOLBottleneck: both compute and memory below 60% |

**Root cause:** With occupancy capped at 12.5% (2 warps/scheduler, limited by shared memory tile size + register pressure), the A100 cannot hide the ~6-cycle fixed-latency stall from tensor-core output dependencies. The scheduler is idle 64% of cycles. The tensor-core pipeline itself is healthy at 38% active-cycle utilization — the bottleneck is latency hiding, not arithmetic throughput or HBM bandwidth. This is the regime addressed by FA3/FA4 warp-specialized persistent kernel designs. The 20–22% shared-memory bank-conflict rate on stores is a secondary 7% overhead worth addressing in Phase 2 tiling work.

**Note on metric interpretation:** NCU's "SM Compute Throughput (29.9%)" is higher than the analytical TFLOP/s percentage (51.0 / 312 = 16.3%) because NCU measures all SM instruction throughput (including non-GEMM ops like softmax, indexing, loads), while the analytical metric only counts mma flops. The gap quantifies the non-matmul fraction of kernel time — directly relevant to the asymmetric hardware scaling narrative.

### A100-SXM4-40GB — torch_sdpa (S=4096, D=128, B=1, H=16, causal)

Not yet profiled. To produce the equivalent profile:
```bash
PROF_KERNEL=torch_sdpa /usr/local/cuda/bin/ncu \
    --set full --target-processes all \
    --kernel-name regex:fmha \
    --csv -o profiles/torch_sdpa_a100_s4096_d128 \
    python bench/profile_one.py
```

Analytical bench comparison (S=4096, D=128, B=1, H=16): torch_sdpa 70.5 TFLOP/s (22.6% of 312) vs triton_prefill 51.0 TFLOP/s (16.3%). NCU SM SOL for torch_sdpa expected to be higher than triton_prefill due to FA2's better occupancy and software pipelining.

| Metric | Value | NCU source |
|---|---|---|
| SM compute throughput (SOL) | TBD | pending profile |
| Tensor-core pipe utilization | TBD | pending profile |
| Memory throughput | TBD | pending profile |
| Achieved occupancy | TBD | pending profile |
| **Bottleneck** | TBD | pending profile |

---

## FP8 numerics (Phase 5 — SM89+ only)

| seqlen | head_dim | scaling | MSE vs bf16 ref | TFLOP/s | speedup vs bf16 |
|---|---|---|---|---|---|
| 4096 | 128 | per-tensor | TBD | TBD | TBD |
| 4096 | 128 | per-head | TBD | TBD | TBD |
