# RESULTS

Cross-architecture benchmark results for flash-attn-lab. Numbers in this document come from `bench/harness.py` runs only; nothing is hand-edited.

Status legend: `TBD` = not yet measured.

## Methodology

- Warmup: 10 iterations per config. Measurement: 200 iterations.
- Reported latency: median over the measurement window. p95 is also stored in the CSV.
- Achieved TFLOP/s: analytical forward flop count for fused attention (see `bench/harness.attention_flops`) divided by median wall time. Causal halves the score region; the flop count reflects this.
- Achieved HBM GB/s: analytical bytes-touched for Q, K, V reads + Out write. This understates true HBM traffic for fused kernels because intermediate spills are not counted.
- All matrices are non-padded; sequence lengths are powers of two.
- Reference oracle: pure PyTorch `softmax(QK^T / sqrt(d)) V` in fp32. Tolerance for fp16/bf16 fused kernels: `atol=1e-2, rtol=1e-2`.

## Per-GPU results

### T4 (SM75)

| seqlen | head_dim | dtype | kernel | median ms | TFLOP/s | HBM GB/s |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 64 | fp16 | torch_sdpa     | TBD | TBD | TBD |
| 1024 | 64 | fp16 | triton_prefill | TBD | TBD | TBD |
| 4096 | 64 | fp16 | torch_sdpa     | TBD | TBD | TBD |
| 4096 | 64 | fp16 | triton_prefill | TBD | TBD | TBD |

### A100 (SM80)

| seqlen | head_dim | dtype | kernel | median ms | TFLOP/s | HBM GB/s |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 64  | bf16 | torch_sdpa     | TBD | TBD | TBD |
| 1024 | 64  | bf16 | triton_prefill | TBD | TBD | TBD |
| 4096 | 128 | bf16 | torch_sdpa     | TBD | TBD | TBD |
| 4096 | 128 | bf16 | triton_prefill | TBD | TBD | TBD |
| 8192 | 128 | bf16 | torch_sdpa     | TBD | TBD | TBD |
| 8192 | 128 | bf16 | triton_prefill | TBD | TBD | TBD |

### L4 (SM89)

| seqlen | head_dim | dtype | kernel | median ms | TFLOP/s | HBM GB/s |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 64  | bf16 | triton_prefill | TBD | TBD | TBD |
| 4096 | 128 | bf16 | triton_prefill | TBD | TBD | TBD |

### RTX PRO 6000 (Blackwell, SM120)

| seqlen | head_dim | dtype | kernel | median ms | TFLOP/s | HBM GB/s |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 128 | bf16 | triton_prefill | TBD | TBD | TBD |
| 8192 | 128 | bf16 | triton_prefill | TBD | TBD | TBD |

## Cross-architecture comparison

The purpose of this section is to contrast where each architecture sits on the roofline at the same workload. Every cell is `TBD` until the sweep has been run on each device.

| GPU | seqlen | head_dim | TFLOP/s | % peak fp16 TC | HBM GB/s | % peak HBM | bound by |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T4 (SM75)        | 4096 | 64 | TBD | TBD | TBD | TBD | TBD |
| A100 40GB (SM80) | 4096 | 64 | TBD | TBD | TBD | TBD | TBD |
| L4 (SM89)        | 4096 | 64 | TBD | TBD | TBD | TBD | TBD |
| RTX PRO 6000 (SM120) | 4096 | 64 | TBD | TBD | TBD | TBD | TBD |

## Bottleneck analysis

Per-GPU breakdown of where the kernel spends its time, measured via Nsight Compute's `--set full` profile saved under `profiles/`. Sections to populate per device:

- Tensor-core utilization (sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active): TBD
- HBM read throughput (dram__bytes_read.sum.per_second): TBD
- L2 hit rate (lts__t_sectors.hit_rate): TBD
- Achieved occupancy (sm__warps_active.avg.pct_of_peak_sustained_active): TBD
- Stall reason mix: TBD

## FP8 numerics

The FP8 kernel is a phase-5 task. This section will report:

- Output MSE vs the bf16 reference, per (seqlen, head_dim).
- Per-tensor vs per-head scaling: TBD.
- e4m3 vs bf16 input dtype: TBD.

| seqlen | head_dim | scaling | MSE vs bf16 ref | TFLOP/s | speedup vs bf16 |
| --- | --- | --- | --- | --- | --- |
| 4096 | 128 | per-tensor | TBD | TBD | TBD |
| 4096 | 128 | per-head   | TBD | TBD | TBD |
