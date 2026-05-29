# DESIGN_NOTES

Append-only log of design decisions, trade-offs, and TODOs. New entries go on top.

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
