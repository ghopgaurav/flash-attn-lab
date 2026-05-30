# flash-attn-lab

From-scratch fused attention in Triton and raw CUDA C++, benchmarked across multiple GPU architectures.

## Thesis

A from-scratch implementation of fused attention — Triton kernels and raw-CUDA kernels for the prefill and decode regimes, benchmarked across Ampere (A100), Ada (L4), and Blackwell SM120 (RTX PRO 6000) via Google Colab. The headline result is a cross-architecture roofline study that shows how the matmul vs non-matmul balance shifts across tensor-core generations — engaging the "asymmetric hardware scaling" narrative that motivated FlashAttention-4.

## Headline results

![cross-architecture roofline](plots/roofline.png)

| GPU | SM | dtype | S | D | kernel | TFLOP/s | % TC peak | HBM GB/s | % HBM peak | AI (F/B) | NCU SOL compute |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A100-SXM4-40GB | 80 | bf16 | 8192 | 128 | triton_prefill | 116.4 | 37.3% | 56.9 | 3.7% | 2048 | TBD (NCU pending) |
| A100-SXM4-40GB | 80 | bf16 | 8192 | 128 | torch_sdpa | 147.7 | 47.3% | 72.2 | 4.6% | 2048 | TBD (NCU pending) |
| T4 | 75 | fp16 | 4096 | 64 | triton_prefill | TBD | TBD | TBD | TBD | TBD | TBD |
| L4 | 89 | bf16 | 4096 | 128 | triton_prefill | TBD | TBD | TBD | TBD | TBD | TBD |
| RTX PRO 6000 | 120 | bf16 | 8192 | 128 | triton_prefill | TBD | TBD | TBD | TBD | TBD | TBD |

A100 numbers are measured (2026-05-29, `bench/results/a100.csv`). % TC peak = TFLOP/s / 312. % HBM peak = HBM GB/s / 1555. NCU SOL compute = hardware-counter ground truth — run `bench/ncu_extract.py` to populate. See [RESULTS.md](RESULTS.md) for the full per-GPU table.

## What's in this repo

```
flash_attn_lab/
  triton_kernels/   prefill (working) + skeletons for decode, GQA, FP8
  cuda_kernels/     matmul, reduction, online softmax, decode attention (.cu) + pybind11 bindings
  ops/              public attention() op, registered via torch.library.custom_op
  utils/            device introspection + PyTorch reference implementations
bench/              sweep harness, roofline analysis, plotting
tests/              correctness checks against the PyTorch reference
profiles/           Nsight Compute (.ncu-rep) artifacts go here
notebooks/          colab_setup.ipynb (mount Drive, install, smoke test)
plots/              generated plots (roofline, latency, speedup)
```

## Quickstart

```bash
git clone https://github.com/ghopgaurav/flash-attn-lab.git
cd flash-attn-lab
pip install -e .
python -m bench.harness --device-check
```

`pip install -e .` works on CPU-only machines; the CUDA kernels are JIT-compiled on first use, so there's nothing to build at install time. On Colab, run `notebooks/colab_setup.ipynb` end-to-end.

## Kernels implemented

| kernel | language | regime | status |
| --- | --- | --- | --- |
| `triton_attention_prefill` | Triton | prefill (M=N, causal/non-causal) | working |
| `triton_decode_attention` | Triton | decode (M=1, KV cache) | skeleton |
| `triton_gqa_attention` | Triton | prefill, grouped KV heads | skeleton |
| `triton_fp8_attention` | Triton | prefill, FP8 (SM89+) | skeleton |
| `matmul_tiled` | CUDA C++ | tiled FP32 GEMM | working |
| `row_sum` | CUDA C++ | warp-shuffle reduction | working |
| `softmax_online` | CUDA C++ | row-wise online softmax | working |
| `decode_attention` | CUDA C++ | naive fused decode | working |

## Reproduction

The benchmark sweep is intended to be re-run on each GPU. The harness writes one CSV row per (kernel, dtype, batch, heads, seqlen, head_dim) and tags every row with the GPU name, SM version, git SHA, and timestamp.

```bash
python -m bench.harness \
    --kernels torch_sdpa triton_prefill \
    --dtypes bf16 \
    --seqlens 512 1024 2048 4096 8192 \
    --head-dims 64 128 \
    --num-heads 16 \
    --batch-sizes 1 2 \
    --out bench/results/$(hostname).csv \
    --checkpoint-dir /content/drive/MyDrive/flash_attn_lab_checkpoints
```

After the sweep, generate the roofline and latency plots:

```bash
python -m bench.roofline --csv bench/results/<file>.csv --out plots/roofline.png
python -m bench.plot     --csv bench/results/<file>.csv --out-dir plots/
```

To attach Nsight Compute to a single config:

```bash
ncu --set full --target-processes all \
    -o profiles/triton_prefill_a100_4096_d64 \
    python -m bench.harness --smoke
```

## Design and decisions

See [DESIGN_NOTES.md](DESIGN_NOTES.md) for the running log of design decisions, autotune choices, dtype matrices, and any TODOs flagged during a session.

## Limitations

This repo is honest about what it does and does not do.

- **No FlashAttention-3 baseline.** FA3 requires Hopper (SM90); none of the targeted Colab GPUs are Hopper. The cross-architecture story is interesting precisely because it covers everything *except* Hopper.
- **No multi-GPU.** All kernels are single-device; there is no NCCL, no tensor-parallel split, no sequence-parallel split.
- **Forward-only.** The backward pass is registered as a `torch.library` autograd stub that raises `NotImplementedError` with a clear pointer to phase 4. Use `torch.nn.functional.scaled_dot_product_attention` if you need a backward pass today.
- **No graph mode lowering.** `torch.compile` works because the op has a fake-tensor implementation, but no Inductor-level fusion templates have been added.
- **Not yet optimized.** The CUDA decode-attention kernel is a correctness-first baseline (intentional — see DESIGN_NOTES.md).

## References

- Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS 2022.
- Tri Dao. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.* 2023.
- Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao. *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision.* 2024.
- Jay Shah, Tri Dao, et al. *FlashAttention-4* (announcement / blog post, late 2025).
- Triton tutorials: [Fused Softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html), [Matrix Multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html), [Fused Attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html).
- NVIDIA. *CUDA C++ Programming Guide*, current revision.

## License

MIT. See [LICENSE](LICENSE).
