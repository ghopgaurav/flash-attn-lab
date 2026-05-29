"""Benchmark sweep harness.

Walks a config grid (seqlens x head_dims x num_heads x batch_sizes x dtypes
x kernels), times each combination, computes achieved TFLOP/s and HBM GB/s,
and emits one CSV row per measurement. Tags each row with GPU name, SM
version, dtype, kernel name, git SHA, and timestamp so a downstream notebook
can pivot per-GPU without re-running anything.

Designed to survive Colab disconnects:
    - `--checkpoint-dir DIR` flushes the CSV after every row to a file in
      DIR; resuming with the same out CSV will skip rows already present.
    - `--device-check` prints device info and exits without timing anything.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import logging
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Optional

import torch

from flash_attn_lab.utils.device import DeviceInfo, get_device_info, select_dtype
from flash_attn_lab.utils.reference import reference_attention_prefill

logger = logging.getLogger(__name__)


CSV_FIELDS = [
    "timestamp",
    "git_sha",
    "gpu_name",
    "sm",
    "kernel",
    "regime",
    "dtype",
    "batch",
    "num_heads",
    "seqlen",
    "head_dim",
    "causal",
    "median_ms",
    "p95_ms",
    "tflops_achieved",
    "hbm_gb_s_achieved",
    "ok",
    "note",
]


@dataclass
class BenchConfig:
    seqlens: list[int]
    head_dims: list[int]
    num_heads: list[int]
    batch_sizes: list[int]
    dtypes: list[str]
    kernels: list[str]
    causal: bool = True
    warmup: int = 10
    iters: int = 200

    @classmethod
    def smoke(cls) -> "BenchConfig":
        return cls(
            seqlens=[1024],
            head_dims=[64],
            num_heads=[8],
            batch_sizes=[2],
            dtypes=["bf16"],
            kernels=["torch_sdpa", "triton_prefill"],
            warmup=2,
            iters=10,
        )


# ----- timing primitives -----------------------------------------------------


def _cuda_time_ms(fn, warmup: int, iters: int) -> tuple[float, float]:
    """Return (median_ms, p95_ms) over `iters` after `warmup` warmups."""
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    median = samples[len(samples) // 2]
    p95_idx = max(0, int(0.95 * len(samples)) - 1)
    return median, samples[p95_idx]


# ----- analytical FLOP / byte counts ----------------------------------------


def attention_flops(B: int, H: int, M: int, N: int, D: int, causal: bool) -> int:
    """Forward-only flop count for fused attention.

    QK^T:  2 * B * H * M * N * D  (one multiply + one add per inner product)
    softmax: ignored (~M*N exps and adds, dominated by matmuls)
    PV:    2 * B * H * M * N * D
    Causal halves the score region on average -> 1/2 factor on both matmuls.
    """
    base = 4 * B * H * M * N * D
    if causal and M == N:
        base //= 2
    return base


def attention_bytes(B: int, H: int, M: int, N: int, D: int, dtype_bytes: int) -> int:
    """HBM bytes touched: read Q, K, V; write Out. Ignores intermediate spills."""
    elems = B * H * (M + N + N + M) * D
    return elems * dtype_bytes


# ----- kernel adapters -------------------------------------------------------


def _make_inputs(B: int, H: int, S: int, D: int, dtype: torch.dtype, device: torch.device):
    g = torch.Generator(device=device).manual_seed(0)
    q = torch.randn((B, H, S, D), generator=g, device=device, dtype=dtype)
    k = torch.randn((B, H, S, D), generator=g, device=device, dtype=dtype)
    v = torch.randn((B, H, S, D), generator=g, device=device, dtype=dtype)
    return q, k, v


def _kernel_torch_sdpa(q, k, v, causal: bool):
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)


def _kernel_triton_prefill(q, k, v, causal: bool):
    from flash_attn_lab.triton_kernels.prefill import triton_attention_prefill

    return triton_attention_prefill(q, k, v, causal=causal)


def _kernel_reference(q, k, v, causal: bool):
    return reference_attention_prefill(q, k, v, causal=causal)


def _kernel_flash_attn(q, k, v, causal: bool):  # pragma: no cover - optional
    # FA2 expects (B, S, H, D); transpose only inside this adapter.
    from flash_attn import flash_attn_func  # type: ignore

    qb = q.transpose(1, 2).contiguous()
    kb = k.transpose(1, 2).contiguous()
    vb = v.transpose(1, 2).contiguous()
    out = flash_attn_func(qb, kb, vb, causal=causal)
    return out.transpose(1, 2)


_KERNEL_DISPATCH = {
    "torch_sdpa": _kernel_torch_sdpa,
    "triton_prefill": _kernel_triton_prefill,
    "reference": _kernel_reference,
    "flash_attn": _kernel_flash_attn,
}


def _resolve_dtype(name: str) -> torch.dtype:
    table = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    if name not in table:
        raise ValueError(f"unknown dtype: {name}")
    return table[name]


# ----- run loop --------------------------------------------------------------


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _existing_keys(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    seen = set()
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            seen.add(
                (
                    row.get("kernel"),
                    row.get("dtype"),
                    row.get("batch"),
                    row.get("num_heads"),
                    row.get("seqlen"),
                    row.get("head_dim"),
                    row.get("causal"),
                )
            )
    return seen


def run_sweep(
    cfg: BenchConfig,
    out_csv: Path,
    checkpoint_dir: Optional[Path] = None,
    info: Optional[DeviceInfo] = None,
) -> int:
    """Execute the sweep and write rows to `out_csv`. Returns rows written."""
    if info is None:
        info = get_device_info()
    if not info.is_cuda:
        raise RuntimeError("benchmarks require a CUDA GPU; got CPU-only environment")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = out_csv.exists()
    seen = _existing_keys(out_csv)
    git_sha = _git_sha()

    fout = out_csv.open("a", newline="")
    writer = csv.DictWriter(fout, fieldnames=CSV_FIELDS)
    if not file_exists:
        writer.writeheader()
        fout.flush()

    rows_written = 0
    device = torch.device("cuda")

    for kernel_name, dtype_name, B, H, S, D in product(
        cfg.kernels, cfg.dtypes, cfg.batch_sizes, cfg.num_heads, cfg.seqlens, cfg.head_dims
    ):
        key = (kernel_name, dtype_name, str(B), str(H), str(S), str(D), str(cfg.causal))
        if key in seen:
            logger.info("skip (already in CSV): %s", key)
            continue
        if kernel_name not in _KERNEL_DISPATCH:
            logger.warning("unknown kernel %s; skipping", kernel_name)
            continue

        try:
            dtype = _resolve_dtype(dtype_name)
            q, k, v = _make_inputs(B, H, S, D, dtype, device)
            fn = _KERNEL_DISPATCH[kernel_name]

            def _bench():
                fn(q, k, v, cfg.causal)

            # First call: triggers any autotune / JIT.
            _bench()
            torch.cuda.synchronize()

            median_ms, p95_ms = _cuda_time_ms(_bench, cfg.warmup, cfg.iters)

            flops = attention_flops(B, H, S, S, D, cfg.causal)
            seconds = median_ms / 1e3
            tflops = flops / seconds / 1e12

            dtype_bytes = torch.tensor([], dtype=dtype).element_size()
            bytes_moved = attention_bytes(B, H, S, S, D, dtype_bytes)
            gb_s = bytes_moved / seconds / 1e9

            row = {
                "timestamp": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "git_sha": git_sha,
                "gpu_name": info.name,
                "sm": info.sm_str,
                "kernel": kernel_name,
                "regime": "prefill",
                "dtype": dtype_name,
                "batch": B,
                "num_heads": H,
                "seqlen": S,
                "head_dim": D,
                "causal": cfg.causal,
                "median_ms": f"{median_ms:.4f}",
                "p95_ms": f"{p95_ms:.4f}",
                "tflops_achieved": f"{tflops:.3f}",
                "hbm_gb_s_achieved": f"{gb_s:.3f}",
                "ok": True,
                "note": "",
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("bench failed for %s", key)
            row = {f: "" for f in CSV_FIELDS}
            row.update(
                {
                    "timestamp": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "git_sha": git_sha,
                    "gpu_name": info.name,
                    "sm": info.sm_str,
                    "kernel": kernel_name,
                    "dtype": dtype_name,
                    "batch": B,
                    "num_heads": H,
                    "seqlen": S,
                    "head_dim": D,
                    "causal": cfg.causal,
                    "ok": False,
                    "note": f"{type(exc).__name__}: {exc}",
                }
            )

        writer.writerow(row)
        if checkpoint_dir is not None:
            fout.flush()
            os.fsync(fout.fileno())
            ckpt = checkpoint_dir / out_csv.name
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            try:
                # Best-effort copy to checkpoint dir for Colab Drive safety.
                ckpt.write_bytes(out_csv.read_bytes())
            except Exception:  # pragma: no cover
                logger.exception("failed to write checkpoint copy to %s", ckpt)
        else:
            fout.flush()

        rows_written += 1
        logger.info(
            "wrote row %d: %s %s B=%d H=%d S=%d D=%d -> %.3f ms",
            rows_written,
            kernel_name,
            dtype_name,
            B,
            H,
            S,
            D,
            float(row.get("median_ms") or 0.0),
        )

    fout.close()
    return rows_written


# ----- CLI -------------------------------------------------------------------


def _device_check_report(info: DeviceInfo) -> str:
    fields = dataclasses.asdict(info)
    width = max(len(k) for k in fields)
    lines = ["device-check report:"]
    for k, v in fields.items():
        lines.append(f"  {k:<{width}} : {v}")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="flash-attn-lab benchmark sweep")
    p.add_argument("--out", type=Path, default=Path("bench/results/sweep.csv"))
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="If set, copy the CSV here after every row (Colab disconnect safety).",
    )
    p.add_argument(
        "--device-check",
        action="store_true",
        help="Print device info and exit without running benchmarks.",
    )
    p.add_argument("--smoke", action="store_true", help="Run the tiny smoke config.")
    p.add_argument(
        "--seqlens",
        type=int,
        nargs="+",
        default=[512, 1024, 2048, 4096],
    )
    p.add_argument("--head-dims", type=int, nargs="+", default=[64, 128])
    p.add_argument("--num-heads", type=int, nargs="+", default=[8, 16])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2])
    p.add_argument(
        "--dtypes",
        type=str,
        nargs="+",
        default=["bf16"],
        choices=["fp32", "fp16", "bf16"],
    )
    p.add_argument(
        "--kernels",
        type=str,
        nargs="+",
        default=["torch_sdpa", "triton_prefill"],
    )
    p.add_argument("--causal", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose >= 2 else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    info = get_device_info()
    if args.device_check:
        print(_device_check_report(info))
        return 0

    if not info.is_cuda:
        print("no CUDA device available; refusing to run benchmarks", file=sys.stderr)
        return 2

    if args.smoke:
        cfg = BenchConfig.smoke()
    else:
        cfg = BenchConfig(
            seqlens=args.seqlens,
            head_dims=args.head_dims,
            num_heads=args.num_heads,
            batch_sizes=args.batch_sizes,
            dtypes=args.dtypes,
            kernels=args.kernels,
            causal=args.causal,
            warmup=args.warmup,
            iters=args.iters,
        )

    n = run_sweep(cfg, args.out, checkpoint_dir=args.checkpoint_dir, info=info)
    print(f"wrote {n} rows to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
