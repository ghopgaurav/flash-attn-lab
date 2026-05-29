"""Plots derived from sweep CSVs.

Every plot in the README/RESULTS docs is generated from this module against
the merged sweep CSVs under `bench/results/`. None of these functions
generate synthetic data — if a CSV is missing or empty, the plot is skipped
with a warning.

Plots:
    plot_latency_vs_seqlen
    plot_tflops_vs_seqlen
    plot_kernel_speedup_bar
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


def _load_ok(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv)
    df = df.loc[df["ok"].astype(str).str.lower().isin({"true", "1"})].copy()
    return df


def plot_latency_vs_seqlen(
    csv: Path,
    out_path: Path,
    head_dim: Optional[int] = None,
    batch: Optional[int] = None,
    num_heads: Optional[int] = None,
) -> Optional[Path]:
    df = _load_ok(csv)
    if df.empty:
        logger.warning("no rows; skipping %s", out_path)
        return None
    if head_dim is not None:
        df = df[df["head_dim"] == head_dim]
    if batch is not None:
        df = df[df["batch"] == batch]
    if num_heads is not None:
        df = df[df["num_heads"] == num_heads]
    if df.empty:
        logger.warning("filter eliminated all rows; skipping %s", out_path)
        return None

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for kernel, sub in df.groupby("kernel"):
        sub = sub.sort_values("seqlen")
        ax.plot(sub["seqlen"], sub["median_ms"], marker="o", label=str(kernel))
    ax.set_xlabel("seqlen")
    ax.set_ylabel("median latency (ms)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.grid(True, which="both", linewidth=0.4, alpha=0.4)
    ax.legend(fontsize=8)
    title_bits = ["latency vs seqlen"]
    if head_dim is not None:
        title_bits.append(f"D={head_dim}")
    if num_heads is not None:
        title_bits.append(f"H={num_heads}")
    if batch is not None:
        title_bits.append(f"B={batch}")
    ax.set_title(", ".join(title_bits))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_tflops_vs_seqlen(csv: Path, out_path: Path) -> Optional[Path]:
    df = _load_ok(csv)
    if df.empty:
        logger.warning("no rows; skipping %s", out_path)
        return None
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for kernel, sub in df.groupby("kernel"):
        sub = sub.sort_values("seqlen")
        ax.plot(sub["seqlen"], sub["tflops_achieved"].astype(float), marker="o", label=str(kernel))
    ax.set_xlabel("seqlen")
    ax.set_ylabel("achieved TFLOP/s")
    ax.set_xscale("log", base=2)
    ax.grid(True, which="both", linewidth=0.4, alpha=0.4)
    ax.legend(fontsize=8)
    ax.set_title("achieved TFLOP/s vs seqlen")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_kernel_speedup_bar(
    csv: Path,
    out_path: Path,
    baseline: str = "torch_sdpa",
) -> Optional[Path]:
    df = _load_ok(csv)
    if df.empty:
        logger.warning("no rows; skipping %s", out_path)
        return None
    if baseline not in df["kernel"].unique():
        logger.warning("baseline %s missing; skipping %s", baseline, out_path)
        return None

    keys = ["dtype", "batch", "num_heads", "seqlen", "head_dim"]
    base = df[df["kernel"] == baseline].set_index(keys)["median_ms"].astype(float)
    others = df[df["kernel"] != baseline].copy()
    others["speedup"] = others.apply(
        lambda r: base.get(tuple(r[k] for k in keys), float("nan")) / float(r["median_ms"]),
        axis=1,
    )
    grouped = others.groupby("kernel")["speedup"].mean().sort_values(ascending=False)
    if grouped.empty:
        logger.warning("no non-baseline kernels; skipping %s", out_path)
        return None

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(grouped.index.tolist(), grouped.values.tolist())
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel(f"mean speedup over {baseline}")
    ax.set_title(f"speedup over {baseline} (geomean over configs)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Generate plots from a sweep CSV")
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("plots"))
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_latency_vs_seqlen(args.csv, args.out_dir / "latency_vs_seqlen.png")
    plot_tflops_vs_seqlen(args.csv, args.out_dir / "tflops_vs_seqlen.png")
    plot_kernel_speedup_bar(args.csv, args.out_dir / "speedup_bar.png")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
