"""Roofline analysis.

Given a CSV of kernel measurements (from `bench.harness`), compute arithmetic
intensity (FLOP/byte) for each kernel and overlay them on the GPU's
roofline. The peak-spec table below is hardcoded with citations; if you add
a new GPU, add a row with the source.

Peak numbers correspond to dense fp16/bf16 tensor-core throughput (the
relevant axis for fused attention) and HBM bandwidth.

Sources for peak specs:
    - T4               (SM75): NVIDIA T4 datasheet (April 2019).
                       fp16 tensor cores: 65 TFLOP/s, HBM: 320 GB/s.
    - A100 40GB        (SM80): NVIDIA A100 datasheet rev 2.0 (June 2021).
                       bf16 tensor cores: 312 TFLOP/s, HBM2e: 1555 GB/s.
    - A100 80GB        (SM80): NVIDIA A100 80GB datasheet.
                       bf16 tensor cores: 312 TFLOP/s, HBM2e: 2039 GB/s.
    - L4               (SM89): NVIDIA L4 datasheet (March 2023).
                       bf16 tensor cores: 121 TFLOP/s, GDDR6: 300 GB/s.
    - RTX PRO 6000 (Blackwell, SM120): NVIDIA RTX PRO 6000 datasheet
                       (April 2025). bf16 tensor cores: 503.8 TFLOP/s,
                       GDDR7: 1792 GB/s. Verify against NVIDIA's published
                       spec sheet before citing in RESULTS.md.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")  # safe in headless / Colab no-display contexts
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GpuPeak:
    name: str
    sm: str
    fp16_tflops: float
    hbm_gb_s: float
    source: str


GPU_PEAKS: dict[str, GpuPeak] = {
    "Tesla T4": GpuPeak("Tesla T4", "SM75", 65.0, 320.0, "NVIDIA T4 datasheet"),
    "NVIDIA A100-PCIE-40GB": GpuPeak(
        "A100 40GB", "SM80", 312.0, 1555.0, "NVIDIA A100 datasheet"
    ),
    "NVIDIA A100-SXM4-40GB": GpuPeak(
        "A100 40GB SXM", "SM80", 312.0, 1555.0, "NVIDIA A100 datasheet"
    ),
    "NVIDIA A100-SXM4-80GB": GpuPeak(
        "A100 80GB", "SM80", 312.0, 2039.0, "NVIDIA A100 80GB datasheet"
    ),
    "NVIDIA L4": GpuPeak("L4", "SM89", 121.0, 300.0, "NVIDIA L4 datasheet"),
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition": GpuPeak(
        "RTX PRO 6000", "SM120", 503.8, 1792.0, "NVIDIA RTX PRO 6000 datasheet"
    ),
}


def lookup_peaks(gpu_name: str) -> Optional[GpuPeak]:
    """Match a GPU name from `torch.cuda.get_device_properties().name` to a peak row.

    The match is case-insensitive substring; this is robust to small naming
    differences across CUDA versions (e.g. "NVIDIA " prefix changes).
    """
    needle = gpu_name.lower()
    for key, peak in GPU_PEAKS.items():
        if key.lower() in needle or needle in key.lower():
            return peak
    return None


def compute_intensity(df: pd.DataFrame) -> pd.DataFrame:
    """Add `flops`, `bytes`, `intensity` columns from existing measurements.

    Uses the achieved TFLOP/s and HBM GB/s columns: intensity = flops / bytes
    is the same across both achieved and analytical, but this lets us plot
    both achieved performance and the kernel's own arithmetic intensity.
    """
    df = df.copy()
    # achieved flops per second / achieved bytes per second = ops/byte.
    flops_per_s = df["tflops_achieved"].astype(float) * 1e12
    bytes_per_s = df["hbm_gb_s_achieved"].astype(float) * 1e9
    df["intensity_flops_per_byte"] = flops_per_s / bytes_per_s.replace(0, float("nan"))
    df["achieved_tflops"] = df["tflops_achieved"].astype(float)
    return df


def plot_roofline(
    df: pd.DataFrame,
    peak: GpuPeak,
    out_path: Path,
    title: Optional[str] = None,
) -> Path:
    """Produce a roofline plot to `out_path` (PNG)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = compute_intensity(df.loc[df["ok"].astype(str).str.lower().isin({"true", "1"})])

    fig, ax = plt.subplots(figsize=(8, 5))

    # Roofline curves: y = min(peak_flops, intensity * peak_bw).
    intensities = [2 ** i for i in range(-2, 12)]  # 0.25 .. 2048
    peak_flops_t = peak.fp16_tflops
    peak_bw_t = peak.hbm_gb_s / 1000.0  # TB/s
    roof_y = [min(peak_flops_t, i * peak_bw_t) for i in intensities]
    ax.plot(intensities, roof_y, label=f"roofline ({peak.name})", color="black", linewidth=1.5)

    ax.axhline(peak_flops_t, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(intensities[-1], peak_flops_t, f" {peak_flops_t:.0f} TFLOP/s", va="bottom", ha="right")

    if not df.empty:
        for kernel, sub in df.groupby("kernel"):
            ax.scatter(
                sub["intensity_flops_per_byte"],
                sub["achieved_tflops"],
                label=str(kernel),
                s=30,
                alpha=0.85,
            )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("achieved TFLOP/s")
    ax.set_title(title or f"Roofline — {peak.name} ({peak.sm})")
    ax.grid(True, which="both", linewidth=0.4, alpha=0.4)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Roofline plot from a sweep CSV")
    p.add_argument("--csv", type=Path, required=True, help="Sweep CSV from bench/harness.py")
    p.add_argument("--out", type=Path, default=Path("plots/roofline.png"))
    p.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="Override GPU name lookup; defaults to the gpu_name column in the CSV.",
    )
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    df = pd.read_csv(args.csv)
    if df.empty:
        logger.error("CSV is empty: %s", args.csv)
        return 2

    gpu_name = args.gpu or df["gpu_name"].iloc[0]
    peak = lookup_peaks(gpu_name)
    if peak is None:
        logger.error(
            "no peak-spec row for GPU '%s'; add it to GPU_PEAKS in roofline.py", gpu_name
        )
        return 3

    out = plot_roofline(df, peak, args.out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
