"""Parse NCU output and emit the hardware-counter metrics table.

Three usage modes:

  Mode A — run NCU inline and parse stdout:
      python bench/ncu_extract.py \\
          --kernel triton_prefill \\
          --profile-args "B=1,H=16,S=4096,D=128"

  Mode B — parse a previously saved NCU text report:
      python bench/ncu_extract.py --ncu-txt profiles/triton_prefill.txt

  Mode C — compute % of peak from the existing bench CSV only (no NCU):
      python bench/ncu_extract.py --csv bench/results/a100.csv

Mode C is the offline path that fills the README table without needing NCU.
Modes A / B require a CUDA GPU with NCU installed (sudo on Colab).
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

# ── A100-SXM4-40GB hardware ceilings (public spec sheets) ─────────────────────
# Source: NVIDIA A100 Tensor Core GPU Architecture whitepaper (2020).
A100_40GB_PEAK_BF16_TFLOPS = 312.0   # BF16 tensor core, no sparsity
A100_40GB_PEAK_FP16_TFLOPS = 312.0   # FP16 tensor core, no sparsity
A100_40GB_PEAK_HBM_GBS = 1555.0      # HBM2e bandwidth
A100_80GB_PEAK_HBM_GBS = 2039.0

# Default ceiling values (overridden by --gpu flag).
DEFAULT_PEAK_TFLOPS = A100_40GB_PEAK_BF16_TFLOPS
DEFAULT_PEAK_HBM = A100_40GB_PEAK_HBM_GBS


# ── NCU metric names (Nsight Compute 2023+) ────────────────────────────────────
# Speed of Light metrics that map to the "% ceiling" column.
_NCU_SOL_COMPUTE = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
_NCU_SOL_MEMORY  = "l1tex__throughput.avg.pct_of_peak_sustained_elapsed"
_NCU_TC_CYCLES   = "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"
_NCU_DRAM_BW     = "dram__bytes.sum.per_second"
_NCU_L2_HIT      = "lts__t_sectors_hit.avg.pct_of_peak_sustained_elapsed"
_NCU_OCCUPANCY   = "sm__warps_active.avg.pct_of_peak_sustained_active"

NCU_METRICS = ",".join([
    _NCU_SOL_COMPUTE,
    _NCU_SOL_MEMORY,
    _NCU_TC_CYCLES,
    _NCU_DRAM_BW,
    _NCU_L2_HIT,
    _NCU_OCCUPANCY,
])


# ── NCU invocation ─────────────────────────────────────────────────────────────

def find_ncu() -> str:
    """Return path to ncu binary, or raise."""
    for candidate in ["ncu", "/usr/local/cuda/bin/ncu"]:
        try:
            subprocess.run([candidate, "--version"], capture_output=True, check=True)
            return candidate
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass
    raise FileNotFoundError(
        "ncu not found. On Colab, try: sudo apt-get install cuda-nsight-compute-12-4"
    )


def run_ncu(kernel: str, prof_args: str, sudo: bool = True) -> str:
    """Invoke NCU on profile_one.py and return stdout text."""
    ncu = find_ncu()
    env_str = ""
    if prof_args:
        parts = [p.strip() for p in prof_args.split(",")]
        env_str = " ".join(f"PROF_{k.upper()}={v}" for k, v in (p.split("=") for p in parts))

    profile_script = str(Path(__file__).resolve().parent / "profile_one.py")
    cmd = (
        f"{'sudo ' if sudo else ''}{ncu} "
        f"--metrics {NCU_METRICS} "
        f"--target-processes all "
        f"--kernel-name regex:_attn_fwd_kernel "
        f"PROF_KERNEL={kernel} {env_str} "
        f"python {profile_script}"
    )
    print(f"[ncu_extract] running: {cmd}", file=sys.stderr)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout + result.stderr


# ── NCU text parser ────────────────────────────────────────────────────────────

def parse_ncu_text(text: str) -> dict[str, float]:
    """Extract metric name → value from NCU plain-text output."""
    metrics: dict[str, float] = {}
    # NCU text output has lines like:
    #   sm__throughput.avg.pct_of_peak_sustained_elapsed      %     52.30
    pattern = re.compile(
        r"([\w.]+)\s+"            # metric name
        r"(?:[%\w/]+)\s+"         # unit
        r"([\d,.]+)"              # value (may have commas)
    )
    for line in text.splitlines():
        m = pattern.search(line)
        if m:
            name = m.group(1)
            val_str = m.group(2).replace(",", "")
            try:
                metrics[name] = float(val_str)
            except ValueError:
                pass
    return metrics


def fmt_ncu_table(metrics: dict[str, float], kernel: str, config: str) -> str:
    """Format extracted NCU metrics as a markdown table row + explanation."""
    sol_compute = metrics.get(_NCU_SOL_COMPUTE, float("nan"))
    sol_memory = metrics.get(_NCU_SOL_MEMORY, float("nan"))
    tc_cycles = metrics.get(_NCU_TC_CYCLES, float("nan"))
    dram_bw_bs = metrics.get(_NCU_DRAM_BW, float("nan"))
    l2_hit = metrics.get(_NCU_L2_HIT, float("nan"))
    occupancy = metrics.get(_NCU_OCCUPANCY, float("nan"))

    dram_gbps = dram_bw_bs / 1e9 if dram_bw_bs == dram_bw_bs else float("nan")
    bound = "Compute" if sol_compute >= sol_memory else "Memory"

    lines = [
        f"## NCU hardware-counter metrics  ({kernel}, {config})",
        "",
        "| Metric | Value | Source |",
        "|---|---|---|",
        f"| SM compute throughput (SOL) | {sol_compute:.1f}% of peak | NCU `{_NCU_SOL_COMPUTE}` |",
        f"| Tensor-core pipe utilization | {tc_cycles:.1f}% of peak | NCU `{_NCU_TC_CYCLES}` |",
        f"| Memory throughput (SOL)      | {sol_memory:.1f}% of peak | NCU `{_NCU_SOL_MEMORY}` |",
        f"| DRAM bandwidth achieved      | {dram_gbps:.0f} GB/s | NCU `{_NCU_DRAM_BW}` |",
        f"| L2 hit rate                  | {l2_hit:.1f}% | NCU `{_NCU_L2_HIT}` |",
        f"| Warp occupancy               | {occupancy:.1f}% | NCU `{_NCU_OCCUPANCY}` |",
        f"| **Bottleneck**               | **{bound}** | max(compute SOL, memory SOL) |",
    ]
    return "\n".join(lines)


# ── CSV path: compute % of peak from bench CSV ─────────────────────────────────

def csv_to_pct_table(
    csv_path: Path,
    peak_tflops: float = DEFAULT_PEAK_TFLOPS,
    peak_hbm: float = DEFAULT_PEAK_HBM,
) -> str:
    """Compute % of hardware ceiling from the bench sweep CSV."""
    try:
        import pandas as pd
    except ImportError:
        return "ERROR: pandas required for --csv mode (pip install pandas)"

    df = pd.read_csv(csv_path)
    df = df[df["ok"].astype(str).str.lower().isin({"true", "1"})].copy()
    df["tflops"] = df["tflops_achieved"].astype(float)
    df["hbm"] = df["hbm_gb_s_achieved"].astype(float)
    df["pct_tc"] = (df["tflops"] / peak_tflops * 100).round(1)
    df["pct_hbm"] = (df["hbm"] / peak_hbm * 100).round(1)
    # Arithmetic intensity: TFLOP/s / (HBM GB/s) scaled to FLOP/byte
    df["ai"] = ((df["tflops"] * 1e12) / (df["hbm"] * 1e9)).round(0).astype(int)
    df["bound"] = df.apply(
        lambda r: "Compute" if r["pct_tc"] >= r["pct_hbm"] else "Memory", axis=1
    )

    lines = [
        f"## Benchmark-derived % of peak  (A100-SXM4-40GB: {peak_tflops} TFLOP/s bf16 TC, {peak_hbm} GB/s HBM)",
        "",
        "| kernel | dtype | S | D | B | AI (F/B) | TFLOP/s | % TC peak | HBM GB/s | % HBM peak | bound |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in df.sort_values(["kernel", "seqlen", "head_dim", "batch"]).iterrows():
        lines.append(
            f"| {r['kernel']} | {r['dtype']} | {r['seqlen']} | {r['head_dim']} | {r['batch']} "
            f"| {r['ai']:,} | {r['tflops']:.1f} | **{r['pct_tc']}%** "
            f"| {r['hbm']:.0f} | **{r['pct_hbm']}%** | {r['bound']} |"
        )
    lines += [
        "",
        f"> Peak figures: bf16 TC {peak_tflops} TFLOP/s, HBM {peak_hbm} GB/s. "
        f"Source: NVIDIA A100 Tensor Core GPU Architecture whitepaper.",
        "> Arithmetic intensity = achieved TFLOP/s / achieved HBM GB/s (FLOP/byte).",
        "> Note: % TC peak from FLOP counting; NCU `sm__pipe_tensor_cycles_active` gives the "
        "hardware-counter ground truth (see `bench/ncu_extract.py`).",
    ]
    return "\n".join(lines)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Extract NCU hardware-counter metrics")
    p.add_argument("--kernel", default="triton_prefill", choices=["triton_prefill", "torch_sdpa"])
    p.add_argument(
        "--profile-args",
        default="B=1,H=16,S=4096,D=128",
        help="Comma-separated KEY=VALUE pairs forwarded as PROF_* env vars",
    )
    p.add_argument("--no-sudo", action="store_true", help="Run NCU without sudo")
    p.add_argument("--ncu-txt", type=Path, default=None, help="Parse a saved NCU text file")
    p.add_argument(
        "--csv", type=Path, default=None,
        help="Compute %% of peak from a bench harness CSV (no NCU required)",
    )
    p.add_argument(
        "--peak-tflops", type=float, default=DEFAULT_PEAK_TFLOPS,
        help="GPU bf16 tensor-core peak TFLOP/s",
    )
    p.add_argument(
        "--peak-hbm", type=float, default=DEFAULT_PEAK_HBM,
        help="GPU peak HBM bandwidth GB/s",
    )
    args = p.parse_args(argv)

    if args.csv is not None:
        print(csv_to_pct_table(args.csv, args.peak_tflops, args.peak_hbm))
        return 0

    if args.ncu_txt is not None:
        text = args.ncu_txt.read_text()
    else:
        if not torch.cuda.is_available() if _has_torch() else True:
            print("ERROR: CUDA required for live NCU profiling", file=sys.stderr)
            return 2
        text = run_ncu(args.kernel, args.profile_args, sudo=not args.no_sudo)

    metrics = parse_ncu_text(text)
    if not metrics:
        print("WARNING: no metrics parsed from NCU output. Raw output:\n", file=sys.stderr)
        print(text[:4000], file=sys.stderr)
        return 3

    config = args.profile_args.replace(",", " ")
    print(fmt_ncu_table(metrics, args.kernel, config))
    return 0


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
