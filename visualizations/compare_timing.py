#!/usr/bin/env python3
"""Compare wall-clock time per length across several results files.

Usage:
    python visualizations/compare_timing.py results/v0.9.3_cpu_1_to_85.json \
        results/v0.10.0_gpu_1_to_85.json --out plots/timing_comparison.png

The first file is the baseline for the speed-up panel. Only the per-length
metadata is read (rulers are skipped), so even large files load in seconds.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ruler_data as rd  # noqa: E402
from generate_plots import apply_style, PALETTE, GRID  # noqa: E402


def main():
    """Plot clock time per length for each file and the speed-up over the first."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", help="results JSON files; the first is the baseline")
    p.add_argument("--out", default="plots/timing_comparison.png")
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()
    apply_style()

    runs = []
    for path in args.files:
        res = rd.load_results(path, cache_dir=None, with_rulers=False)
        rows = [r for r in res.sorted() if r.completed and r.clock_time > 0]
        label = os.path.splitext(os.path.basename(path))[0]
        runs.append((label, {r.length: r.clock_time for r in rows}))

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 2]})
    base_label, base = runs[0]
    for i, (label, times) in enumerate(runs):
        Ls = sorted(times)
        color = PALETTE[i % len(PALETTE)]
        ax.plot(Ls, [times[L] for L in Ls], "o-", ms=3, lw=1, color=color, label=label)
        if i:
            common = [L for L in Ls if L in base]
            ax2.plot(common, [base[L] / times[L] for L in common], "o-", ms=3, lw=1, color=color,
                     label=label)
    ax.set_yscale("log")
    ax.set_ylabel("wall-clock seconds")
    ax.set_title("Wall-clock time per length")
    ax.legend(fontsize=9)
    ax2.axhline(1, color=GRID, lw=1)
    ax2.set_yscale("log")
    ax2.set_ylabel(f"speed-up vs\n{base_label}")
    ax2.set_xlabel("ruler length L")
    if len(runs) > 1:
        ax2.legend(fontsize=9)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
