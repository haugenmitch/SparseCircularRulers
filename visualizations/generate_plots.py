#!/usr/bin/env python3
"""Generate the sparse circular ruler figures described in README.md.

Usage:
    python visualizations/generate_plots.py results/v0.10.2_gpu.json
    python visualizations/generate_plots.py results/v0.10.2_gpu.json --only relations boundaries
    python visualizations/generate_plots.py results/v0.10.2_gpu.json --focus 57 100 --jobs 8

The per-length analysis runs in a process pool (large lengths are split into
chunks), then every figure is drawn in the same pool. See README.md for what
each figure shows and why it exists.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm, ListedColormap  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ruler_data as rd  # noqa: E402

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

BG = "#404040"
FG = "#e0e0e0"
GRID = "#606060"
RED, GREEN, BLUE, YELLOW, PURPLE, TEAL, ORANGE = (
    "#f08080", "#7fe97f", "#7cb2f0", "#f0e68c", "#c792ea", "#80cbc4", "#ffb74d")
PALETTE = [RED, GREEN, BLUE, YELLOW, PURPLE, TEAL, ORANGE]

# Same values as ~/repos/StyleGuide/mch.mplstyle, used when that file is not found.
MCH_STYLE = {
    "figure.facecolor": BG, "figure.edgecolor": BG, "figure.titlesize": 18,
    "figure.titleweight": "bold", "axes.facecolor": BG, "axes.edgecolor": FG,
    "axes.labelcolor": FG, "axes.labelsize": 12, "axes.linewidth": 1.5,
    "axes.grid": True, "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.titlepad": 12, "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": FG, "ytick.color": FG, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "grid.color": GRID, "grid.linestyle": "--", "grid.linewidth": 0.8, "grid.alpha": 0.5,
    "font.family": "sans-serif",
    "font.sans-serif": ["JuliaMono", "DejaVu Sans", "Arial", "sans-serif"],
    "font.size": 11, "text.color": FG, "lines.linewidth": 2, "lines.markersize": 8,
    "legend.facecolor": "#333333", "legend.edgecolor": GRID, "legend.fontsize": 10,
    "legend.framealpha": 0.8, "savefig.facecolor": BG, "savefig.edgecolor": BG,
    "savefig.bbox": "tight",
}


def apply_style() -> None:
    """Use the mch matplotlib style, or the built-in copy of it."""
    home = os.path.expanduser("~")
    candidates = [
        os.environ.get("MCH_STYLE", ""),
        os.path.join(home, "repos/StyleGuide/mch.mplstyle"),
        os.path.join(home, "repos/plotting_style/mch.mplstyle"),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            plt.style.use(path)
            return
    try:
        plt.style.use("mch")
    except OSError:
        plt.rcParams.update(MCH_STYLE)


def heat_cmap(name: str = "magma") -> ListedColormap:
    """Colormap whose masked (zero) cells show the axes background."""
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad(alpha=0)
    return cmap


def k_color(k: int):
    """Distinct, ordered colour per segment count (visible on the grey background)."""
    return plt.get_cmap("rainbow")(0.05 + 0.95 * min(k - 1, 14) / 14)


def stacked(heights, figsize):
    """Vertically stacked panels sharing x, with a colorbar column beside each."""
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(len(heights), 2, height_ratios=heights, width_ratios=[1, 0.018],
                          wspace=0.02, hspace=0.12)
    axes, caxes = [], []
    for i in range(len(heights)):
        axes.append(fig.add_subplot(gs[i, 0], sharex=axes[0] if axes else None))
        caxes.append(fig.add_subplot(gs[i, 1]))
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    return fig, axes, caxes


def fmt_ruler(row) -> str:
    """A ruler as text, like [1, 2, 3]."""
    return "[" + ", ".join(str(int(v)) for v in row) + "]"


# ---------------------------------------------------------------------------
# Per-length analysis (runs in worker processes)
# ---------------------------------------------------------------------------

RESULTS: rd.Results | None = None   # set before the pool forks
SUMMARY: "Summary | None" = None    # set before figure tasks fork
ARGS: argparse.Namespace | None = None

REL_BINS = np.linspace(0, 4, 41)  # segment value relative to the mean segment L/k


@dataclass
class LengthStats:
    """Additive statistics for one length (chunks merge by summing)."""
    length: int
    k: int
    n: int = 0
    seg_hist: np.ndarray = None      # count of segments with value v
    ones_hist: np.ndarray = None     # rulers with c unit segments
    adjacency: np.ndarray = None     # cyclic neighbour pairs (both orders)
    mult_sum: np.ndarray = None      # sum over rulers of arcs measuring d
    largest_min: int = 10**9
    largest_max: int = 0
    used: int = 0                    # rulers included in the usage statistics
    inv_sum: np.ndarray = None       # involvement fraction summed per relative-size bin
    ess_sum: np.ndarray = None
    rel_count: np.ndarray = None
    forced_min_hist: np.ndarray = None  # rulers whose least-crossed boundary is forced c times
    forced_hist: np.ndarray = None      # boundaries forced c times

    def merge(self, other: "LengthStats") -> None:
        """Add another chunk's statistics to these."""
        self.n += other.n
        self.used += other.used
        self.largest_min = min(self.largest_min, other.largest_min)
        self.largest_max = max(self.largest_max, other.largest_max)
        for name in ("seg_hist", "ones_hist", "adjacency", "mult_sum", "inv_sum",
                     "ess_sum", "rel_count", "forced_min_hist", "forced_hist"):
            mine, theirs = getattr(self, name), getattr(other, name)
            setattr(self, name, theirs if mine is None else mine + theirs)


def chunk_stats(task: tuple[int, int, int]) -> LengthStats:
    """Statistics for rulers start:stop of one length."""
    L, start, stop = task
    res = RESULTS.lengths[L]
    rulers = res.rulers[start:stop]
    k = res.num_segments
    st = LengthStats(length=L, k=k, n=rulers.shape[0])
    vmax = L + 1
    st.seg_hist = np.bincount(rulers.ravel(), minlength=vmax)[:vmax]
    st.ones_hist = np.bincount((rulers == 1).sum(axis=1), minlength=k + 1)
    a = rulers.astype(np.int64)
    b = np.roll(rulers, -1, axis=1).astype(np.int64)
    pairs = np.concatenate([(a * vmax + b).ravel(), (b * vmax + a).ravel()]) if k > 1 else np.zeros(0, np.int64)
    st.adjacency = np.bincount(pairs, minlength=vmax * vmax).reshape(vmax, vmax)
    st.mult_sum = rd.distance_multiplicity(rulers, L).sum(axis=0).astype(np.int64)
    largest = rulers.max(axis=1)
    st.largest_min, st.largest_max = int(largest.min()), int(largest.max())

    # Usage statistics, optionally on an evenly spaced sample of the chunk.
    if not ARGS.need_usage:
        return st
    limit = ARGS.max_rulers
    if limit:
        share = math.ceil(limit * rulers.shape[0] / res.count)
        rulers = rd.sample_rows(rulers, share)
    st.used = rulers.shape[0]
    nb = len(REL_BINS) - 1
    st.inv_sum = np.zeros(nb)
    st.ess_sum = np.zeros(nb)
    st.rel_count = np.zeros(nb)
    st.forced_min_hist = np.zeros(L + 1, np.int64)
    st.forced_hist = np.zeros(L + 1, np.int64)
    if k >= 2:
        u = rd.usage(rulers, L)
        half = max(L // 2, 1)
        rel = rulers * k / L
        bins = np.clip(np.digitize(rel, REL_BINS) - 1, 0, nb - 1).ravel()
        st.inv_sum = np.bincount(bins, weights=(u["involved"] / half).ravel(), minlength=nb)
        st.ess_sum = np.bincount(bins, weights=(u["essential"] / half).ravel(), minlength=nb)
        st.rel_count = np.bincount(bins, minlength=nb).astype(float)
        st.forced_min_hist = np.bincount(u["forced"].min(axis=1), minlength=L + 1)[:L + 1]
        st.forced_hist = np.bincount(u["forced"].ravel(), minlength=L + 1)[:L + 1]
    return st


def link_task(L: int) -> tuple[int, dict, int]:
    """Edges from L-1 to L, plus the number of distinct segment multisets at L."""
    cur = RESULTS.lengths[L]
    prev = RESULTS.lengths.get(L - 1)
    links = rd.link_lengths(prev.rulers, cur.rulers) if prev is not None else {}
    multisets = int(np.unique(rd.row_keys(np.sort(cur.rulers, axis=1))).size) if cur.count else 0
    return L, links, multisets


@dataclass
class Summary:
    """Every length's merged statistics, links, and counts."""
    lengths: list[int]
    stats: dict[int, LengthStats]
    links: dict[int, dict]
    multisets: dict[int, int]
    k: dict[int, int] = field(default_factory=dict)
    counts: dict[int, int] = field(default_factory=dict)


def analyse(results: rd.Results, lengths: list[int], pool, chunk: int) -> Summary:
    """Compute statistics and one-edit links for `lengths` in the pool."""
    tasks = []
    for L in lengths:
        n = results.lengths[L].count
        tasks += [(L, s, min(s + chunk, n)) for s in range(0, n, chunk)]
    # Largest work first keeps the pool busy until the end.
    tasks.sort(key=lambda t: -(t[2] - t[1]) * results.lengths[t[0]].num_segments ** 3)
    stats: dict[int, LengthStats] = {}
    link_lengths = sorted(lengths, key=lambda L: -results.lengths[L].count) if ARGS.need_links else []
    link_jobs = pool.imap_unordered(link_task, link_lengths)
    for st in pool.imap_unordered(chunk_stats, tasks):
        if st.length in stats:
            stats[st.length].merge(st)
        else:
            stats[st.length] = st
    links, multisets = {}, {}
    for L, lk, ms in link_jobs:
        links[L] = lk
        multisets[L] = ms
    summary = Summary(lengths=lengths, stats=stats, links=links, multisets=multisets)
    for L in lengths:
        summary.k[L] = results.lengths[L].num_segments
        summary.counts[L] = results.lengths[L].count
    return summary


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def save(fig, name: str) -> str:
    """Save a figure to the output directory and close it."""
    path = os.path.join(ARGS.outdir, name)
    fig.savefig(path, dpi=ARGS.dpi)
    plt.close(fig)
    return path


def drops(S: Summary) -> list[int]:
    """Lengths whose minimal segment count is below the previous length's."""
    return [L for L in S.lengths[1:] if L - 1 in S.k and S.k[L] < S.k[L - 1]]


def isolated(S: Summary) -> list[int]:
    """Lengths with no one-edit link to L-1 or L+1 (both neighbours analysed)."""
    out = []
    for L in S.lengths:
        if L - 1 not in S.k or L + 1 not in S.k:
            continue
        n_in = sum(len(v) for v in S.links.get(L, {}).values())
        n_out = sum(len(v) for v in S.links.get(L + 1, {}).values())
        if n_in == 0 and n_out == 0:
            out.append(L)
    return out


def fig_segments_vs_length():
    """Minimal k against the lower bound, and slack."""
    S = SUMMARY
    Ls = np.array(S.lengths)
    ks = np.array([S.k[L] for L in Ls])
    lb = np.array([RESULTS.lengths[L].lower_bound for L in Ls])
    slack = ks * (ks - 1) - (Ls - 1)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 2]})
    ax.step(Ls, lb, where="mid", color=YELLOW, ls="--", lw=1.5,
            label="lower bound: smallest k with k(k-1)+1 ≥ L")
    ax.plot(Ls, ks, "o", color=BLUE, ms=5, label="minimal k found")
    d = drops(S)
    ax.plot(d, [S.k[L] for L in d], "o", mfc="none", mec=RED, ms=12, mew=2,
            label="k drops below k(L-1)")
    for L in d:
        ax.annotate(str(L), (L, S.k[L]), textcoords="offset points", xytext=(0, -18),
                    ha="center", fontsize=8, color=RED)
    ax.set_ylabel("segments k")
    ax.set_title("Minimal segment count per length")
    ax.legend(loc="upper left")
    ax2.bar(Ls, slack, color=[k_color(k) for k in ks], width=0.85)
    ax2.set_ylabel("slack = k(k-1) - (L-1)")
    ax2.set_xlabel("ruler length L")
    ax2.set_title("Spare arcs: arcs available beyond the L-1 distances needed", fontsize=12)
    return save(fig, "segments_vs_length.png")


def fig_solutions_vs_length():
    """Solutions per length, joined along each k plateau."""
    S = SUMMARY
    fig, ax = plt.subplots(figsize=(12, 6.5))
    by_k = defaultdict(list)
    for L in S.lengths:
        by_k[S.k[L]].append(L)
    for k, Ls in sorted(by_k.items()):
        # connect consecutive lengths of the same k plateau
        runs, run = [], [Ls[0]]
        for L in Ls[1:]:
            if L == run[-1] + 1:
                run.append(L)
            else:
                runs.append(run)
                run = [L]
        runs.append(run)
        for i, run in enumerate(runs):
            ax.plot(run, [S.counts[L] for L in run], "-o", color=k_color(k), ms=4, lw=1.2,
                    label=f"k = {k}" if i == 0 else None)
    iso = isolated(S)
    ax.plot(iso, [S.counts[L] for L in iso], "o", mfc="none", mec=FG, ms=11, mew=1.5,
            label="no one-edit link to L±1")
    ax.set_yscale("log")
    ax.set_xlabel("ruler length L")
    ax.set_ylabel("distinct solutions (up to rotation/reflection)")
    ax.set_title("Number of minimal solutions per length")
    ax.legend(ncol=3, fontsize=8, loc="upper left")
    return save(fig, "solutions_vs_length.png")


def pair_slack(k: int, L: int) -> int:
    """Spare complementary arc pairs: k(k-1)/2 pairs must cover floor(L/2) distance classes."""
    return k * (k - 1) // 2 - L // 2


def fig_solutions_vs_slack():
    """Solutions against spare arc pairs, by k and parity."""
    S = SUMMARY
    fig, ax = plt.subplots(figsize=(11, 6.5))
    by_k = defaultdict(list)
    for L in S.lengths:
        by_k[S.k[L]].append(L)
    for k, Ls in sorted(by_k.items()):
        if k < 3:
            continue
        color = k_color(k)
        groups = defaultdict(list)
        for L in Ls:
            groups[pair_slack(k, L)].append(S.counts[L])
        xs = np.array(sorted(groups))
        ys = np.array([np.exp(np.mean(np.log(groups[x]))) for x in xs])
        ax.plot(xs, ys, "-", color=color, lw=1.5, label=f"k = {k}")
        even = [L for L in Ls if L % 2 == 0]
        odd = [L for L in Ls if L % 2 == 1]
        ax.plot([pair_slack(k, L) for L in even], [S.counts[L] for L in even], "o", color=color, ms=6)
        ax.plot([pair_slack(k, L) for L in odd], [S.counts[L] for L in odd], "o", mfc="none",
                mec=color, mew=1.5, ms=9)
        ax.annotate(f"k={k}", (xs[-1], ys[-1]), textcoords="offset points", xytext=(6, 0),
                    fontsize=8, color=color, va="center")
    ax.plot([], [], "o", color=FG, ms=6, label="even L")
    ax.plot([], [], "o", mfc="none", mec=FG, mew=1.5, ms=9, label="odd L")
    ax.set_yscale("log")
    ax.set_xlabel("spare arc pairs = k(k-1)/2 - floor(L/2)   (0 = perfect difference set)")
    ax.set_ylabel("distinct solutions")
    ax.set_title("Solutions grow with spare arc pairs (L = 2m and 2m+1 share an x value)")
    ax.legend(ncol=2, fontsize=8, loc="lower right")
    return save(fig, "solutions_vs_slack.png")


def fig_count_pairs():
    """Count ratios of neighbouring lengths, and distance classes against arc pairs."""
    S = SUMMARY
    Ls = [L for L in S.lengths if L - 1 in S.counts and L > 2]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 9.5), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 2]})
    meds = {}
    for parity, color, label in (
            (1, GREEN, "odd L = 2m+1 vs 2m: same number of distance classes"),
            (0, RED, "even L = 2m vs 2m-1: one more distance class")):
        same = [L for L in Ls if L % 2 == parity and S.k[L] == S.k[L - 1]]
        r = np.array([S.counts[L] / S.counts[L - 1] for L in same])
        meds[parity] = 10 ** np.median(np.abs(np.log10(r)))
        ax.plot(same, r, "o", color=color, ms=6, label=label)
    changed = [L for L in Ls if S.k[L] != S.k[L - 1]]
    ax.plot(changed, [S.counts[L] / S.counts[L - 1] for L in changed], "x", color=FG, ms=7, mew=1.5,
            label="k changes between L-1 and L")
    ax.axhline(1, color=FG, lw=1)
    ax.set_yscale("log")
    ax.set_ylabel("N(L) / N(L-1)")
    ax.set_title("Solution counts come in pairs (2m, 2m+1)")
    ax.legend(loc="lower left", fontsize=9)
    ax2.text(0.99, 0.05, f"typical factor between neighbours (same k):\n"
            f"odd L: {meds[1]:.2f}x    even L: {meds[0]:.2f}x",
            transform=ax2.transAxes, ha="right", va="bottom", fontsize=10,
            bbox=dict(facecolor="#333333", edgecolor=GRID, alpha=0.9))
    allL = np.array(S.lengths)
    ks = np.array([S.k[L] for L in allL])
    classes = allL // 2
    pairs = ks * (ks - 1) // 2
    ax2.fill_between(allL, classes, pairs, step="mid", color=BLUE, alpha=0.3,
                     label="spare arc pairs")
    ax2.plot(allL, pairs, drawstyle="steps-mid", color=BLUE, lw=1.5,
             label="complementary arc pairs available, k(k-1)/2")
    ax2.plot(allL, classes, drawstyle="steps-mid", color=YELLOW, lw=1.5,
             label="distance classes to cover, floor(L/2)")
    ax2.set_ylabel("count")
    ax2.set_xlabel("ruler length L")
    ax2.set_title("Distance classes grow only at even L", fontsize=12)
    ax2.legend(loc="upper left", fontsize=9)
    return save(fig, "solution_count_pairs.png")


def fig_segment_value_range():
    """Range of the largest segment per length."""
    S = SUMMARY
    Ls = np.array(S.lengths)
    lo = np.array([S.stats[L].largest_min for L in Ls])
    hi = np.array([S.stats[L].largest_max for L in Ls])
    mean = Ls / np.array([S.k[L] for L in Ls])
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.fill_between(Ls, lo, hi, color=BLUE, alpha=0.35, step="mid",
                    label="largest segment: range over solutions")
    ax.plot(Ls, hi, color=BLUE, lw=1, drawstyle="steps-mid")
    ax.plot(Ls, lo, color=BLUE, lw=1, drawstyle="steps-mid")
    ax.plot(Ls, mean, color=YELLOW, lw=2, label="mean segment L/k")
    ax.plot(Ls, Ls / 2, color=GRID, lw=1, ls=":", label="L/2")
    ax.set_xlabel("ruler length L")
    ax.set_ylabel("segment value")
    ax.set_title("How large the largest segment gets")
    ax.legend(loc="upper left")
    return save(fig, "segment_value_range.png")


def fig_segment_frequency_heatmap():
    """Segments of each value per solution, per length."""
    S = SUMMARY
    Ls = S.lengths
    vmax = max(int(np.nonzero(S.stats[L].seg_hist)[0].max()) for L in Ls)
    grid = np.zeros((vmax + 1, len(Ls)))
    for j, L in enumerate(Ls):
        h = S.stats[L].seg_hist[: vmax + 1]
        grid[: len(h), j] = h / S.stats[L].n
    grid = np.ma.masked_less_equal(grid, 0)
    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap=heat_cmap("viridis"),
                   norm=LogNorm(vmin=1e-4, vmax=grid.max()),
                   extent=(Ls[0] - 0.5, Ls[-1] + 0.5, -0.5, vmax + 0.5), interpolation="nearest")
    ax.plot(Ls, [L / S.k[L] for L in Ls], color=RED, lw=1.2, label="mean segment L/k")
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("segments of this value per solution")
    ax.set_xlabel("ruler length L")
    ax.set_ylabel("segment value")
    ax.set_title("Segment value frequency")
    ax.grid(False)
    ax.legend(loc="upper left")
    return save(fig, "segment_frequency_heatmap.png")


def fig_segment_structure():
    """Unit segments per solution, and solutions per segment multiset."""
    S = SUMMARY
    Ls = S.lengths
    kmax = max(S.k.values())
    grid = np.zeros((kmax + 1, len(Ls)))
    for j, L in enumerate(Ls):
        h = S.stats[L].ones_hist
        grid[: len(h), j] = h / S.stats[L].n
    fig, (ax, ax2), (cax, cax2) = stacked([3, 2], (12, 9))
    top = int(np.nonzero(grid.sum(axis=1))[0].max())
    im = ax.imshow(np.ma.masked_less_equal(grid[: top + 1], 0), origin="lower", aspect="auto",
                   cmap=heat_cmap("magma"), norm=LogNorm(vmin=1e-5, vmax=1),
                   extent=(Ls[0] - 0.5, Ls[-1] + 0.5, -0.5, top + 0.5), interpolation="nearest")
    fig.colorbar(im, cax=cax).set_label("fraction of solutions")
    ax.set_ylabel("unit segments per solution")
    ax.set_title("Unit segments: every minimal ruler has at least one, few have many")
    ax.grid(False)
    ratio = [S.counts[L] / max(S.multisets[L], 1) for L in Ls]
    ax2.plot(Ls, ratio, "o-", color=GREEN, ms=4, lw=1)
    ax2.set_ylabel("solutions per distinct\nsegment multiset")
    ax2.set_xlabel("ruler length L")
    ax2.set_title("Reorderings of the same segment values that are also solutions", fontsize=12)
    cax2.axis("off")
    return save(fig, "segment_structure.png")


def adjacency_matrix(L: int) -> np.ndarray:
    """Neighbour pair counts at L, for values 1 up to the largest segment."""
    adj = SUMMARY.stats[L].adjacency
    vmax = int(np.nonzero(SUMMARY.stats[L].seg_hist)[0].max())
    return adj[1: vmax + 1, 1: vmax + 1]


def fig_adjacency_by_length():
    """Neighbour pair matrices for a spread of lengths."""
    S = SUMMARY
    picks = ARGS.adjacency_lengths or pick_lengths(S, 12)
    picks = [L for L in picks if L in S.stats and S.k[L] > 1]
    cols = 4
    rows = math.ceil(len(picks) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.0 * rows), squeeze=False)
    for ax, L in zip(axes.flat, picks):
        m = adjacency_matrix(L).astype(float)
        frac = np.ma.masked_less_equal(m / m.sum(), 0)
        n = m.shape[0]
        ax.imshow(frac, origin="lower", cmap=heat_cmap("magma"),
                  norm=LogNorm(vmin=max(frac.min(), 1e-6), vmax=frac.max()),
                  extent=(0.5, n + 0.5, 0.5, n + 0.5), interpolation="nearest")
        ax.plot([0.5, L / 2 - 0.5], [L / 2 - 0.5, 0.5], color=TEAL, lw=0.8, ls="--")
        ax.set_xlim(0.5, n + 0.5)
        ax.set_ylim(0.5, n + 0.5)
        ax.set_title(f"L = {L}  (k = {S.k[L]}, {S.counts[L]:,} sol.)", fontsize=10)
        ax.grid(False)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(labelsize=8)
    for ax in list(axes.flat)[len(picks):]:
        ax.axis("off")
    fig.supxlabel("neighbour segment value")
    fig.supylabel("segment value")
    fig.suptitle("Which segment values sit next to each other (cyclic, both directions, log colour;\n"
                 "dashed: neighbour pairs summing to L/2)")
    fig.tight_layout()
    return save(fig, "adjacency_by_length.png")


def fig_adjacency_focus(L: int):
    """Neighbour pairs and conditional neighbours at one length."""
    S = SUMMARY
    m = adjacency_matrix(L).astype(float)
    n = m.shape[0]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6.8))
    frac = np.ma.masked_less_equal(m / m.sum(), 0)
    im = a1.imshow(frac, origin="lower", cmap=heat_cmap("magma"),
                   norm=LogNorm(vmin=frac.min(), vmax=frac.max()),
                   extent=(0.5, n + 0.5, 0.5, n + 0.5), interpolation="nearest")
    fig.colorbar(im, ax=a1, pad=0.01).set_label("share of all neighbour pairs")
    a1.set_title("Neighbour pairs")
    rows = m.sum(axis=1, keepdims=True)
    cond = np.ma.masked_less_equal(np.divide(m, rows, out=np.zeros_like(m), where=rows > 0), 0)
    im2 = a2.imshow(cond, origin="lower", cmap=heat_cmap("viridis"), vmin=0, vmax=cond.max(),
                    extent=(0.5, n + 0.5, 0.5, n + 0.5), interpolation="nearest")
    fig.colorbar(im2, ax=a2, pad=0.01).set_label("P(neighbour | segment)")
    a2.set_title("Neighbours of each segment value")
    for ax in (a1, a2):
        ax.plot([0.5, L / 2 - 0.5], [L / 2 - 0.5, 0.5], color=TEAL, lw=1, ls="--",
                label="a + b = L/2")
        ax.set_xlim(0.5, n + 0.5)
        ax.set_ylim(0.5, n + 0.5)
        ax.legend(loc="upper right", fontsize=9)
        ax.set_xlabel("neighbour segment value")
        ax.set_ylabel("segment value")
        ax.grid(False)
    fig.suptitle(f"Segment adjacency at L = {L} (k = {S.k[L]}, {S.counts[L]:,} solutions)")
    fig.tight_layout()
    return save(fig, f"adjacency_L{L}.png")


def pick_lengths(S: Summary, count: int) -> list[int]:
    """About `count` lengths (k >= 4) spread evenly over the range."""
    usable = [L for L in S.lengths if S.k[L] >= 4]
    idx = np.linspace(0, len(usable) - 1, count).round().astype(int)
    return sorted({usable[i] for i in idx})


def fig_relations_by_length():
    """Share of solutions with a one-edit parent or child."""
    S = SUMMARY
    Ls = [L for L in S.lengths if L > 1]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 8.5), sharex=True)
    for name, color, label in (("extend", BLUE, "extend: one segment +1 (same k)"),
                               ("split", GREEN, "split: s → a, b with a+b = s+1 (k+1)")):
        frac_in, frac_out = [], []
        for L in Ls:
            e = S.links.get(L, {}).get(name, np.zeros((0, 2)))
            frac_in.append(len(np.unique(e[:, 1])) / S.counts[L] if len(e) else np.nan)
            nxt = S.links.get(L + 1, {}).get(name, np.zeros((0, 2)))
            frac_out.append(len(np.unique(nxt[:, 0])) / S.counts[L] if len(nxt) else np.nan)
        ax.plot(Ls, frac_in, "o", color=color, ms=5, label=label)
        ax2.plot(Ls, frac_out, "o", color=color, ms=5, label=label)
    for L in isolated(S):
        for a in (ax, ax2):
            a.axvline(L, color=RED, lw=1, alpha=0.6)
    ax.axvline(np.nan, color=RED, lw=1, label="isolated length (no link either way)")
    ax.set_yscale("log")
    ax2.set_yscale("log")
    ax.set_ylabel("share of solutions at L\nwith a parent at L-1")
    ax2.set_ylabel("share of solutions at L\nwith a child at L+1")
    ax2.set_xlabel("ruler length L")
    ax.set_title("How often a solution is one edit away from a solution one unit shorter / longer")
    ax.legend(loc="lower left", fontsize=9)
    return save(fig, "relations_by_length.png")


def families(S: Summary):
    """Connected components of the one-edit graph across all lengths."""
    offsets, total = {}, 0
    for L in S.lengths:
        offsets[L] = total
        total += S.counts[L]
    parent = np.arange(total)

    def find(x):
        """Union-find root of x, compressing the path."""
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for L, lk in S.links.items():
        if L - 1 not in offsets:
            continue
        for pairs in lk.values():
            for a, b in pairs:
                ra, rb = find(offsets[L - 1] + int(a)), find(offsets[L] + int(b))
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    linked = set()
    for L, lk in S.links.items():
        for pairs in lk.values():
            if len(pairs):
                linked.update((offsets[L - 1] + pairs[:, 0]).tolist())
                linked.update((offsets[L] + pairs[:, 1]).tolist())
    length_of = np.repeat(np.array(S.lengths), [S.counts[L] for L in S.lengths])
    comps = defaultdict(list)
    for node in linked:
        comps[find(node)].append(node)
    fams = []
    for root, nodes in comps.items():
        ls = length_of[nodes]
        fams.append({"root": root, "lmin": int(ls.min()), "lmax": int(ls.max()), "size": len(nodes),
                     "per_length": np.bincount(ls - ls.min())})
    singletons = total - len(linked)
    return fams, singletons, offsets


def fig_solution_families():
    """Family spans and the 30 longest families."""
    S = SUMMARY
    fams, singletons, offsets = families(S)
    spans = np.array([f["lmax"] - f["lmin"] + 1 for f in fams])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 7.5), gridspec_kw={"width_ratios": [1, 2]})
    counts = np.bincount(spans, minlength=2)
    counts[1] = singletons
    xs = np.nonzero(counts)[0]
    ax.bar(xs, counts[xs], color=BLUE)
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("lengths spanned by the family")
    ax.set_ylabel("families")
    ax.set_title("Family span (1 = no one-edit relative)", fontsize=12)
    top = sorted(fams, key=lambda f: (-(f["lmax"] - f["lmin"]), -f["size"]))[:30]
    for y, f in enumerate(top):
        Ls = np.arange(f["lmin"], f["lmax"] + 1)
        sizes = f["per_length"]
        ax2.plot([Ls[0], Ls[-1]], [y, y], color=GRID, lw=1, zorder=1)
        ax2.scatter(Ls, np.full(len(Ls), y), s=12 + 30 * np.log10(np.maximum(sizes, 1)),
                    c=[k_color(S.k[L]) for L in Ls], zorder=2, edgecolors="none")
        ax2.annotate(f"{f['size']}", (Ls[-1], y), textcoords="offset points", xytext=(6, 0),
                     va="center", fontsize=8)
    ax2.invert_yaxis()
    ax2.set_yticks([])
    ax2.set_xlabel("ruler length L (dot size ~ log members at L, colour = k)")
    ax2.set_title("30 longest families (number = total members)", fontsize=12)
    fig.suptitle("Families of solutions linked by one-unit edits across lengths")
    fig.tight_layout()
    return save(fig, "solution_families.png")


def fig_lineage_small():
    """Every small solution and its one-edit links."""
    S = SUMMARY
    maxL = ARGS.lineage_max
    Ls = [L for L in S.lengths if L <= maxL]
    ypos: dict[int, np.ndarray] = {}
    fig, ax = plt.subplots(figsize=(16, 9))
    for L in Ls:
        n = S.counts[L]
        lk = S.links.get(L, {})
        key = np.full(n, np.inf)
        if L - 1 in ypos:
            sums, cnt = np.zeros(n), np.zeros(n)
            for pairs in lk.values():
                if len(pairs):
                    np.add.at(sums, pairs[:, 1], ypos[L - 1][pairs[:, 0]])
                    np.add.at(cnt, pairs[:, 1], 1)
            key = np.where(cnt > 0, sums / np.maximum(cnt, 1), np.inf)
        order = np.argsort(key, kind="stable")
        y = np.empty(n)
        y[order] = (np.arange(n) + 0.5) / n
        ypos[L] = y
    for L in Ls:
        if L - 1 not in ypos:
            continue
        for name, color in (("extend", BLUE), ("split", GREEN)):
            pairs = S.links.get(L, {}).get(name, np.zeros((0, 2), int))
            for a, b in pairs:
                ax.plot([L - 1, L], [ypos[L - 1][a], ypos[L][b]], color=color, lw=0.5, alpha=0.5)
    for L in Ls:
        ax.scatter(np.full(S.counts[L], L), ypos[L], s=8 if S.counts[L] > 50 else 25,
                   color=k_color(S.k[L]), zorder=3, edgecolors="none")
    ax.plot([], [], color=BLUE, label="extend (same k)")
    ax.plot([], [], color=GREEN, label="split (k+1)")
    ax.set_yticks([])
    ax.set_xticks(Ls)
    ax.set_xlabel("ruler length L (dot colour = k; each column is spread over the full height)")
    ax.set_title(f"Every solution up to L = {maxL} and its one-edit links")
    ax.legend(loc="upper left")
    ax.grid(False)
    return save(fig, "lineage_small.png")


def example_ruler(L: int) -> np.ndarray:
    """The first (smallest) solution at L."""
    return RESULTS.lengths[L].rulers[0]


def fig_involvement_focus(L: int):
    """Segment involvement for one solution and for all solutions at L."""
    res = RESULTS.lengths[L]
    k = res.num_segments
    ruler = example_ruler(L)
    half = L // 2
    sums = rd.arc_sums(ruler[None])[0].ravel()
    seg, _ = rd.arc_index(k)
    status = np.zeros((half, k))  # 0 unused, 1 optional, 2 essential
    for d in range(1, half + 1):
        arcs = np.nonzero(sums == d)[0]
        member = seg[arcs]
        status[d - 1][member.any(axis=0)] = 1
        status[d - 1][member.all(axis=0)] = 2
    rulers = rd.sample_rows(res.rulers, ARGS.max_rulers or None)
    u = rd.usage(rulers, L)
    fig = plt.figure(figsize=(17, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 3], width_ratios=[1.1, 1], hspace=0.08, wspace=0.18)
    a_bar = fig.add_subplot(gs[0, 0])
    a_mat = fig.add_subplot(gs[1, 0], sharex=a_bar)
    a_all = fig.add_subplot(gs[:, 1])
    cmap = ListedColormap([BG, BLUE, GREEN])
    a_mat.imshow(status, origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=2,
                 extent=(-0.5, k - 0.5, 0.5, half + 0.5), interpolation="nearest")
    a_mat.set_xticks(range(k))
    a_mat.set_xticklabels([f"s{i}={int(v)}" for i, v in enumerate(ruler)], rotation=60, fontsize=8)
    a_mat.set_ylabel("distance d (≤ L/2)")
    a_mat.grid(False)
    inv = (status > 0).sum(axis=0)
    ess = (status == 2).sum(axis=0)
    a_bar.bar(range(k), inv, color=BLUE, label="involved (some arc for d uses it)")
    a_bar.bar(range(k), ess, color=GREEN, label="essential (every arc for d uses it)")
    a_bar.set_ylabel("distances")
    a_bar.tick_params(labelbottom=False)
    a_bar.legend(fontsize=8, loc="upper right")
    a_bar.set_title(f"Example {fmt_ruler(ruler)}", fontsize=11)
    vals = rulers.ravel()
    invf = u["involved"].ravel()
    vmax = int(vals.max())
    H, xe, ye = np.histogram2d(vals, invf, bins=[np.arange(0.5, vmax + 1.5), np.arange(-0.5, half + 1.5)])
    im = a_all.imshow(np.ma.masked_equal(H.T, 0), origin="lower", aspect="auto",
                      cmap=heat_cmap("magma"), norm=LogNorm(),
                      extent=(xe[0], xe[-1], ye[0], ye[-1]), interpolation="nearest")
    means = np.bincount(vals, weights=invf, minlength=vmax + 1) / np.maximum(np.bincount(vals, minlength=vmax + 1), 1)
    present = np.bincount(vals, minlength=vmax + 1) > 0
    a_all.plot(np.nonzero(present)[0], means[present], color=GREEN, lw=1.5, label="mean")
    fig.colorbar(im, ax=a_all, pad=0.01).set_label("segments")
    a_all.set_xlabel("segment value")
    a_all.set_ylabel("distances d ≤ L/2 the segment is involved in")
    sampled = "" if rulers.shape[0] == res.count else f" (sample of {rulers.shape[0]:,})"
    a_all.set_title(f"All {res.count:,} solutions{sampled}", fontsize=11)
    a_all.legend(loc="upper right")
    a_all.grid(False)
    fig.suptitle(f"Which segments measure each distance, L = {L} (k = {k})")
    return save(fig, f"segment_involvement_L{L}.png")


def fig_involvement_vs_length():
    """Involvement against relative segment size, grouped by k."""
    S = SUMMARY
    centers = (REL_BINS[:-1] + REL_BINS[1:]) / 2
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), sharex=True)
    by_k = defaultdict(lambda: [np.zeros(len(centers)) for _ in range(3)])
    for L in S.lengths:
        st = S.stats[L]
        if st.k < 6:
            continue
        acc = by_k[st.k]
        acc[0] += st.inv_sum
        acc[1] += st.ess_sum
        acc[2] += st.rel_count
    for k, (inv, ess, cnt) in sorted(by_k.items()):
        ok = cnt >= 20
        ax.plot(centers[ok], inv[ok] / cnt[ok], "-", color=k_color(k), lw=1.5, label=f"k = {k}")
        ax2.plot(centers[ok], ess[ok] / cnt[ok], "-", color=k_color(k), lw=1.5, label=f"k = {k}")
    for a in (ax, ax2):
        a.set_xlabel("segment value relative to the mean segment (s·k/L)")
        a.axvline(1, color=GRID, lw=1)
    ax.set_ylabel("share of distances d ≤ L/2")
    ax.set_title("Involved: some arc measuring d uses the segment", fontsize=12)
    ax2.set_title("Essential: every arc measuring d uses the segment", fontsize=12)
    ax.legend(ncol=2, fontsize=8)
    fig.suptitle("Segment involvement against relative segment size (all lengths, grouped by k)")
    fig.tight_layout()
    return save(fig, "segment_involvement_vs_length.png")


def fig_boundary_focus(L: int):
    """Crossings per boundary for the least-forced solution at L."""
    res = RESULTS.lengths[L]
    k = res.num_segments
    # Show the solution that comes closest to having a never-forced boundary.
    rulers = rd.sample_rows(res.rulers, ARGS.max_rulers or None)
    u_all = rd.usage(rulers, L)
    best = int(np.argmin(u_all["forced"].min(axis=1)))
    ruler = rulers[best]
    forced, crossed = u_all["forced"][best], u_all["crossed"][best]
    bound = max(0, L - k * (k + 1) // 2)
    fig = plt.figure(figsize=(16, 7.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.4])
    ax = fig.add_subplot(gs[0, 0], projection="polar")
    marks = np.concatenate([[0], np.cumsum(ruler)[:-1]])
    theta = 2 * np.pi * marks / L
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    for i in range(k):
        t0 = theta[i]
        t1 = t0 + 2 * np.pi * ruler[i] / L
        ts = np.linspace(t0, t1, 50)
        ax.plot(ts, np.ones_like(ts), color=FG, lw=2)
        if ruler[i] / L > 0.025:
            ax.text((t0 + t1) / 2, 1.14, str(int(ruler[i])), ha="center", va="center", fontsize=9)
    cmap = plt.get_cmap("plasma")
    norm = plt.Normalize(forced.min(), max(forced.max(), forced.min() + 1))
    ax.scatter(theta, np.ones(k), s=150, c=forced, cmap=cmap, norm=norm, zorder=3, edgecolors=FG)
    last = -1.0
    for j in range(k):
        if theta[j] - last > 0.12 or j == int(np.argmin(forced)):
            ax.text(theta[j], 0.84, f"b{j}", ha="center", va="center", fontsize=8, color=FG)
            last = theta[j]
    ax.set_ylim(0, 1.25)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.grid(False)
    ax.spines["polar"].set_visible(False)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.6, pad=0.08) \
        .set_label("forced crossings")
    ax.set_title(f"{fmt_ruler(ruler)}", fontsize=10)
    a2 = fig.add_subplot(gs[0, 1])
    xs = np.arange(k)
    a2.bar(xs - 0.2, crossed, width=0.4, color=BLUE, label="crossed by some arc measuring d")
    a2.bar(xs + 0.2, forced, width=0.4, color=[cmap(norm(v)) for v in forced],
           label="forced: every arc measuring d crosses it")
    a2.axhline(bound, color=RED, lw=1.5, ls="--", label=f"lower bound L - k(k+1)/2 = {bound}")
    a2.set_xticks(xs)
    a2.set_xticklabels([f"b{j}" for j in xs])
    a2.set_xlabel("boundary (b_j sits before segment s_j)")
    a2.set_ylabel(f"distances d in 1..{L - 1}")
    a2.legend(fontsize=9, loc="lower right")
    a2.set_title("Distances that cross each boundary", fontsize=12)
    fig.suptitle(f"Boundary crossings at L = {L} (k = {k}): the solution with the least-forced boundary")
    fig.tight_layout()
    return save(fig, f"boundary_crossings_L{L}.png")


def fig_boundary_vs_length():
    """Forced crossings at the least-crossed boundary, per length."""
    S = SUMMARY
    Ls = np.array([L for L in S.lengths if S.k[L] >= 2])
    lo, med, hi, zero, used = [], [], [], [], []
    for L in Ls:
        h = S.stats[L].forced_min_hist
        c = np.cumsum(h)
        tot = c[-1]
        used.append(tot)
        lo.append(np.searchsorted(c, 1))
        med.append(np.searchsorted(c, tot / 2))
        hi.append(np.searchsorted(c, tot))
        zero.append(h[0])
    ks = np.array([S.k[L] for L in Ls])
    bound = np.maximum(0, Ls - ks * (ks + 1) // 2)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1.3]})
    ax.fill_between(Ls, lo, hi, color=BLUE, alpha=0.3, step="mid", label="range over solutions")
    ax.plot(Ls, med, color=BLUE, drawstyle="steps-mid", lw=1.5, label="median solution")
    ax.plot(Ls, bound, color=RED, ls="--", lw=1.5, drawstyle="steps-mid",
            label="lower bound L - k(k+1)/2")
    ax.set_ylabel("forced crossings at the\nleast-crossed boundary")
    ax.set_title("Can the circle be cut somewhere without losing a distance?")
    ax.legend(loc="upper left")
    zero = np.array(zero, float)
    share = zero / np.maximum(np.array(used, float), 1)
    ax2.bar(Ls, np.where(share > 0, share, np.nan), color=GREEN, width=0.8)
    with_zero = [int(L) for L, z in zip(Ls, zero) if z]
    if with_zero:
        ax2.text(0.99, 0.9, f"only at L = {', '.join(map(str, with_zero))}; "
                 f"{int(zero.sum())} solutions in total", transform=ax2.transAxes,
                 ha="right", va="top", fontsize=10)
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel("share with a never-\nforced boundary")
    ax2.set_xlabel("ruler length L")
    return save(fig, "boundary_crossings_vs_length.png")


def fig_distance_redundancy():
    """Extra arc pairs per distance, and the share spent on small distances."""
    S = SUMMARY
    Ls = S.lengths
    dmax = max(Ls) // 2
    grid = np.full((dmax, len(Ls)), np.nan)
    for j, L in enumerate(Ls):
        st = S.stats[L]
        mean = st.mult_sum / st.n
        half = L // 2
        # Arcs come in complementary pairs measuring d and L-d, so count extra
        # pairs per distance class d <= L/2. At d = L/2 (even L) both arcs of a
        # pair measure the same distance.
        extra = mean[1: half + 1] - 1
        if L % 2 == 0 and half:
            extra[-1] = mean[half] / 2 - 1
        grid[:half, j] = extra
    fig, (ax, ax2), (cax, cax2) = stacked([3, 1.2], (14, 10))
    masked = np.ma.masked_invalid(np.where(grid > 0, grid, np.nan))
    im = ax.imshow(masked, origin="lower", aspect="auto", cmap=heat_cmap("inferno"),
                   norm=LogNorm(vmin=1e-3, vmax=np.nanmax(grid)),
                   extent=(Ls[0] - 0.5, Ls[-1] + 0.5, 0.5, dmax + 0.5), interpolation="nearest")
    fig.colorbar(im, cax=cax).set_label("mean extra arc pairs measuring d")
    ax.set_ylabel("distance d (≤ L/2)")
    ax.set_title("Where the spare arcs go: distances measured more than once")
    ax.grid(False)
    for cut, color in ((1, YELLOW), (3, ORANGE), (10, RED)):
        small = np.array([np.nansum(grid[:cut, j]) for j in range(len(Ls))])
        total = np.array([np.nansum(grid[:, j]) for j in range(len(Ls))])
        ax2.plot(Ls, np.divide(small, total, out=np.zeros_like(small), where=total > 0), "o-",
                 color=color, ms=3, lw=1, label=f"d ≤ {cut}")
    ax2.set_ylabel("share of spare\narc pairs")
    ax2.set_xlabel("ruler length L")
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc="upper right", fontsize=8, ncol=3)
    cax2.axis("off")
    return save(fig, "distance_redundancy.png")


def fig_polar_rulers():
    """The first solution of each length as a ring."""
    S = SUMMARY
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(0, 3)
    for L in S.lengths:
        ruler = example_ruler(L)
        k = len(ruler)
        widths = 2 * np.pi * ruler / L
        starts = np.concatenate([[0], np.cumsum(widths)[:-1]])
        ax.bar(starts + widths / 2, 0.9, width=widths, bottom=L, color=cmap(norm(ruler * k / L)),
               edgecolor=BG, linewidth=0.4)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, pad=0.08, shrink=0.6).set_label("segment value relative to L/k")
    ax.set_yticks([L for L in S.lengths if L % 20 == 0])
    ax.set_yticklabels([f"L={L}" for L in S.lengths if L % 20 == 0], fontsize=8)
    ax.set_xticks([])
    ax.set_title("First canonical solution per length (one ring per L, angle ∝ segment length)")
    return save(fig, "polar_rulers.png")


def fig_search_performance():
    """Time, work, throughput, and hit rate per length."""
    rows = [RESULTS.lengths[L] for L in SUMMARY.lengths]
    Ls = np.array([r.length for r in rows])
    clock = np.array([r.clock_time for r in rows])
    evald = np.array([r.evaluated for r in rows], float)
    space = np.array([float(math.comb(r.length - 1, r.num_segments - 1)) if r.num_segments else np.nan
                      for r in rows])
    hits = np.array([r.count for r in rows], float)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    a, b, c, d = axes.flat
    ok = clock > 0
    a.plot(Ls[ok], clock[ok], "o", color=BLUE, ms=4)
    a.set_yscale("log")
    a.set_ylabel("seconds")
    a.set_title("Wall-clock time per length", fontsize=12)
    b.plot(Ls, space, "-", color=YELLOW, lw=1.5, label="all compositions C(L-1, k-1)")
    b.plot(Ls[evald > 0], evald[evald > 0], "o", color=GREEN, ms=4, label="rulers evaluated")
    b.set_yscale("log")
    b.legend(fontsize=9)
    b.set_title("Search space vs work done", fontsize=12)
    rate = np.divide(evald, clock, out=np.full_like(evald, np.nan), where=clock > 0)
    c.plot(Ls, rate, "o", color=PURPLE, ms=4)
    c.set_yscale("log")
    c.set_ylabel("evaluations / s")
    c.set_xlabel("ruler length L")
    c.set_title("Throughput (wall clock)", fontsize=12)
    d.plot(Ls, np.divide(hits, evald, out=np.full_like(hits, np.nan), where=evald > 0), "o",
           color=RED, ms=4, label="solutions / evaluated")
    d.plot(Ls, np.divide(evald, space, out=np.full_like(hits, np.nan), where=space > 0), "o",
           color=TEAL, ms=4, label="evaluated / C(L-1, k-1)")
    d.set_yscale("log")
    d.set_xlabel("ruler length L")
    d.legend(fontsize=9)
    d.set_title("Hit rate and share of the space visited", fontsize=12)
    fig.suptitle(f"Search performance ({os.path.basename(RESULTS.source)}, v{RESULTS.version})")
    fig.tight_layout()
    return save(fig, "search_performance.png")


GROUPS = {
    "overview": [fig_segments_vs_length, fig_solutions_vs_length, fig_solutions_vs_slack, fig_count_pairs,
                 fig_segment_value_range, fig_segment_frequency_heatmap, fig_segment_structure,
                 fig_polar_rulers],
    "adjacency": [fig_adjacency_by_length, ("focus", fig_adjacency_focus)],
    "relations": [fig_relations_by_length, fig_solution_families, fig_lineage_small],
    "involvement": [fig_involvement_vs_length, ("focus", fig_involvement_focus)],
    "boundaries": [fig_boundary_vs_length, ("focus", fig_boundary_focus)],
    "redundancy": [fig_distance_redundancy],
    "performance": [fig_search_performance],
}


def run_figure(job):
    """Draw one figure; returns its path and the seconds it took."""
    func, arg = job
    t = time.time()
    path = func(arg) if arg is not None else func()
    return path, time.time() - t


def main():
    """Parse arguments, analyse the results, and draw the requested figures."""
    global RESULTS, SUMMARY, ARGS
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("filename", help="results JSON written by the search")
    p.add_argument("--outdir", default="plots", help="output directory (default: plots)")
    p.add_argument("--only", nargs="+", choices=sorted(GROUPS), help="figure groups to draw")
    p.add_argument("--focus", nargs="+", type=int, default=[57, 100],
                   help="lengths that get detailed per-length figures (default: 57 100)")
    p.add_argument("--adjacency-lengths", nargs="+", type=int,
                   help="lengths shown in adjacency_by_length.png (default: 12 spread out)")
    p.add_argument("--lineage-max", type=int, default=27,
                   help="largest length drawn in lineage_small.png (default: 27)")
    p.add_argument("--min-length", type=int, default=1)
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--include-incomplete", action="store_true",
                   help="also analyse lengths whose search did not finish")
    p.add_argument("--max-rulers", type=int, default=0,
                   help="cap on rulers per length for the arc-usage statistics (0 = all)")
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                   help="worker processes (default: half the logical CPUs; the analysis is "
                        "memory-bound, so hyperthreads do not help)")
    p.add_argument("--chunk", type=int, default=100_000, help="rulers per analysis task")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--cache-dir", default=rd.DEFAULT_CACHE_DIR,
                   help="where parsed rulers are cached ('' disables the cache)")
    ARGS = p.parse_args()
    os.makedirs(ARGS.outdir, exist_ok=True)
    apply_style()

    t0 = time.time()
    RESULTS = rd.load_results(ARGS.filename, cache_dir=ARGS.cache_dir or None)
    lengths = [r.length for r in RESULTS.with_rulers()
               if (r.completed or ARGS.include_incomplete)
               and r.length >= ARGS.min_length
               and (ARGS.max_length is None or r.length <= ARGS.max_length)]
    skipped = [r.length for r in RESULTS.with_rulers() if not r.completed and not ARGS.include_incomplete]
    print(f"Loaded {sum(RESULTS.lengths[L].count for L in lengths):,} rulers over "
          f"{len(lengths)} lengths in {time.time() - t0:.1f}s"
          + (f" (skipping incomplete lengths {skipped})" if skipped else ""))
    focus = [L for L in ARGS.focus if L in lengths]

    groups = ARGS.only or list(GROUPS)
    # Skip the expensive per-ruler analyses that no requested figure uses.
    ARGS.need_usage = bool({"involvement", "boundaries"} & set(groups))
    ARGS.need_links = bool({"relations", "overview"} & set(groups))
    ctx = mp.get_context("fork")
    t1 = time.time()
    with ctx.Pool(ARGS.jobs) as pool:
        SUMMARY = analyse(RESULTS, lengths, pool, ARGS.chunk)
    print(f"Analysed lengths in {time.time() - t1:.1f}s")

    jobs = []
    for group in groups:
        for item in GROUPS[group]:
            if isinstance(item, tuple):
                jobs += [(item[1], L) for L in focus]
            else:
                jobs.append((item, None))
    t2 = time.time()
    with ctx.Pool(min(ARGS.jobs, len(jobs))) as pool:
        for path, secs in pool.imap_unordered(run_figure, jobs):
            print(f"  {path}  ({secs:.1f}s)")
    print(f"Drew {len(jobs)} figures in {time.time() - t2:.1f}s; total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
