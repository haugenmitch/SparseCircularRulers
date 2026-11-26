# Visualizations

Figures of what minimal circular rulers look like and how they relate across lengths. Numbers below are from `results/v0.10.2_gpu.json` (lengths 1–131, 11.7 M solutions).

## Running

```bash
pip install -r visualizations/requirements.txt          # numpy, matplotlib
python visualizations/generate_plots.py results/v0.10.2_gpu.json --outdir visualizations/plots
python visualizations/compare_timing.py results/v0.9.3_cpu_1_to_85.json results/v0.10.2_gpu.json \
    --out visualizations/plots/timing_comparison.png
```

| `generate_plots.py` option | Effect |
| --- | --- |
| `--only GROUP …` | Only some groups: `overview adjacency relations involvement boundaries redundancy performance`. Unneeded analyses are skipped. |
| `--focus L …` | Lengths with detailed per-length figures (default `57 100`). |
| `--adjacency-lengths L …` | Lengths in the adjacency grid (default: 12 spread out). |
| `--lineage-max L` | Largest length in `lineage_small.png` (default 27). |
| `--min-length`, `--max-length`, `--include-incomplete` | Which lengths. |
| `--max-rulers N` | Sample at most N rulers per length for arc-usage statistics. |
| `--jobs N`, `--dpi N` | Worker processes (default: half the CPUs), resolution (default 150). |

The style is `~/repos/StyleGuide/mch.mplstyle` when present (or `$MCH_STYLE`), else the same values built in.

## How it stays fast

- `ruler_data.py` pulls each `"rulers"` array out of the raw bytes with a regular expression and parses it with numpy into one `(solutions, k)` matrix per length; `json.load` on the 595 MB file needs over 26 GB.
- Parsed matrices are cached in `visualizations/.cache/` (keyed by path, size, and mtime): 9 s the first time, 0.1 s after.
- Analyses are vectorized (`np.bincount` over chunks), rulers are matched across lengths by hashing canonical forms, and work runs in a process pool.

The whole v0.10.2 file takes 29 s and 0.8 GB.

## Terms

- **Slack**: k(k-1) − (L-1), arcs beyond the L-1 distances needed. **Spare pairs**: k(k-1)/2 − ⌊L/2⌋, the same in complementary arc pairs, each covering one class {d, L-d}.
- **Edits** from L-1 to L: **extend** grows one segment by 1 (same k); **split** turns a segment s into a, b with a + b = s + 1 (k + 1).
- A segment is **involved** in d ≤ L/2 if some arc measuring d contains it, **essential** if every one does.
- An arc **crosses** a boundary (mark) inside it. A boundary is **forced** for d if every arc measuring d crosses it.

## Figures

### Overview

| Figure | Shows |
| --- | --- |
| `segments_vs_length.png` | Minimal k against the lower bound, and slack. k meets the bound up to L = 19, then is often one above it, and almost always from L = 94. k drops at 21, 31, 39, 57, 73, 91, 95, 117; 21, 31, 57, 73, 91 are perfect difference sets. |
| `solutions_vs_length.png` | Solutions per length, lines joining each k plateau. Each new k starts at its peak (2.4 M at L = 94) and falls in steps of two lengths. Circled lengths have no one-edit link to L±1. |
| `solution_count_pairs.png` | N(L)/N(L-1) by parity. Within a plateau, 2m → 2m+1 changes the count by about 1.18×; 2m-1 → 2m by about 3.3×, always down. Only even lengths add a distance class, so 2m and 2m+1 have the same spare pairs. |
| `solutions_vs_slack.png` | Counts against spare pairs. Even and odd lengths coincide; counts grow about 3× per spare pair for every k. |
| `segment_value_range.png` | Range of each solution's largest segment, against L/k and L/2. Even the smallest is 1.5–2.5 × L/k: no solution is near evenly spaced. |
| `segment_frequency_heatmap.png` | Segment values per solution (log colour). Small values dominate; the tail reaches about L/2. |
| `segment_structure.png` | Unit segments per solution (mean 1.3–1.4), and solutions per distinct segment multiset (up to 11.4 at L = 38, where slack allows reorderings). |
| `polar_rulers.png` | The first solution of each length as a ring, coloured by value relative to L/k. |

### Adjacency

| Figure | Shows |
| --- | --- |
| `adjacency_by_length.png` | How often values a and b are neighbours, for 12 lengths. Almost all pairs sum to at most L/2, since the complement of their arc must still measure other distances. |
| `adjacency_L{L}.png` | The pair matrix and P(neighbour \| segment). At L = 100 the largest segments (40–50) sit next to a 1, 2, or 3 about 70% of the time. |

### Relations across lengths

| Figure | Shows |
| --- | --- |
| `relations_by_length.png` | Share of solutions with a one-edit parent at L-1 or child at L+1. Nearly all below L ≈ 20; about 4% at L = 81, 1.5% at 100, 0.25% at 120. |
| `solution_families.png` | Connected components of the one-edit graph. 98.3% of solutions have no relative; only the L = 1–20 family spans more than 11 lengths. So an incremental search could not replace the full one. |
| `lineage_small.png` | Every solution up to L = 27 and its links: a dense web up to L ≈ 20 that thins quickly. |

### Arc usage

| Figure | Shows |
| --- | --- |
| `segment_involvement_L{L}.png` | Involved and essential distances per segment for one solution, and involvement against value for all solutions at L. Involvement falls with segment size. |
| `segment_involvement_vs_length.png` | The same, relative to L/k, grouped by k. Curves for all k coincide: small segments are involved in about 40% of distances, segments of 3–4 × L/k in about 20%. |
| `boundary_crossings_L{L}.png` | The solution with the least-forced boundary at L. At L = 100 it still has 26 forced distances, against a lower bound of 22. |
| `boundary_crossings_vs_length.png` | Forced crossings at each solution's least-crossed boundary against the bound L − k(k+1)/2. Only 12 solutions, all at L ≤ 9, have a never-forced boundary, where the circle could be cut and a segment inserted to make a longer solution. |
| `distance_redundancy.png` | Extra arc pairs per distance. Spare pairs spread evenly over distances; small distances are not over-measured. |

### Performance

| Figure | Shows |
| --- | --- |
| `search_performance.png` | Time, nodes against all compositions C(L-1, k-1), throughput, and hit rate. The search visits 0.1–1% of compositions; time jumps 10–100× with each new k. |
| `timing_comparison.png` | Time per length across results files (`compare_timing.py`), and speed-up over the first. v0.10 on the GPU is about 100× faster than v0.9.3 on the CPU by L = 85. |

## Files

| File | Purpose |
| --- | --- |
| `generate_plots.py` | Analysis and all figures. |
| `ruler_data.py` | Loading, caching, and vectorized helpers (arc sums, canonical forms, one-edit links, segment and boundary usage). |
| `compare_timing.py` | Timing comparison across results files (metadata only). |

The interactive pages in `tools/visualizers/` are separate: `index.html` analyzes one ruler, and `solution_graph.html` draws the one-edit graph in 3D.
