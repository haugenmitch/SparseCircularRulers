# Visualizations

Figures that show what minimal complete circular sparse rulers look like, how they relate across lengths, and why they exist. Numbers below come from `results/v0.10.2_gpu.json` (lengths 1–131, 11.7 M distinct solutions; length 132 is incomplete and is skipped by default).

## Running

```bash
pip install -r visualizations/requirements.txt          # numpy + matplotlib only
python visualizations/generate_plots.py results/v0.10.2_gpu.json --outdir visualizations/plots
python visualizations/compare_timing.py results/v0.9.3_cpu_1_to_85.json results/v0.10.2_gpu.json \
    --out visualizations/plots/timing_comparison.png
```

Useful options for `generate_plots.py`:

| Option | Effect |
| --- | --- |
| `--only GROUP …` | Draw only some groups: `overview adjacency relations involvement boundaries redundancy performance`. Analyses no requested figure needs are skipped (`--only performance` runs in ~4 s). |
| `--focus L …` | Lengths that get the detailed per-length figures (default `57 100`). |
| `--adjacency-lengths L …` | Lengths in the adjacency grid (default: 12 spread over the range). |
| `--lineage-max L` | Largest length in `lineage_small.png` (default 27). |
| `--min-length`, `--max-length`, `--include-incomplete` | Choose lengths. |
| `--max-rulers N` | Sample at most N rulers per length for the arc-usage statistics (default: all). |
| `--jobs N` | Worker processes (default: half the logical CPUs). |
| `--dpi N` | Output resolution (default 150). |

The style follows `~/repos/StyleGuide/mch.mplstyle` when present (also `$MCH_STYLE` or an installed `mch` style), and falls back to the same values built into the script.

## Efficiency

The old script ran `json.load` on the whole results file and kept every ruler as Python lists inside a pandas frame. On the 595 MB v0.10.2 file it passed 26 GB of memory in 36 s and had to be killed. The new pipeline:

- **Parses rulers with numpy, not `json`.** `ruler_data.py` pulls each `"rulers"` array out of the raw bytes with one regular expression and converts the digits with `np.array(..., dtype=uint16)`, giving one `(solutions, k)` matrix per length. The first parse takes 9 s.
- **Caches the parsed matrices** in `visualizations/.cache/*.npz`, keyed by the file's path, size, and mtime. Later runs load 11.7 M rulers in 0.1 s.
- **Vectorizes every analysis.** Arc sums come from prefix sums; distance multiplicity, segment involvement, and boundary crossings use `np.bincount` over (ruler, distance, segment) indices in small chunks; canonical forms only compare rotations that start at the row minimum.
- **Matches rulers across lengths by hashing.** Linking L-1 to L searches from whichever side produces fewer candidates (parents of L, or children of L-1), hashes canonical forms to 64 bits, and confirms each hash hit exactly.
- **Runs in parallel.** Large lengths are split into 100 k-ruler chunks spread over a process pool, and the figures are drawn in the same pool.

| | Old script | New script |
| --- | --- | --- |
| v0.10.2 (595 MB, 131 lengths) | > 26 GB, killed | 29 s, 0.8 GB peak (first run +9 s to parse) |
| v0.10.0 (61 MB, 85 lengths) | 7.0 s, 4.2 GB | 3.1 s, 0.85 GB |

Dependencies dropped from pandas + seaborn + numpy + matplotlib to numpy + matplotlib.

## What changed from the previous figures

| Previous figure | Change | Reason |
| --- | --- | --- |
| `segments_vs_length.png` (actual, lower bound, difference as scatter) | Kept the name. Lower bound drawn as a step, lengths where k drops are circled, and a panel of the slack k(k-1) − (L-1) was added. | The difference series sat at 0–2 on the same axis as k and was hard to read. Slack (spare arcs) drives solution counts, as `solutions_vs_slack.png` and `solution_count_pairs.png` show. |
| `num_rulers_log.png`, `solutions_vs_length_colored.png` | Merged into `solutions_vs_length.png`, with lines joining lengths on the same k plateau and isolated lengths circled. | The two were the same scatter. Joining plateaus shows the sawtooth: each new k starts at its peak count and decays. |
| `segment_stats_vs_length.png`, `max_vs_minmax_segments.png` | Replaced by `segment_value_range.png`. | They mixed a count (k) with segment values on one axis. The new figure shows the range of the largest segment against L/k and L/2. |
| `heatmap_segment_freq.png` | Now `segment_frequency_heatmap.png`: log colour, empty cells transparent, L/k overlaid. | On a linear scale the 1s dominate and everything above ~10 looked empty. |
| `segment_adjacency_heatmap.png` (pooled over all lengths) | Replaced by `adjacency_by_length.png` and `adjacency_L{L}.png`. | Pooling lengths 1–131 blurs together pairs whose meaning depends on L. Per-length panels show the a + b ≤ ~L/2 triangle and the diagonal of equal neighbours. |
| `polar_rulers.png` | Coloured by segment value relative to L/k instead of `value % 20`; rings start at north. | Colour now means something: large segments stand out at every length. |
| `solution_ratios_log.png`, `cost_per_eval_log.png`, `cpu_time_vs_length.png`, `throughput_log.png`, `theoretical_complexity_log.png` | Merged into `search_performance.png` (2×2). Uses wall-clock time; CPU time on GPU runs only counts the dispatch thread. | Five single-series plots of the same run read better side by side. The "global ratio" (solutions / evaluations across all lengths) was dropped because it has no meaning per length. |
| `plot_v094_timing.py` | Replaced by `compare_timing.py`. | The old script hard-coded `results/v094*.json`, which no longer exists. The new one takes any files and adds a speed-up panel. |

## Figures

### Overview

**`segments_vs_length.png`.** Minimal k per length against the counting lower bound (smallest k with k(k-1)+1 ≥ L), plus slack = k(k-1) − (L-1), the number of arcs available beyond the L-1 distances needed.
*Why:* it shows where the search meets the bound and where it needs one more segment.
*Shows:* k equals the bound up to L = 19. After that it is one above the bound on a growing share of lengths, and from L = 94 on almost everywhere. The lengths where k drops below k(L-1) are 21, 31, 39, 57, 73, 91, 95, 117. 21, 31, 57, 73, 91 are k² − k + 1 with k-1 a prime power, the perfect difference sets.

**`solutions_vs_length.png`.** Distinct solutions per length, coloured by k, with lines joining consecutive lengths on the same plateau.
*Why:* the counts have a sawtooth structure that a plain scatter hides.
*Shows:* each new k starts with the most solutions (2.4 M at L = 94, 1.9 M at L = 114). The count then falls in steps of two lengths: L = 2m and 2m+1 have similar counts, and each step costs a factor of about 3 (see `solution_count_pairs.png`). Circled lengths (50, 64, 78, 91, 92, 109–112, 129, 130) have no solution one edit away from a solution at L-1 or L+1.

**`solution_count_pairs.png`.** Top: N(L) / N(L-1), the ratio of solution counts at neighbouring lengths, split by parity of L. Bottom: distance classes to cover, ⌊L/2⌋, against complementary arc pairs available, k(k-1)/2.
*Why:* counts at neighbouring lengths are often nearly equal, and the figure shows which neighbours pair up and why.
*Shows:* while k stays the same, going from an even L = 2m to the odd 2m+1 changes the count by a typical factor of 1.18. Going from an odd L = 2m-1 to the even 2m changes it by a typical factor of 3.3, and always downward. So the pairs are (2m, 2m+1), not (2m-1, 2m).

The cause is counting. An arc (a run of consecutive segments) and the arc made of the remaining segments always measure d and L-d together. A ruler with k segments has k(k-1) proper arcs, which form k(k-1)/2 complementary pairs, and each pair covers one *distance class* {d, L-d}. A ruler is complete exactly when every class with 1 ≤ d ≤ ⌊L/2⌋ is covered. At L = 2m the classes are {1, 2m-1}, …, {m-1, m+1}, {m}. At L = 2m+1 they are {1, 2m}, …, {m, m+1}. That is m classes both times. Only the step to an even length adds a class. So L = 2m and 2m+1 have the same k(k-1)/2 pairs to place over the same number of classes, the same number of spare pairs k(k-1)/2 − ⌊L/2⌋, and about the same number of solutions. The lower bound follows the same rule: k(k-1)/2 ≥ ⌊L/2⌋ is the bound k(k-1)+1 ≥ L in the top figure.

This is a counting match, not a one-to-one map between the two lengths. Only about 1.7 % of the solutions at L = 97 can be made from a solution at L = 96 by growing one segment. The pairing breaks down in the sparse tail of each plateau, where counts are small and noisy (L = 48/49: 124 vs 42; L = 130/131: 16 vs 130).

**`solutions_vs_slack.png`.** The same counts, plotted against spare arc pairs k(k-1)/2 − ⌊L/2⌋. Filled dots are even L and open circles odd L; the line joins the geometric mean at each x.
*Why:* to test whether spare capacity explains how many solutions there are. The previous version used spare arcs k(k-1) − (L-1), which differs by one between L = 2m and 2m+1. That split each pair into two points and drew a zigzag that hid the pairing.
*Shows:* even and odd lengths with the same number of spare pairs land on top of each other. For every k the count rises roughly exponentially with spare pairs (about 3× per pair), and the curves for different k are close to parallel. With zero spare pairs (perfect difference sets) there are only a handful of solutions.

**`segment_value_range.png`.** The smallest and largest "largest segment" over all solutions at each L, against L/k and L/2.
*Why:* the longest gap bounds how sparse a ruler can be.
*Shows:* up to about L = 90 some solution has a largest segment close to L/2 (a segment longer than L/2 cannot work, because its complement must also be measured). Even the smallest largest segment is about 1.5–2.5 × L/k, so no solution is close to evenly spaced.

**`segment_frequency_heatmap.png`.** Segments of each value per solution, per length, log colour.
*Why:* to see which segment values solutions use.
*Shows:* small values dominate at every length, the tail extends to about L/2, and lengths with few solutions (thin columns) use a sparse subset of values.

**`segment_structure.png`.** Top: share of solutions with c unit segments. Bottom: solutions per distinct segment multiset.
*Why:* distance 1 must be a single segment, so every minimal ruler has a 1; the question is how many and whether the order of the segments matters.
*Shows:* every solution has at least one 1, and the mean is about 1.3–1.4. The multiset ratio is 1 on sparse lengths and peaks at the start of each plateau (11.4 at L = 38), meaning that where there is slack, several orderings of the same segment values all work.

**`polar_rulers.png`.** The first canonical solution of every length as one ring, angle proportional to segment length, coloured by value relative to L/k.
*Why:* a quick picture of what solutions look like.
*Shows:* every canonical ruler starts with its run of 1s (dark wedge at north), followed by mid-sized segments and one or two long ones.

### Adjacency: which segment lengths border each other

**`adjacency_by_length.png`.** For 12 lengths: how often a segment of value a sits next to one of value b (cyclic, both orders, log colour). The dashed line marks a + b = L/2.
*Why:* you asked how often segment lengths border each other for the solutions of a given length. Doing this per length keeps pairs at different scales apart.
*Shows:* almost all neighbour pairs sum to at most about L/2. Two adjacent segments form an arc, and its complement must stay long enough to measure the other distances. A faint diagonal of equal neighbours (a = b) shows up at many lengths. On lengths with many solutions the pattern is smooth; on sparse lengths it is a scatter of specific pairs.

**`adjacency_L{L}.png`** (per `--focus` length). The pair matrix, plus P(neighbour | segment).
*Why:* the conditional view shows what sits next to a given segment value.
*Shows:* at L = 100 the largest segments (40–50) are usually next to a 1, 2 or 3 (70 % of the time for a 50). Long gaps are flanked by short segments so that the long arc and its short extensions both measure useful distances.

### Relations between solutions of different lengths

Two one-unit edits turn a solution at L-1 into a candidate at L:

- **extend**: one segment grows by 1 (k stays the same);
- **split**: a segment s becomes two segments a, b with a + b = s + 1 (k grows by 1).

`ruler_data.link_lengths` finds every such pair in which both rulers are solutions.

**`relations_by_length.png`.** Share of solutions at L with a parent at L-1 (top) and with a child at L+1 (bottom), per edit type. Isolated lengths are marked.
*Why:* you asked how solution rulers are related between lengths.
*Shows:* below L ≈ 20 nearly every solution comes from a shorter one. The share then falls steadily, to about 4 % at L = 81, 1.5 % at L = 100 and 0.25 % at L = 120. Split edits only matter at the end of a plateau: every solution at the last length of each plateau up to L = 39 splits into a solution of the next k, but beyond L = 50 the share is 0–50 %, and L = 91's six solutions have no split child at all.

**`solution_families.png`.** Connected components of the one-edit graph over all lengths: how many lengths each spans, and the 30 longest.
*Why:* to see whether solutions form long lineages or are mostly isolated.
*Shows:* 98.3 % of the 11.7 M solutions have no relative one edit away. Only one family spans more than 11 lengths: the small-ruler family from L = 1 to 20. Long chains do exist, for example `[1, 1, 4, 9, 2, 17, 3, 7, x, 18, 6]` is a solution for every x from 12 to 21 (L = 80–89). Most solutions at large L are not reachable from shorter solutions, which is why an incremental search would not replace the full search.

**`lineage_small.png`.** Every solution up to L = 27 as a dot, with every extend (blue) and split (green) link drawn.
*Why:* the full graph is only readable at small L, and it shows the structure the summary figures count.
*Shows:* the graph is a dense web up to L ≈ 20, all of L = 20 comes from splits of L = 19 and earlier, and links thin out quickly after that.

### Segment involvement

For a ruler and a distance d ≤ L/2, look at every arc (run of consecutive segments) that measures d. A segment is **involved** in d if some such arc contains it, and **essential** to d if every such arc contains it. Only d ≤ L/2 is counted, because d and L-d are complementary arcs and every segment lies in one of each pair.

**`segment_involvement_L{L}.png`.** Left: for one example solution, which segments are involved in / essential to each distance. Right: across all solutions at L, segment value against the number of distances it is involved in.
*Why:* you asked how often each segment is involved in generating the distances of a solution.
*Shows:* in `[1, 1, 1, 1, 1, 8, 7, 8, 6, 22, 7, 37]` the run of 1s and the mid-sized segments are each involved in 19–27 of the 50 distances, while the 22 is involved in only 7. Across all L = 100 solutions, involvement falls steadily with segment size, from about 20 distances for a 1 to a few for the largest segments.

**`segment_involvement_vs_length.png`.** Involvement and essentiality as a share of distances, against segment value relative to L/k, combined over all lengths with the same k.
*Why:* to check whether the per-length pattern holds everywhere.
*Shows:* curves for different k fall almost on top of each other. Small segments (below L/k) take part in about 40 % of distances and are essential to about 25 %; segments 3–4 × L/k drop to about 20 % involved and 10 % essential. Short segments do the fine adjustment; long gaps mainly set the coarse distances.

### Boundary crossings

A boundary is the mark between two segments. An arc **crosses** boundary j if j lies strictly inside it. For each distance d in 1..L-1, boundary j is **crossed** if some arc measuring d crosses it, and **forced** if every arc measuring d crosses it.

A boundary with zero forced crossings is a place where the circle can be cut into a complete linear ruler. Inserting a new segment of any length there gives a complete circular ruler with one more segment, which is the cheap construction you suggested. I checked this on every such case up to L = 9 and it does produce minimal solutions (e.g. `[1, 1, 2, 4]` at L = 8 with a 6–9 inserted gives L = 14–17 solutions).

**`boundary_crossings_L{L}.png`.** For the solution at L with the least-forced boundary: the ruler drawn as a circle with boundaries coloured by forced crossings, and the crossed / forced counts per boundary against the lower bound.
*Why:* you asked how often each boundary is crossed, including whether any boundary is never needed.
*Shows:* at L = 100 the best solution `[1, 2, 1, 4, 6, 18, 7, 2, 17, 22, 8, 12]` still has 26 distances that cannot be measured without crossing its least-used boundary, against a lower bound of 22.

**`boundary_crossings_vs_length.png`.** The forced-crossing count at each solution's least-crossed boundary (range and median over solutions) against the lower bound L − k(k+1)/2, and the share of solutions with a never-forced boundary.
*Why:* to answer whether any solution has a boundary that is never forced.
*Shows:* only 12 solutions do, all at L ≤ 9 (L = 2, 3, 4, 5, 6, 8, 9). The dashed line explains why. Cutting at a boundary leaves a linear ruler with k segments, which has k(k+1)/2 − 1 proper sub-intervals, so at least L − k(k+1)/2 distances must cross every boundary. Since k ≈ √L, that bound is about L/2 and grows with L, so the cut-and-insert construction cannot work beyond the smallest rulers. The best solutions stay within a few crossings of the bound.

### Redundancy

**`distance_redundancy.png`.** For each L and distance d ≤ L/2, the mean number of extra arc pairs measuring d (a pair is the arc for d and its complement for L-d), and the share of spare pairs spent on d ≤ 1, 3, 10.
*Why:* slack drives solution counts, so it is worth seeing where solutions spend it.
*Shows:* spare arcs are spread almost evenly over all distances. At L = 60 and L = 100, distances d ≤ 10 take 33 % and 20 % of spare pairs, matching the uniform shares of 33 % and 20 %. Solutions do not over-measure small distances; any distance can absorb the slack.

### Search performance

**`search_performance.png`.** Wall-clock time, rulers evaluated against all compositions C(L-1, k-1), evaluations per second, and hit rate per length.
*Why:* to track the search itself next to the structure it finds.
*Shows:* the search visits 0.1–1 % of all compositions, the hit rate falls from 1 to below 10⁻¹¹, and time jumps by about 10–100 × at each new k.

**`timing_comparison.png`** (`compare_timing.py`). Wall-clock time per length for several results files, and speed-up against the first.
*Why:* to compare program versions; it replaces the hard-coded v0.9.4 script.
*Shows:* v0.10.x on the GPU is about 100 × faster than v0.9.3 on the CPU by L = 85. Below L ≈ 40 every run is bound by fixed per-length overhead (about 0.05 s), so speed-ups there are noise.

## Files

| File | Purpose |
| --- | --- |
| `generate_plots.py` | Analysis and all figures above. |
| `ruler_data.py` | Loading, caching, and vectorized analysis helpers (arc sums, canonical forms, one-edit links, segment and boundary usage). |
| `compare_timing.py` | Timing comparison across results files (reads only metadata). |
| `requirements.txt` | numpy, matplotlib. |

The interactive browsers in `tools/visualizers/` (`index.html` for single rulers, `solution_graph.html` for the 3D relation graph) are separate and unchanged.
