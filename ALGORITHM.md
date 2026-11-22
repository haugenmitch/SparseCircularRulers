# Algorithm

This document describes how the program finds every minimal complete circular sparse ruler for each length, the tricks that make the search fast, and how the work is split across CPU threads and the GPU. See the [README](README.md) for the definitions.

Source map:

- `src/search.rs`: the backtracking search (`Search`, `Dfs`), pruning, symmetry rules, and work splitting.
- `src/shader.wgsl`: the same search on the GPU.
- `src/gpu.rs`: GPU buffers, passes, and reading results and saved search states back.
- `src/main.rs`: the outer loop over lengths, the segment-count lower bound, `is_canonical`, CPU and GPU scheduling, and checkpoints.

## 1. Outer loop

For each length $L$ (from `--start` to `--end`), the program searches with $n$ segments, starting at a lower bound on $n$. If no complete ruler exists, it tries $n + 1$, and so on. The first $n$ that gives any ruler is the minimum, and every canonical ruler with that $n$ is saved.

### Lower bound on the segment count

A ruler with $n$ marks has $n(n-1)$ ordered pairs of marks, so it measures at most $n(n-1)$ nonzero cyclic distances. All $L - 1$ distances have to be measured, so

$$n^2 - n + 1 \ge L \quad\Longrightarrow\quad n \ge \left\lceil \frac{\sqrt{4L - 3} + 1}{2} \right\rceil.$$

(`get_num_segments_lower_bound`.)

### Lengths that cannot meet the bound

When $L = n^2 - n + 1$ exactly, a ruler with $n$ segments measures every distance exactly once: it is a perfect cyclic difference set of order $q = n - 1$. Such sets are known to exist only when $q$ is a prime power. Bruck–Ryser–Chowla rules out some orders, and a computer search has ruled out every other non-prime-power order below about $2 \cdot 10^9$. So for orders such as 6, 10, 12, and 14 (lengths 43, 111, 157, 211), the search starts at $n + 1$ and skips a search that is known to find nothing.

## 2. Representation

A ruler is a cyclic sequence of segments $s_0, \dots, s_{n-1}$ summing to $L$. The search works with the marks instead: $m_0 = 0$ and $m_i = s_0 + \dots + s_{i-1}$, so $0 = m_0 < m_1 < \dots < m_{n-1} < L$. A cyclic distance $d$ is measured when $m_j - m_i \equiv d \pmod L$ for some pair of marks.

**Fixing $s_0 = 1$.** Distance 1 can only be measured by a single segment of length 1, so every complete ruler (with $L \ge 2$) has a segment of 1. Rotating the ruler so that segment comes first loses nothing, so the search always starts with $m_0 = 0, m_1 = 1$. This also matches the canonical form (section 4): 1 is the smallest possible segment, so the lexicographically smallest rotation or reflection always starts with 1.

**Half the distances.** Measuring $d$ also measures $L - d$ (the same pair in the other direction), so the $L - 1$ distances form $\lfloor L/2 \rfloor$ classes $\{d, L - d\}$. Only distances $1..\lfloor L/2 \rfloor$ are tracked; the ruler is complete when they are all covered.

## 3. The backtracking search

`Dfs::run` places marks left to right. Each partial ruler (`Node`) holds:

- `pos`: the marks placed so far ($k$ of them).
- `marks`: a bitset with bit $m$ set for each mark $m$.
- `rev`: a bitset with bit $L - 1 - m$ set for each mark $m$ (the marks reflected).
- `cov`: the covered distances up to $\lfloor L/2 \rfloor$.
- `tied`: rotations still tied with the start of the ruler (section 4).

### Computing new distances with two shifts

A new mark $p$ lies after every placed mark, so it measures $p - m$ and $L - (p - m)$ for each mark $m$. Both sets come from one shift each:

- `rev >> (L - 1 - p)` moves bit $L - 1 - m$ to bit $p - m$.
- `marks << (L - p)` moves bit $m$ to bit $L - p + m = L - (p - m)$.

Their union contains, for each mark, whichever of the pair falls at or below $L/2$, so OR-ing it into `cov` (keeping only the low words) updates the covered classes in a handful of word operations, with no loop over marks.

**Sliding to the next position.** Moving the candidate from $p$ to $p + 1$ lengthens every $p - m$ by one and shortens every $L - (p - m)$ by one, so the loop over candidate positions shifts the two halves by one bit (`up <<= 1`, `down >>= 1`) instead of recomputing them.

### Pruning rule 1: counting bound

With $k$ marks placed and $r = n - k$ left, the remaining marks form at most $rk$ pairs with the placed marks and $r(r-1)/2$ pairs among themselves. Each pair covers at most one new class, so a child is dropped when

$$\text{missing classes} > rk + \frac{r(r-1)}{2}.$$

Missing classes are counted with `popcount` on `half & !cov`.

### Pruning rule 2: placing the last mark directly

When one mark is left, it has to measure every missing distance. Let $c$ be the smallest missing distance. The last mark $p$ measures $c$ only if $p = m + c$ or $p = m + L - c$ for some placed mark $m$, so the candidate positions are

`(marks << c) | (marks << (L - c))`

intersected with the allowed range. The same is done for the second smallest missing distance, and only the few positions left (out of up to $L$) are checked in full. This replaces the innermost loop of the search, where most nodes are.

### Pruning rule 3: range limits from symmetry

`Search::limit` bounds each mark from above using $s_1 \le s_{n-1}$ (the ruler must not be larger than its reflection about $m_0$, see section 4) and leaving room for the marks after it:

- Mark 2: $s_1 = m_2 - 1$, and the last mark is at least $m_2 + n - 3$ and at most $L - s_1$, so $m_2 \le \lfloor (L + 4 - n)/2 \rfloor$.
- Mark $j > 2$: $m_j \le L - s_1 - (n - 1 - j)$.

`Search::lowest` bounds each mark from below using the rotation rule in section 4.

## 4. Symmetry breaking

A circular ruler has up to $2n$ representations ($n$ rotations, each read in two directions). Only the canonical one, the lexicographically smallest, is kept. `is_canonical` (in `main.rs`) checks this on complete rulers: since the ruler starts with 1, it only compares against rotations and reflections that also start at a segment of 1.

Checking only complete rulers would still search up to $2n$ copies of every ruler, so the search also rejects partial rulers that can no longer be canonical:

- **Reflection about the start:** $s_1 \le s_{n-1}$, enforced through the range limits above.
- **Rotations:** when a later segment $s_i = 1$ is placed, the rotation starting there is "tied" with the ruler so far. While it stays tied ($s_i, s_{i+1}, \dots$ equals $s_0, s_1, \dots$), each new segment must be at least the corresponding segment from the start, or that rotation would be smaller. The tied rotations are a bitmask (`tied`), and the largest required segment gives the lowest position for the next mark (`lowest`). A rotation leaves the mask once its segment is larger.
- **Reflections at new 1s:** when a new segment $s_t = 1$ is placed, the reflection $s_t, s_{t-1}, \dots, s_0$ is compared with $s_0, s_1, \dots, s_t$ right away; if it is smaller, the partial ruler is dropped.

These rules only compare segments that are already placed. Comparisons that wrap around past the last segment need the whole ruler, so `is_canonical` still runs on every complete ruler. On the CPU the rules visit about 20% fewer nodes.

## 5. Splitting the work

`Search::prefixes` expands the search tree breadth first, applying all the pruning and symmetry rules, until a depth has at least `target` partial rulers (prefixes): $2^{16}$ for the CPU and $2^{21}$ for the GPU. It stops at depth $n - 2$ and at $2^{24}$ prefixes. Prefixes come out in lexicographic order, which checkpoints rely on (section 8). Each prefix is then searched to the end independently.

## 6. CPU search

- **Dynamic scheduling:** each Rayon thread takes the next unsearched prefix from an atomic counter as soon as it finishes its last one, so there is no barrier between batches. Results go back over a channel to the main thread, which also writes checkpoints.
- **Bitsets sized by length:** `Bits<W>` is a `[u64; W]` with `W` a const generic, so every loop has a constant bound and unrolls. Positions take `W` words and distances `H` words (only up to $L/2$): `<1, 1>` for $L \le 64$, `<2, 1>` for $L \le 127$, and `<4, 2>` above.
- **Native instructions:** `.cargo/config.toml` builds with `-C target-cpu=native`, so `count_ones` and `trailing_zeros` compile to `popcnt` and `tzcnt`. Binaries may not run on older CPUs.
- **Cheap progress counting:** each thread counts nodes locally and adds them to the shared counter (and checks for Ctrl-C) every 65,536 nodes.

## 7. GPU search

The GPU (through `wgpu`, so Vulkan, Metal, or DirectX 12) runs the same search in a compute shader: each GPU thread searches one prefix in a *slot* that keeps its state between passes. There are $2^{18}$ slots.

### The shader

- **Iterative search:** WGSL has no recursion, so the shader keeps an explicit stack: the marks `pos`, the next position to try at each depth `nextp`, and the covered distances at each depth `covs`. Backtracking removes a mark by toggling its bits in `marks` and `rev`.
- **Same pruning:** the counting bound, the two-shift distance update with one-bit sliding, the last-mark candidates from the two smallest missing distances, and the $s_1 \le s_{n-1}$ range limits.
- **No rotation or reflection rules:** checking them needs per-depth state indexed by each thread's own depth, which cost more GPU time than the nodes they removed saved. The prefixes handed to the GPU (including those split from its tail, below) are still pruned by these rules on the CPU, which cuts the work at length 94 from 533 to 491 billion nodes. The CPU filters the GPU's rulers with `is_canonical`.
- **Bitsets in registers:** the bitset width (`WORDS`, 32-bit words for positions, and `HWORDS` for distances up to $L/2$) is substituted into the shader source when its pipeline is built, one pipeline per width. Words are only indexed by constants; a runtime index is turned into a chain of `select`s over all words (`word`), so the compiler keeps the bitsets in registers instead of spilling them to memory.
- **Driver workarounds:** some forms crash the radv (Mesa) shader compiler and are avoided, such as early returns inside loops and trailing-zero builtins inside loops (the lowest set bit is found with `countOneBits((x & -x) - 1)` instead).

### Passes and node budgets

A dispatch that runs too long trips the driver's GPU timeout, so each thread searches at most `node_budget` nodes per pass, then saves its state (depth, marks, next positions, and the prefix's depth: 66 words per slot) and stops. The next pass resumes it. The budget adapts between $2^8$ and $2^{22}$: it doubles when a pass takes under 100 ms (so launches stay cheap) and halves when one takes over 500 ms. Between passes the CPU puts new prefixes in the slots that finished.

### Collecting results

Rulers are written to a shared results buffer. A thread reserves space for all its rulers at a mark at once with an atomic add, and gives it back if they do not fit; it then stops and redoes that mark in the next pass. So a ruler is never dropped or reported twice. The 64-bit node count is kept in two 32-bit atomics, carrying into the high word on overflow. Only the rulers and finished slots actually written are copied back.

### Splitting the tail

A few prefixes have far larger subtrees than the rest. At length 94 about a third of the time was spent with only a few hundred slots busy. So once every prefix has started, nothing split is waiting, and at most 1/8 of the slots are busy, the CPU reads the busy slots' saved states back and splits each into what it has left to search (`Search::remaining`):

- For each depth $i$ from the prefix's depth to the current depth, every allowed position after the one currently being searched at depth $i$ becomes a new prefix of $i + 1$ marks.
- If the slot was placing the last mark, its current node is searched again whole, since the last mark is placed all at once.

Together these cover exactly the rest of the slot's search, and they fill the idle slots. Each original prefix counts its outstanding pieces and is marked finished when the last one is. This took length 94 from 137 s to 70 s.

## 8. Checkpoints and resuming

Every 10 seconds, and on Ctrl-C, the state is saved:

- **What is saved:** the segments of the first prefix not fully searched (`checkpoint_ruler`), the prefixes fully searched after it as ranges of indices (`checkpoint_finished`, since GPU slots finish out of order), the rulers found so far, node counts, and times.
- **Resuming:** the search is split again the same way, the first unfinished prefix is found by binary search over the sorted prefixes, and the finished ranges are skipped. A resumed run may repeat part of a prefix, so the rulers are sorted and deduplicated at the end.
- **Cheap saves:** each finished length's JSON is serialized once and cached, and the current length's rulers are serialized as they are found, so a save only serializes what changed. The file is written on a background thread to a temporary file that then replaces the save file, so it is never half written and the search does not wait for it.

## 9. Testing

- `matches_brute_force` compares the search with a brute-force enumeration of every composition with $s_0 = 1$, for every length up to 30, at the minimal segment count and the one above it, with 1 and 50 prefixes and with both bitset widths.
- `remaining_covers_rest_of_search` checks that splitting a search at every depth along its first path finds exactly the rulers of the whole search.
- `main.rs` tests `is_canonical` and the lower bound, including the perfect-difference-set lengths.

## Measured gains

From the README, at length 94 on the GPU:

| Change | Effect |
| --- | --- |
| Splitting the GPU tail | 137 s → 70 s |
| Tracking only half the distances, plus symmetry-pruned prefixes | 70 s → 55 s |
| Rotation and reflection rules on prefixes | 533 → 491 billion nodes |

On the CPU, the rotation and reflection rules visit about 20% fewer nodes.
