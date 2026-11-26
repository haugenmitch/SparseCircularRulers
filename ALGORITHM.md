# Algorithm

How the program finds every minimal complete circular ruler of each length, and what makes it fast. Definitions are in the [README](README.md).

| Section | Code |
| --- | --- |
| 1. Outer loop and lower bound | `main.rs`, `ruler.rs` |
| 2–4. Representation, search, symmetry | `search.rs` (`Search`, `Dfs`), `ruler.rs` (`is_canonical`) |
| 5–6. Splitting and CPU scheduling | `search.rs` (`prefixes`), `length.rs` |
| 7. GPU search | `shader.wgsl`, `gpu.rs`, `length.rs` (`GpuSlots`) |
| 8. Checkpoints | `length.rs`, `state.rs` |

## 1. Outer loop

For each length $L$, search with $n$ segments starting at a lower bound. If no ruler exists, try $n + 1$. The first $n$ with any ruler is the minimum; all its canonical rulers are saved.

### Lower bound

$n$ marks form $n(n-1)$ ordered pairs, so they measure at most $n(n-1)$ nonzero distances. All $L - 1$ must be measured:

$$n^2 - n + 1 \ge L \quad\Longrightarrow\quad n \ge \left\lceil \frac{\sqrt{4L - 3} + 1}{2} \right\rceil.$$

### Lengths that cannot meet it

When $L = n^2 - n + 1$, a ruler with $n$ segments measures every distance exactly once: a perfect cyclic difference set of order $q = n - 1$. These exist only for prime-power $q$ (Bruck–Ryser–Chowla rules out some orders; a computer search rules out every other order below about $2 \cdot 10^9$). So for $q$ = 6, 10, 12, 14 (lengths 43, 111, 157, 211) the search starts at $n + 1$.

## 2. Representation

The search works with marks $0 = m_0 < m_1 < \dots < m_{n-1} < L$, where $m_i = s_0 + \dots + s_{i-1}$. Distance $d$ is measured when $m_j - m_i \equiv d \pmod L$.

- **$s_0 = 1$.** Only a segment of 1 measures distance 1, so every complete ruler has one. Rotating it to the front fixes $m_0 = 0, m_1 = 1$, and matches the canonical form (which starts with its smallest segment).
- **Half the distances.** Measuring $d$ also measures $L - d$, so only $1..\lfloor L/2 \rfloor$ are tracked.

## 3. Backtracking search

`Dfs::run` places marks left to right. A partial ruler (`Node`) holds:

- `pos`: the marks placed so far ($k$ of them);
- `marks`: bit $m$ for each mark;
- `rev`: bit $L - 1 - m$ for each mark;
- `cov`: covered distances up to $\lfloor L/2 \rfloor$;
- `tied`: rotations still tied with the start (section 4).

### New distances in two shifts

A new mark $p$, after every placed mark $m$, measures $p - m$ and $L - (p - m)$:

- `rev >> (L - 1 - p)` moves bit $L - 1 - m$ to $p - m$;
- `marks << (L - p)` moves bit $m$ to $L - (p - m)$.

OR-ing both into `cov` updates the covered distances in a few word operations. Moving the candidate from $p$ to $p + 1$ shifts the two halves by one bit each (`up <<= 1`, `down >>= 1`) instead of recomputing them.

### Pruning 1: counting bound

With $k$ marks placed and $r = n - k$ left, at most $rk + r(r-1)/2$ new pairs can be formed, each covering at most one new distance. A child is dropped when

$$\text{missing} > rk + \frac{r(r-1)}{2},$$

with missing distances counted by `popcount` on `half & !cov`.

### Pruning 2: the last mark directly

The last mark must measure the smallest missing distance $c$, so it lies at $m + c$ or $m + L - c$ for some mark $m$:

`(marks << c) | (marks << (L - c))`

Intersecting this with the same set for the second smallest missing distance leaves a few candidates out of up to $L$, which are checked in full. This replaces the innermost loop, where most nodes are.

### Pruning 3: range limits

`Search::limit` bounds each mark from above using $s_1 \le s_{n-1}$ (section 4) and room for later marks:

- mark 2: $m_2 \le \lfloor (L + 4 - n)/2 \rfloor$;
- mark $j > 2$: $m_j \le L - s_1 - (n - 1 - j)$.

`Search::lowest` bounds each mark from below with the rotation rule.

## 4. Symmetry breaking

A circular ruler has up to $2n$ representations. Only the lexicographically smallest is kept. `is_canonical` checks complete rulers, comparing only against rotations and reflections that start at a segment of 1.

To avoid searching up to $2n$ copies, partial rulers that can no longer be canonical are dropped:

- **Reflection about the start:** $s_1 \le s_{n-1}$, through the range limits.
- **Rotations:** when a later segment $s_i = 1$ is placed, its rotation is *tied* with the ruler. While $s_i, s_{i+1}, \dots$ equals $s_0, s_1, \dots$, each new segment must be at least the matching one from the start. Tied rotations are a bitmask (`tied`); one leaves it once its segment is larger.
- **Reflections at new 1s:** when $s_t = 1$ is placed, $s_t, s_{t-1}, \dots, s_0$ is compared with $s_0, \dots, s_t$ at once.

Comparisons that wrap past the last segment need the whole ruler, so `is_canonical` still runs on every complete ruler. On the CPU these rules visit about 20% fewer nodes.

## 5. Splitting the work

`Search::prefixes` expands the tree breadth first, with all pruning, until a depth has at least $2^{16}$ (CPU) or $2^{21}$ (GPU) partial rulers, called prefixes. It stops at depth $n - 2$ or $2^{24}$ prefixes. Prefixes are in lexicographic order, which checkpoints rely on.

## 6. CPU search

- **Dynamic scheduling:** each Rayon thread takes the next prefix from an atomic counter as soon as it finishes one. Results go over a channel to the main thread, which checkpoints.
- **Bitsets sized by length:** `Bits<W>` is `[u64; W]` with `W` a const generic, so loops unroll. Positions take `W` words and distances `H`: `<1, 1>` for $L \le 64$, `<2, 1>` for $L \le 127$, `<4, 2>` above.
- **Native instructions:** `-C target-cpu=native` turns `count_ones` and `trailing_zeros` into `popcnt` and `tzcnt` (about 1.45× faster).
- **Cheap counting:** threads add their node counts to the shared counter, and check for Ctrl-C, every 65,536 nodes.

## 7. GPU search

The same search runs in a compute shader (wgpu). Each GPU thread searches one prefix in a *slot* that keeps its state between passes; there are $2^{18}$ slots.

### The shader

- **Explicit stack:** WGSL has no recursion, so the shader keeps the marks `pos` and the covered distances per depth `covs`. Backtracking toggles a mark's bits off.
- **Same pruning:** counting bound, two-shift update, last-mark candidates, and range limits.
- **Only the first step of the symmetry rules:** after a later segment of 1, the next is at least $s_1$; a new 1 must follow a segment of at least $s_1$. These need no per-depth state. The full rules cost more GPU time than they saved, so they prune the prefixes on the CPU instead, and the CPU filters GPU rulers with `is_canonical`.
- **Registers, not memory:** the bitset widths (`WORDS`, `HWORDS`) and the segment count (`MARKS`) are substituted into the source per pipeline. Words are indexed only by constants (a runtime index becomes a chain of `select`s), and per-depth arrays hold only `MARKS` entries, so everything stays in registers.
- **Last mark together:** threads in a subgroup run in lockstep and the last-mark step is costly, so a thread there waits (with subgroup ballots) until 1/8 of its subgroup's searching threads are there too.
- **Driver workarounds:** forms that crash the radv compiler are avoided, such as early returns and trailing-zero builtins inside loops (the lowest set bit is `countOneBits((x & -x) - 1)`).

### Passes

A long dispatch trips the driver's GPU timeout, so each thread searches at most `node_budget` nodes per pass, saves its state (66 words per slot), and stops. The budget doubles when a pass takes under 100 ms and halves over 500 ms, between $2^8$ and $2^{22}$. Between passes the CPU refills finished slots.

### Results

A thread reserves room for all its rulers at a mark with one atomic add, and gives it back if they do not fit; it then redoes that mark next pass. So no ruler is dropped or reported twice. The 64-bit node count is two 32-bit atomics with a carry. Only the rulers and finished slots actually written are copied back.

### Splitting the tail

A few prefixes have far larger subtrees than the rest. Once every prefix has started and at most 1/8 of slots are busy, the CPU reads the busy slots' states and splits each (`Search::remaining`):

- for each depth $i$ from the prefix's depth to the current depth, every allowed position after the current one at depth $i$ becomes a new prefix;
- a slot placing the last mark is searched again whole, since the last mark is placed at once.

These cover exactly the rest of the slot's search. Each original prefix is finished when its last piece is.

## 8. Checkpoints

Every 10 seconds and on Ctrl-C the state is saved:

- **Saved:** the segments of the first unfinished prefix (`checkpoint_ruler`), prefixes finished after it as index ranges (`checkpoint_finished`, since GPU slots finish out of order), the rulers so far, node counts, and times.
- **Resuming:** the search is split the same way, the first unfinished prefix is found by binary search, and finished ranges are skipped. Part of a prefix may repeat, so rulers are deduplicated at the end.
- **Cheap saves:** finished lengths are serialized once and the current length's rulers as they are found. A background thread writes a temporary file and renames it over the save file.

## 9. Tests

- `matches_brute_force`: the search against enumerating every composition with $s_0 = 1$, for $L \le 30$, at the minimal segment count and one more, with 1 and 50 prefixes and both bitset widths.
- `remaining_covers_rest_of_search`: splitting at every depth of the first path finds exactly the whole search's rulers.
- `finds_ruler_of_length_255`: the widest bitsets reach the top bits.
- `ruler.rs`: `is_canonical` and the lower bound, including perfect-difference-set lengths.

## Measured gains

On the GPU unless noted.

| Change | Effect |
| --- | --- |
| Splitting the GPU tail | L = 94: 137 s → 70 s |
| Half the distances, symmetry-pruned prefixes | L = 94: 70 s → 55 s; 533 → 491 billion nodes |
| Per-depth arrays sized by segment count | L = 130: 20 → 34 billion nodes/s; L = 1–100: 7:29 → 5:44 |
| First step of the symmetry rules in the shader | L = 132 sample: 1,390 s → 1,246 s; L = 1–100: 5:44 → 5:23 |
| Last mark taken together | L = 132 sample: 1,246 s → 974 s; L = 1–100: 5:23 → 4:39 |
| Symmetry rules (CPU) | about 20% fewer nodes |
| Native `popcnt` (CPU) | about 1.45× faster |

Tried without gain: full symmetry rules in the shader (fewer nodes, but each slower), per-depth arrays in workgroup memory (half speed with covered distances there), building children's shifted bitsets from the parent's, and keeping every GPU slot busy in the tail (about 3% at best).
