# Sparse Rulers Rust

A Rust program to find minimal complete circular sparse rulers.

## Definitions

- **Circular Sparse Ruler**: A set of $k$ segments $s_1, s_2, \dots, s_k$ that sum to a total length $L$.
- **Complete**: A ruler is complete if every distance $d \in \{1, 2, \dots, L-1\}$ can be formed by summing a contiguous sequence of segments (cyclically).
- **Minimal**: For a given length $L$, the ruler has the minimum possible number of segments $k$ to be complete and the ruler is no longer than $L$.

## Requirements & Design

- **Segment Representation**: Rulers are represented as an array of segment lengths (e.g., `[u8; k]`).
- **Search Strategy**: Backtracking search exploring segment combinations.
- **Concurrency**: Multi-threaded approach where the main thread dispatches partial rulers (e.g., `[A, B, C, None, None]`) to worker threads.
- **Goal**: Find all complete rulers for a specific length $L$ and number of segments $k$.

## Project Status

- [x] Initial Design & Requirements
- [x] Project Setup
- [x] Core Logic Implementation (Validation: [x], Search: [x])
- [x] Multi-threading
- [ ] Optimization
- [ ] Refactor program to allow genericizing of integer sequence calculations
- [x] Unit tests
- [ ] Info readout/summary of results files
- [x] CUDA/threading using the GPU
- [x] Re-add latest ruler found display to UI
- [ ] Make sure the program is robust up to length 256
- [x] "/s evals" -> "evals/s"
- [x] Robust symmetry breaking (Circular/Reflection) to ensure unique rulers in output.
- [ ] Add an ALGORITHM.md file to describe the algorithms, efficiency gains, and tricks used in the calculations in this program

## Possible Efficiency Improvements

Ideas not implemented yet, roughly in order of expected gain:

- [ ] **Canonical rules during the search.** Only `s_1 <= s_{n-1}` is applied while searching; the full canonical check runs on complete rulers. The canonical rotation starts at the longest run of 1s, and every other 1 must be followed (and preceded) by segments no smaller than those after the first, so many partial rulers could be cut early. Alternatively, generate bracelets directly (e.g. Sawada's algorithms) for close to the full 2n symmetry reduction.
- [ ] **Skip segment counts proven impossible.** When `L = n^2 - n + 1`, a ruler with `n` segments is a perfect cyclic difference set. Bruck–Ryser–Chowla, plus the computational result that no such set exists below order about 2·10^9 unless the order is a prime power, rules out cases such as `L = 43, n = 7` and `L = 111, n = 11`; those searches could start at `n + 1`.
- [ ] **Track only distances 1..floor(L/2).** Distances `d` and `L - d` are covered together, so half the bitset is enough.
- [ ] **Specialize the CPU bitsets by length.** The CPU search always works on 4 × u64; using u64 for `L <= 64` and u128 for `L <= 128` (or const generics) would remove the unused words. Building with `-C target-cpu=native` (hardware `popcnt`) alone made the CPU search about 1.45× faster in a quick test.
- [ ] **Remove the per-batch barrier on the CPU.** Partial rulers are searched in batches of `64 × threads` with a barrier between batches; streaming them to the thread pool with dynamic work stealing would keep every thread busy across the uneven subtrees.
- [ ] **Split the GPU tail.** Each GPU slot searches one partial ruler to the end, and a few partial rulers have huge subtrees: at length 94 about a third of the time is spent with only a few hundred slots busy. Splitting long-running partial rulers into deeper ones when slots go idle (or starting from deeper prefixes) would keep the GPU full.
- [ ] **Checkpoint finished partial rulers individually.** GPU slots finish partial rulers out of order, but the checkpoint only records the first unfinished one, so a resumed run repeats everything finished after it.
- [ ] **Make checkpoints cheap.** Every checkpoint (every 10 s) re-serializes the whole state, including every earlier length's rulers (hundreds of MB by length 100), on the dispatch thread while workers wait. Save only the current length and append solutions, use a binary format, or write in the background.
