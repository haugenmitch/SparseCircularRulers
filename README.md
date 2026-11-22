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
- [x] Make sure the program is robust up to length 255
- [x] "/s evals" -> "evals/s"
- [x] Robust symmetry breaking (Circular/Reflection) to ensure unique rulers in output.
- [ ] Add an ALGORITHM.md file to describe the algorithms, efficiency gains, and tricks used in the calculations in this program

## Possible Efficiency Improvements

Ideas listed roughly in order of expected gain; checked ones are implemented:

- [x] **Canonical rules during the search.** Besides `s_1 <= s_{n-1}`, the search now tracks the rotations starting at later 1s that still tie with the ruler's start (the next segment must be at least the matching one from the start) and checks the reflection starting at each new 1 as it is placed. On the CPU this visits about 20% fewer nodes. The GPU shader does not apply these rules (checking them there cost more than it saved), but the partial rulers it is given, including those split off its tail, are pruned by them (length 94: 533 → 491 billion nodes). The full canonical check still runs on complete rulers. Generating bracelets directly (e.g. Sawada's algorithms) could get closer to the full 2n symmetry reduction.
- [x] **Skip segment counts proven impossible.** When `L = n^2 - n + 1`, a ruler with `n` segments is a perfect cyclic difference set. Bruck–Ryser–Chowla, plus the computational result that no such set exists below order about 2·10^9 unless the order is a prime power, rules out cases such as `L = 43, n = 7` and `L = 111, n = 11`; those searches now start at `n + 1`.
- [x] **Track only distances 1..floor(L/2).** Distances `d` and `L - d` are covered together, so only half are tracked: the CPU keeps covered distances in fewer words, and the GPU stores each depth's covered distances in half the words (length 94: 70 s → 55 s together with the pruned partial rulers).
- [x] **Specialize the CPU bitsets by length.** The CPU search used 4 × u64 for every length; its bitsets now take 1, 2, or 4 words by length (const generics), and `.cargo/config.toml` builds with `-C target-cpu=native` (hardware `popcnt`), so binaries may not run on older CPUs.
- [x] **Remove the per-batch barrier on the CPU.** Partial rulers were searched in batches of `64 × threads` with a barrier between batches; now each thread takes the next partial ruler as soon as it finishes one, and the main thread collects results and checkpoints meanwhile.
- [x] **Split the GPU tail.** Each GPU slot searches one partial ruler to the end, and a few partial rulers have huge subtrees: at length 94 about a third of the time was spent with only a few hundred slots busy. Now, once every partial ruler has started and at most 1/8 of the slots are busy, the busy slots' saved search states are read back and what each has left to search is handed out again as deeper partial rulers (length 94: 137 s → 70 s).
- [x] **Checkpoint finished partial rulers individually.** GPU slots finish partial rulers out of order, but the checkpoint only recorded the first unfinished one, so a resumed run repeated everything finished after it. Checkpoints now also save the partial rulers finished after it (`checkpoint_finished`, as ranges of indices), which a resumed run skips.
- [x] **Make checkpoints cheap.** Every checkpoint (every 10 s) re-serialized the whole state, including every earlier length's rulers (hundreds of MB by length 100), on the dispatch thread while workers waited. Now each finished length is serialized once and the current length's rulers as they are found, and the file is written on a background thread (to a temporary file that then replaces the save file, so it is never half written). The format is unchanged.
