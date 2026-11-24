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
- [x] Optimization
- [x] Unit tests
- [x] CUDA/threading using the GPU
- [x] Re-add latest ruler found display to UI
- [x] Make sure the program is robust up to length 255
- [x] "/s evals" -> "evals/s"
- [x] Robust symmetry breaking (Circular/Reflection) to ensure unique rulers in output.
- [x] Add an [ALGORITHM.md](ALGORITHM.md) file to describe the algorithms, efficiency gains, and tricks used in the calculations in this program

## Possible Efficiency Improvements

Ideas listed roughly in order of expected gain; checked ones are implemented:

- [x] **Canonical rules during the search.** Besides `s_1 <= s_{n-1}`, the search now tracks the rotations starting at later 1s that still tie with the ruler's start (the next segment must be at least the matching one from the start) and checks the reflection starting at each new 1 as it is placed. On the CPU this visits about 20% fewer nodes. The GPU shader does not apply these rules (checking them there cost more than it saved), but the partial rulers it is given, including those split off its tail, are pruned by them (length 94: 533 → 491 billion nodes). The full canonical check still runs on complete rulers. Generating bracelets directly (e.g. Sawada's algorithms) could get closer to the full 2n symmetry reduction.
- [x] **Skip segment counts proven impossible.** When `L = n^2 - n + 1`, a ruler with `n` segments is a perfect cyclic difference set. Bruck–Ryser–Chowla, plus the computational result that no such set exists below order about 2·10^9 unless the order is a prime power, rules out cases such as `L = 43, n = 7` and `L = 111, n = 11`; those searches now start at `n + 1`.
- [x] **Track only distances 1..floor(L/2).** Distances `d` and `L - d` are covered together, so only half are tracked: the CPU keeps covered distances in fewer words, and the GPU stores each depth's covered distances in half the words (length 94: 70 s → 55 s together with the pruned partial rulers).
- [x] **Specialize the CPU bitsets by length.** The CPU search used 4 × u64 for every length; its bitsets now take 1, 2, or 4 words by length (const generics), and `.cargo/config.toml` builds with `-C target-cpu=native` (hardware `popcnt`), so binaries may not run on older CPUs.
- [x] **Remove the per-batch barrier on the CPU.** Partial rulers were searched in batches of `64 × threads` with a barrier between batches; now each thread takes the next partial ruler as soon as it finishes one, and the main thread collects results and checkpoints meanwhile.
- [x] **Split the GPU tail.** Each GPU slot searches one partial ruler to the end, and a few partial rulers have huge subtrees: at length 94 about a third of the time was spent with only a few hundred slots busy. Now, once every partial ruler has started and at most 1/8 of the slots are busy, the busy slots' saved search states are read back and what each has left to search is handed out again as deeper partial rulers (length 94: 137 s → 70 s).
- [x] **Size the GPU's per-depth arrays by segment count.** The shader kept each thread's marks, next positions, and covered distances in arrays of 32 entries, indexed by the thread's own depth; in registers, every access costs code for every entry. The arrays now hold only `num_segments` entries, the next positions are derived from the marks, and the per-node check counts only the words that hold distances up to `L / 2`. Throughput at length 130 (13 segments) went from 20 to 34 billion nodes/s, and the whole length takes 832 s; lengths 1–100 take 5:44 instead of 7:29 (length 94: 54 s → 42 s). Keeping every GPU slot busy in the tail was measured too: at length 130 the GPU kept 22–27 billion nodes/s with as few as 20,000 slots busy, so it would save only about 3% and was not pursued further.
- [x] **Apply the first step of the symmetry rules on the GPU.** The full rotation and reflection rules in the shader cut 12% of the nodes at 13 segments and 26% at 14, but their per-depth state slowed every node by as much (no gain at 14 segments). Their first step needs no state: after a later segment of 1 the next segment is at least `s_1`, and a new segment of 1 must follow one of at least `s_1`. With only that, registers are unchanged and a 1-in-64 sample of length 132 (14 segments) takes 1,246 s instead of 1,390 s (16% fewer nodes); length 130 (13 segments) takes 807 s instead of 831 s, and length 94 36.6 s instead of 39.3 s; lengths 1–100 take 5:23 instead of 5:44. Short lengths are slightly slower (length 85: 2.8 s → 3.1 s). Moving the per-depth arrays to workgroup memory was tried too: with the covered distances there, the compiler used twice the registers and the search ran at half speed; with only the marks there, it was 2% faster.
- [x] **Take the GPU's last-mark step together.** With 14 segments the search reached the last mark (the costly step that checks which final positions complete the ruler) in 10% of its steps, against 0.4% with 13, so in nearly every step of a 64-thread subgroup some thread took it and the others waited; that, not the hardware, is why 14 segments ran at 18 billion evaluations/s against 31 billion. A thread at the last mark now waits (with subgroup ballots) until at least 1/8 of its subgroup's searching threads are there too. Together with the previous item, the 1-in-64 sample of length 132 takes 974 s (1,246 s with that item alone, 1,390 s before both; 19 instead of 15 billion evaluations/s with every slot busy), length 130 takes 762 s instead of 807 s, length 94 33.7 s, and lengths 1–100 4:39 instead of 5:23. Waiting for 1/2 of the threads helped length 94 more but slowed 13 segments; capping the wait did not help. GPUs without subgroup support skip the wait. Also tried: building each child's shifted distance bitsets from its parent's on descent (no gain) and shifting bitsets by whole words in stages (crashes the radv shader compiler).
- [x] **Checkpoint finished partial rulers individually.** GPU slots finish partial rulers out of order, but the checkpoint only recorded the first unfinished one, so a resumed run repeated everything finished after it. Checkpoints now also save the partial rulers finished after it (`checkpoint_finished`, as ranges of indices), which a resumed run skips.
- [x] **Make checkpoints cheap.** Every checkpoint (every 10 s) re-serialized the whole state, including every earlier length's rulers (hundreds of MB by length 100), on the dispatch thread while workers waited. Now each finished length is serialized once and the current length's rulers as they are found, and the file is written on a background thread (to a temporary file that then replaces the save file, so it is never half written). The format is unchanged.
