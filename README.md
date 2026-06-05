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
- [ ] Unit tests
- [ ] Info readout/summary of results files
- [x] CUDA/threading using the GPU
- [ ] Re-add latest ruler found display to UI
- [ ] Make sure the program is robust up to length 256 (e.g. `rank` is a u64, is that large enough for long rulers?)
- [ ] "/s evals" -> "evals/s"

