# Sparse Circular Rulers

Finds every minimal complete circular sparse ruler for each length, on CPU threads or the GPU.

## Definitions

- **Circular ruler**: segments $s_1, \dots, s_k$ around a circle of length $L = \sum s_i$.
- **Complete**: every distance $1, \dots, L-1$ is the sum of some run of consecutive segments (wrapping around).
- **Minimal**: no complete ruler of length $L$ has fewer segments.

Rulers are saved in canonical form: the lexicographically smallest of their rotations and reflections.

## Usage

```bash
cargo run --release -- -t 16 -a 1 -z 100 -s results.json   # CPU, 16 threads, lengths 1-100
cargo run --release -- -g -a 1 -z 100 -s results.json       # GPU
cargo run --release -- -g -z 120 -r results.json            # resume and continue to 120
```

| Option | Meaning |
| --- | --- |
| `-a`, `-z` | First and last length (`-a 0`: first unsolved length). |
| `-t N` | CPU threads. |
| `-g` | Search on the GPU (wgpu: Vulkan, Metal, or DirectX 12). |
| `-s`, `-l`, `-r` | Save to, load from, or both (resume). |
| `--json`, `-q` | Print the state as JSON at the end; no progress output. |

State is saved every 10 seconds and on Ctrl-C; a resumed run continues where it stopped. Lengths up to 255 and up to 32 segments are supported.

`.cargo/config.toml` builds with `-C target-cpu=native`, so binaries may not run on older CPUs.

## Layout

| Path | Contents |
| --- | --- |
| `src/main.rs` | Command line and the loop over lengths. |
| `src/length.rs` | One length's search: prefixes, CPU and GPU scheduling, checkpoints. |
| `src/search.rs` | The backtracking search and its pruning rules. |
| `src/gpu.rs`, `src/shader.wgsl` | The GPU search. |
| `src/ruler.rs` | Canonical form and the segment-count lower bound. |
| `src/state.rs` | The save file. |
| `src/progress.rs` | Terminal progress display. |
| `visualizations/` | Figures from results files ([README](visualizations/README.md)). |
| `tools/` | Interactive web visualizers and results extraction scripts. |

[ALGORITHM.md](ALGORITHM.md) explains the search and its optimizations.

## Tests

```bash
cargo test --release
```

The search is checked against brute force for every length up to 30.
