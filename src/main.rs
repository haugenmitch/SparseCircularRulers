//! Finds every minimal complete circular sparse ruler for a range of lengths.

mod gpu;
mod length;
mod progress;
mod ruler;
mod search;
mod state;

use clap::Parser;
use gpu::GpuContext;
use length::{CHECKPOINT_INTERVAL, SearchContext, SearchStatus, search_length};
use progress::Progress;
use ruler::num_segments_lower_bound;
use search::MAX_MARKS;
use state::State;
use std::io::{self, Write};
use std::process::exit;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

const VERSION: &str = env!("CARGO_PKG_VERSION");
/// GPU slots (prefixes searched at once) and most rulers one pass can return.
const GPU_SLOTS: u32 = 1 << 18;
const GPU_MAX_RESULTS: u32 = 1 << 18;

/// Cleared on Ctrl-C; searches stop and save.
static RUNNING: AtomicBool = AtomicBool::new(true);

/// Command-line options.
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Path to load state from
    #[arg(short, long)]
    load: Option<String>,

    /// Path to save state to
    #[arg(short, long)]
    save: Option<String>,

    /// Path to both load and save state (combination)
    #[arg(short, long)]
    resume: Option<String>,

    /// Number of threads to use
    #[arg(short = 't', long, default_value_t = 1)]
    threads: usize,

    /// Starting ruler length
    #[arg(short = 'a', long, default_value_t = 1)]
    start: u8,

    /// End ruler length
    #[arg(short = 'z', long, default_value_t = 255)]
    end: u8,

    /// Output JSON to STDOUT at the end
    #[arg(long)]
    json: bool,

    /// Do not output to STDOUT
    #[arg(short, long)]
    quiet: bool,

    /// Use GPU acceleration (wgpu)
    #[arg(short, long)]
    gpu: bool,
}

/// Loads the state, opens the GPU if asked, and runs the search.
fn main() {
    let cli = Cli::parse();
    rayon::ThreadPoolBuilder::new()
        .num_threads(cli.threads)
        .build_global()
        .expect("Failed to initialize thread pool");

    let load_path = cli.load.as_ref().or(cli.resume.as_ref());
    let save_path = cli.save.as_ref().or(cli.resume.as_ref());
    let mut state = load_path.map_or_else(|| State::new(VERSION), |path| load_state(path));
    state.recalculate_global_metrics();

    let gpu = cli.gpu.then(|| {
        pollster::block_on(GpuContext::new(GPU_SLOTS, GPU_MAX_RESULTS)).unwrap_or_else(|| {
            eprintln!(
                "Error: Failed to initialize GPU. Ensure you have a compatible graphics card and drivers installed."
            );
            exit(1);
        })
    });

    // With `--start 0`, start from the first unsolved length.
    let start = if cli.start >= 1 {
        cli.start
    } else {
        let first = (1..=u8::MAX).find(|i| !state.solutions.get(i).is_some_and(|s| s.completed));
        first.unwrap_or_else(|| {
            if !cli.quiet {
                println!("Every length up to {} is already solved.", u8::MAX);
            }
            if cli.json {
                let _ = state.print_json();
            }
            exit(0);
        })
    };

    run(
        state,
        &cli,
        start,
        save_path.map(String::as_str),
        gpu.as_ref(),
    );
}

/// Loads a save file, asking before using one from another version.
fn load_state(path: &str) -> State {
    let mut state = State::load(path).unwrap_or_else(|| {
        eprintln!(
            "Error: Failed to load state from '{path}'. The file may not exist or is invalid."
        );
        exit(1);
    });
    if state.version != VERSION {
        eprintln!(
            "Warning: The save file version ({}) differs from the program version ({VERSION}).",
            state.version
        );
        eprintln!("Save files from different versions may not be compatible.");
        print!("Would you like to proceed with loading? (y/N): ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        if !input.trim().eq_ignore_ascii_case("y") {
            println!("Loading cancelled.");
            exit(0);
        }
        state.version = VERSION.to_string();
    }
    state
}

/// Searches each unsolved length from `start` to `cli.end`, trying more segments until a
/// ruler is found, and saves the state.
fn run(mut state: State, cli: &Cli, start: u8, save_path: Option<&str>, gpu: Option<&GpuContext>) {
    let progress = Progress::new(cli.quiet);
    ctrlc::set_handler(|| {
        eprintln!("\nInterrupt signal received. Gracefully shutting down...");
        RUNNING.store(false, Ordering::SeqCst);
    })
    .expect("Error setting Ctrl-C handler");

    let mut last_save = Instant::now();
    for length in start..=cli.end {
        if !RUNNING.load(Ordering::SeqCst) {
            break;
        }
        if state.solutions.get(&length).is_some_and(|s| s.completed) {
            continue;
        }
        let mut num_segments = state
            .solutions
            .get(&length)
            .map_or_else(|| num_segments_lower_bound(length), |s| s.num_segments);

        while RUNNING.load(Ordering::SeqCst) {
            let ctx = SearchContext {
                length,
                num_segments,
                save_path,
                running: &RUNNING,
                progress: &progress,
                gpu,
            };
            let status = search_length(&mut state, &ctx);
            if status != SearchStatus::Finished {
                progress.clear();
                stop(&mut state, status, &ctx, cli.quiet);
            }

            let solution = state.solutions.get_mut(&length).unwrap();
            if solution.rulers.is_empty() {
                num_segments += 1;
                continue;
            }
            solution.completed = true;
            let (found, segments, time) = (
                solution.rulers_found,
                solution.num_segments,
                solution.total_clock_time,
            );
            state.recalculate_global_metrics();
            // Save finished lengths every so often, so a crash loses little work.
            if last_save.elapsed() >= CHECKPOINT_INTERVAL {
                if let Err(e) = state.save(save_path, None, false) {
                    eprintln!("Warning: failed to save state: {e}");
                }
                last_save = Instant::now();
            }
            if !cli.quiet {
                progress.suspend(|| {
                    println!(
                        "✔ Length {length}: Found {found} {} with {segments} {} in {time:?}",
                        if found == 1 { "ruler" } else { "rulers" },
                        if segments == 1 { "segment" } else { "segments" },
                    );
                });
            }
            break;
        }
    }

    progress.clear();
    if !RUNNING.load(Ordering::SeqCst) && !cli.quiet {
        println!("\nSearch interrupted. Saving final state...");
    }
    state.recalculate_global_metrics();
    if let Err(e) = state.save(save_path, None, true) {
        let destination = save_path.unwrap_or("stdout");
        eprintln!("Error: Failed to save state to '{destination}': {e}");
        exit(1);
    }
    if cli.json {
        let _ = state.print_json();
    }
}

/// Saves and exits after a search that did not finish.
fn stop(state: &mut State, status: SearchStatus, ctx: &SearchContext, quiet: bool) -> ! {
    match status {
        SearchStatus::GpuFailed => eprintln!("\nGPU search failed, saving and exiting..."),
        SearchStatus::Unsupported => eprintln!(
            "\nError: cannot search length {} with {} segments (the search supports 2 to \
             {MAX_MARKS}), saving and exiting...",
            ctx.length, ctx.num_segments
        ),
        SearchStatus::Interrupted if !quiet => {
            println!("\nReceived interrupt, saving and exiting...")
        }
        _ => {}
    }
    if let Err(e) = state.save(ctx.save_path, None, true) {
        let destination = ctx.save_path.unwrap_or("stdout");
        eprintln!("Error: Failed to save state to '{destination}': {e}");
    }
    exit(if status == SearchStatus::Interrupted {
        0
    } else {
        1
    });
}
