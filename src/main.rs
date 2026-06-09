use clap::Parser;
use cpu_time::ProcessTime;
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressState, ProgressStyle};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::ser::{Formatter, PrettyFormatter, Serializer};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

mod gpu;
use gpu::{GpuContext, SearchParams};

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

#[derive(Serialize, Deserialize)]
struct Solution {
    #[serde(default)]
    completed: bool,
    lower_bound_num_segments: u8,
    num_segments: u8,
    rulers: Vec<Vec<u8>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    checkpoint_ruler: Option<Vec<u8>>,
    rulers_found: u64,
    total_rulers_evaluated: u64,
    total_clock_time: Duration,
    total_cpu_time: Duration,
}

#[derive(PartialEq, Copy, Clone)]
enum SearchStatus {
    Finished,
    Interrupted,
}

struct SearchProgress {
    status_pb: ProgressBar,
    progress_pb: ProgressBar,
    stats_pb: ProgressBar,
    latest_ruler: Arc<Mutex<Option<Vec<u8>>>>,
}

struct SearchContext<'a> {
    length: u8,
    num_segments: u8,
    save_path: Option<&'a str>,
    interrupt: &'a AtomicBool,
    progress: &'a SearchProgress,
    gpu: Option<&'a GpuContext>,
}

struct SearchTiming<'a> {
    base_clock_time: Duration,
    base_cpu_time: Duration,
    clock_start: &'a Instant,
    cpu_start: &'a ProcessTime,
}

struct RangeContext<'a> {
    length: u8,
    num_segments: u8,
    interrupt: &'a AtomicBool,
    eval_counter: &'a AtomicU64,
    found_counter: &'a AtomicU64,
    latest_ruler: &'a Arc<Mutex<Option<Vec<u8>>>>,
}

#[derive(Serialize, Deserialize)]
struct State {
    #[serde(default = "default_version")]
    version: String,
    lengths_solved: u8,
    total_rulers_found: u64,
    total_rulers_evaluated: u64,
    total_clock_time: Duration,
    total_cpu_time: Duration,
    solutions: HashMap<u8, Solution>,
}

impl State {
    fn recalculate_global_metrics(&mut self) {
        self.total_rulers_evaluated = 0;
        self.total_rulers_found = 0;
        self.total_clock_time = Duration::ZERO;
        self.total_cpu_time = Duration::ZERO;
        self.lengths_solved = 0;

        for solution in self.solutions.values() {
            self.total_rulers_evaluated += solution.total_rulers_evaluated;
            self.total_rulers_found += solution.rulers_found;
            self.total_clock_time += solution.total_clock_time;
            self.total_cpu_time += solution.total_cpu_time;
            if solution.completed {
                self.lengths_solved += 1;
            }
        }
    }

    /// Systematically generates and evaluates all possible sparse ruler configurations
    /// for a given length and number of segments using multi-threaded parallel search.
    ///
    /// The search space is partitioned into chunks and distributed across a thread pool.
    /// Progress is tracked via a "low-water mark" to ensure robustness during interruptions.
    fn find_rulers(&mut self, ctx: &SearchContext) -> SearchStatus {
        if ctx.length == 1 && ctx.num_segments == 1 {
            let solution = self.solutions.entry(1).or_insert(Solution {
                completed: true,
                lower_bound_num_segments: 1,
                num_segments: 1,
                rulers: vec![vec![1]],
                checkpoint_ruler: None,
                rulers_found: 1,
                total_rulers_evaluated: 1,
                total_clock_time: Duration::ZERO,
                total_cpu_time: Duration::ZERO,
            });
            solution.completed = true;
            solution.rulers = vec![vec![1]];
            solution.rulers_found = 1;
            solution.total_rulers_evaluated = 1;
            self.recalculate_global_metrics();
            return SearchStatus::Finished;
        }

        let (start_rank, mut found_rulers, base_clock_time, base_cpu_time) =
            self.initialize_solution(ctx);
        let start_found_count = found_rulers.len() as u64;

        if !found_rulers.is_empty() {
            let mut latest = ctx.progress.latest_ruler.lock().unwrap();
            *latest = Some(found_rulers.last().unwrap().clone());
        } else {
            let mut latest = ctx.progress.latest_ruler.lock().unwrap();
            *latest = None;
        }

        let total_space = calculate_total_space(ctx.length, ctx.num_segments) as u64;
        self.setup_progress_bars(
            ctx,
            start_rank,
            total_space,
            found_rulers.len(),
            base_clock_time,
        );

        let eval_counter = Arc::new(AtomicU64::new(start_rank));
        let found_counter = Arc::new(AtomicU64::new(start_found_count));
        let mut last_checkpoint = Instant::now();
        let length_clock_start = Instant::now();
        let length_cpu_start = ProcessTime::now();

        const CHUNK_SIZE: u64 = 100_000_000;
        const UI_UPDATE_INTERVAL: Duration = Duration::from_millis(50);
        const CHECKPOINT_INTERVAL: Duration = Duration::from_secs(10);

        let mut current_rank = start_rank;
        let mut status = SearchStatus::Finished;

        let timing = SearchTiming {
            base_clock_time,
            base_cpu_time,
            clock_start: &length_clock_start,
            cpu_start: &length_cpu_start,
        };

        // UI Thread for smooth updates
        let ui_stop = Arc::new(AtomicBool::new(false));
        let ui_stop_clone = ui_stop.clone();
        let eval_counter_ui = eval_counter.clone();
        let found_counter_ui = found_counter.clone();
        let progress_pb_ui = ctx.progress.progress_pb.clone();
        let stats_pb_ui = ctx.progress.stats_pb.clone();
        let status_pb_ui = ctx.progress.status_pb.clone();
        let latest_ruler_ui = ctx.progress.latest_ruler.clone();
        let length_ui = ctx.length;
        let num_segments_ui = ctx.num_segments;

        let ui_thread = std::thread::spawn(move || {
            while !ui_stop_clone.load(Ordering::Relaxed) {
                let total_evaluated = eval_counter_ui.load(Ordering::Relaxed);
                let total_found = found_counter_ui.load(Ordering::Relaxed);
                let elapsed = base_clock_time + length_clock_start.elapsed();

                progress_pb_ui.set_position(total_evaluated);
                stats_pb_ui.set_position(total_evaluated);

                let ruler_str = if let Ok(latest) = latest_ruler_ui.try_lock() {
                    if let Some(r) = latest.as_ref() {
                        format!(" | Latest: {:?}", r)
                    } else {
                        "".to_string()
                    }
                } else {
                    "".to_string()
                };

                status_pb_ui.set_message(format!(
                    "{}: Found {} rulers with {} segments in {:?}",
                    length_ui, total_found, num_segments_ui, elapsed
                ));
                stats_pb_ui.set_message(ruler_str);

                std::thread::sleep(UI_UPDATE_INTERVAL);
            }
        });

        let range_ctx = RangeContext {
            length: ctx.length,
            num_segments: ctx.num_segments,
            interrupt: ctx.interrupt,
            eval_counter: &eval_counter,
            found_counter: &found_counter,
            latest_ruler: &ctx.progress.latest_ruler,
        };

        while current_rank < total_space {
            if !ctx.interrupt.load(Ordering::SeqCst) {
                status = SearchStatus::Interrupted;
                break;
            }

            let (new_current_rank, all_completed) = if let Some(gpu) = ctx.gpu {
                let chunk_end = (current_rank + CHUNK_SIZE).min(total_space);
                let (mut chunk_found, completed) =
                    gpu_search_range(&range_ctx, gpu, current_rank, chunk_end);
                found_rulers.append(&mut chunk_found);
                (chunk_end, completed)
            } else {
                let num_threads = rayon::current_num_threads() as u64;
                let batch_size = num_threads * 4;
                let batch_end = (current_rank + batch_size * CHUNK_SIZE).min(total_space);
                let chunk_starts: Vec<u64> = (current_rank..batch_end)
                    .step_by(CHUNK_SIZE as usize)
                    .collect();

                let results: Vec<(Vec<Vec<u8>>, bool)> = chunk_starts
                    .into_par_iter()
                    .map(|chunk_start| {
                        let chunk_end = (chunk_start + CHUNK_SIZE).min(total_space);
                        search_range(&range_ctx, chunk_start, chunk_end)
                    })
                    .collect();

                self.process_search_results(
                    &mut found_rulers,
                    results,
                    current_rank,
                    batch_end,
                    CHUNK_SIZE,
                )
            };

            current_rank = new_current_rank;

            if !all_completed {
                status = SearchStatus::Interrupted;
                break;
            }

            let now = Instant::now();
            if now.duration_since(last_checkpoint) >= CHECKPOINT_INTERVAL {
                self.save_checkpoint(
                    ctx,
                    current_rank,
                    eval_counter.load(Ordering::Relaxed),
                    &found_rulers,
                    &timing,
                );
                last_checkpoint = now;
            }
        }

        ui_stop.store(true, Ordering::Relaxed);
        let _ = ui_thread.join();

        self.finalize_solution(
            ctx,
            current_rank,
            eval_counter.load(Ordering::Relaxed),
            found_rulers,
            &timing,
            status,
        );

        status
    }

    fn initialize_solution(
        &mut self,
        ctx: &SearchContext,
    ) -> (u64, Vec<Vec<u8>>, Duration, Duration) {
        let lb = get_num_segments_lower_bound(ctx.length);
        let solution = self.solutions.entry(ctx.length).or_insert(Solution {
            completed: false,
            lower_bound_num_segments: lb,
            num_segments: ctx.num_segments,
            rulers: vec![],
            checkpoint_ruler: None,
            rulers_found: 0,
            total_rulers_evaluated: 0,
            total_clock_time: Duration::ZERO,
            total_cpu_time: Duration::ZERO,
        });

        if !solution.completed && solution.num_segments != ctx.num_segments {
            solution.num_segments = ctx.num_segments;
            solution.total_rulers_evaluated = 0;
            solution.checkpoint_ruler = None;
        }

        let rank = if let Some(r) = solution.checkpoint_ruler.take() {
            calculate_rank(ctx.length, ctx.num_segments, &r) as u64
        } else {
            solution.total_rulers_evaluated
        };

        (
            rank,
            std::mem::take(&mut solution.rulers),
            solution.total_clock_time,
            solution.total_cpu_time,
        )
    }

    fn setup_progress_bars(
        &self,
        ctx: &SearchContext,
        start_rank: u64,
        total_space: u64,
        found_count: usize,
        base_clock_time: Duration,
    ) {
        ctx.progress.progress_pb.set_length(total_space);
        ctx.progress.stats_pb.set_length(total_space);
        ctx.progress.progress_pb.set_position(start_rank);
        ctx.progress.stats_pb.set_position(start_rank);

        ctx.progress.status_pb.set_message(format!(
            "{}: Found {} rulers with {} segments in {:?}",
            ctx.length, found_count, ctx.num_segments, base_clock_time
        ));
    }

    fn process_search_results(
        &self,
        found_rulers: &mut Vec<Vec<u8>>,
        results: Vec<(Vec<Vec<u8>>, bool)>,
        current_rank: u64,
        batch_end: u64,
        chunk_size: u64,
    ) -> (u64, bool) {
        let mut all_completed = true;
        let mut first_incomplete_rank = batch_end;

        for (i, (mut chunk_found, completed)) in results.into_iter().enumerate() {
            let chunk_start = current_rank + (i as u64 * chunk_size);
            if all_completed {
                found_rulers.append(&mut chunk_found);
                if !completed {
                    all_completed = false;
                    first_incomplete_rank = chunk_start;
                }
            }
        }

        (first_incomplete_rank, all_completed)
    }

    fn save_checkpoint(
        &mut self,
        ctx: &SearchContext,
        current_rank: u64,
        total_evaluated: u64,
        found_rulers: &[Vec<u8>],
        timing: &SearchTiming,
    ) {
        let (found, time) = {
            let solution = self.solutions.get_mut(&ctx.length).unwrap();
            solution.checkpoint_ruler =
                Some(unrank(ctx.length, ctx.num_segments, current_rank as f64));
            solution.total_rulers_evaluated = total_evaluated;
            solution.rulers = found_rulers.to_owned();
            solution.rulers_found = solution.rulers.len() as u64;
            solution.total_clock_time = timing.base_clock_time + timing.clock_start.elapsed();
            solution.total_cpu_time = timing.base_cpu_time + timing.cpu_start.elapsed();
            (solution.rulers_found, solution.total_clock_time)
        };
        self.recalculate_global_metrics();
        let _ = save_state(self, ctx.save_path);

        ctx.progress.status_pb.set_message(format!(
            "{}: Found {} rulers with {} segments in {:?}",
            ctx.length, found, ctx.num_segments, time
        ));
    }

    fn finalize_solution(
        &mut self,
        ctx: &SearchContext,
        current_rank: u64,
        total_evaluated: u64,
        found_rulers: Vec<Vec<u8>>,
        timing: &SearchTiming,
        status: SearchStatus,
    ) {
        let solution = self.solutions.get_mut(&ctx.length).unwrap();
        solution.total_rulers_evaluated = total_evaluated;
        solution.rulers = found_rulers;
        solution.rulers_found = solution.rulers.len() as u64;
        solution.total_clock_time = timing.base_clock_time + timing.clock_start.elapsed();
        solution.total_cpu_time = timing.base_cpu_time + timing.cpu_start.elapsed();

        if status == SearchStatus::Interrupted {
            solution.checkpoint_ruler =
                Some(unrank(ctx.length, ctx.num_segments, current_rank as f64));
        } else {
            solution.checkpoint_ruler = None;
        }

        self.recalculate_global_metrics();
    }
}

fn gpu_search_range(
    ctx: &RangeContext,
    gpu: &GpuContext,
    start_rank: u64,
    end_rank: u64,
) -> (Vec<Vec<u8>>, bool) {
    let mut current_rank = start_rank;
    let mut found_rulers = Vec::new();

    const STEPS_PER_THREAD: u32 = 2048;
    const THREADS_PER_BATCH: u32 = 131072;
    let total_steps_per_batch = (STEPS_PER_THREAD as u64) * (THREADS_PER_BATCH as u64);

    let mut in_flight = std::collections::VecDeque::new();
    const MAX_IN_FLIGHT: usize = 2;

    while current_rank < end_rank || !in_flight.is_empty() {
        while current_rank < end_rank && in_flight.len() < MAX_IN_FLIGHT {
            if !ctx.interrupt.load(Ordering::SeqCst) {
                break;
            }

            let remaining = end_rank - current_rank;
            let this_batch_threads = if remaining >= total_steps_per_batch {
                THREADS_PER_BATCH
            } else {
                remaining.div_ceil(STEPS_PER_THREAD as u64) as u32
            };

            let params = SearchParams {
                length: ctx.length as u32,
                num_segments: ctx.num_segments as u32,
                batch_size: this_batch_threads,
                start_rank_low: (current_rank & 0xFFFFFFFF) as u32,
                start_rank_high: (current_rank >> 32) as u32,
                steps_per_thread: STEPS_PER_THREAD,
                _padding: [0; 2],
            };

            let task = gpu.submit_search(&params);
            let processed = (this_batch_threads as u64) * (STEPS_PER_THREAD as u64);
            in_flight.push_back((task, processed.min(remaining)));
            current_rank += processed.min(remaining);
        }

        if let Some((task, batch_processed)) = in_flight.pop_front() {
            let found_ranks = gpu.wait_for_search(task);
            for rank in found_ranks {
                let r = unrank(ctx.length, ctx.num_segments, rank as f64);
                if is_canonical(&r) {
                    {
                        let mut latest = ctx.latest_ruler.lock().unwrap();
                        *latest = Some(r.clone());
                    }
                    found_rulers.push(r);
                    ctx.found_counter.fetch_add(1, Ordering::Relaxed);
                }
            }
            ctx.eval_counter
                .fetch_add(batch_processed, Ordering::Relaxed);
        }

        if !ctx.interrupt.load(Ordering::SeqCst) && current_rank < end_rank {
            break;
        }
    }

    found_rulers.sort();
    found_rulers.dedup();

    (found_rulers, current_rank >= end_rank)
}

/// Searches a specific range of ruler ranks for complete sparse rulers.
///
/// Returns a tuple containing:
/// 1. The list of found rulers.
/// 2. The number of evaluations performed.
/// 3. A boolean indicating if the range was fully searched (true) or interrupted (false).
fn search_range(ctx: &RangeContext, start_rank: u64, end_rank: u64) -> (Vec<Vec<u8>>, bool) {
    let n = ctx.num_segments as usize;
    let mut ruler = unrank(ctx.length, ctx.num_segments, start_rank as f64);
    let mut current_rank = start_rank;
    let mut found_rulers = Vec::new();

    // Pre-calculate constants for is_complete
    let l_shift = ctx.length as usize;
    let u64_blocks = l_shift >> 6;
    let bit_shift = l_shift & 63;
    let final_mask = if (ctx.length as usize) & 63 > 0 {
        (1u64 << ((ctx.length as usize) & 63)) - 1
    } else {
        0
    };

    let mut local_evals = 0;
    let mut last_eval_update = 0;
    const COUNTER_UPDATE_INTERVAL: u64 = 10_000;
    const INTERRUPT_CHECK_INTERVAL: u64 = 1_024;

    while current_rank < end_rank {
        // Efficiency: Periodic interrupt check to minimize atomic overhead
        if local_evals & (INTERRUPT_CHECK_INTERVAL - 1) == 0
            && !ctx.interrupt.load(Ordering::Relaxed)
        {
            ctx.eval_counter
                .fetch_add(local_evals - last_eval_update, Ordering::Relaxed);
            return (found_rulers, false);
        }

        // Symmetry Breaking: Full lexicographical comparison to break all rotation and reflection symmetries.
        if is_canonical(&ruler) {
            if is_complete(&ruler, u64_blocks, bit_shift, final_mask) {
                found_rulers.push(ruler.clone());
                ctx.found_counter.fetch_add(1, Ordering::Relaxed);
                if let Ok(mut latest) = ctx.latest_ruler.try_lock() {
                    *latest = Some(ruler.clone());
                }
            }
            current_rank += 1;
        } else {
            current_rank += 1;

            // Symmetry-Breaking Skip: If ruler[1] > ruler[n-1], we can skip the remaining
            // combinations for the current s_1, s_2, ..., s_{n-2} values.
            if n >= 3 && ruler[1] > ruler[n - 1] && ruler[n - 1] > 1 {
                let skip = (ruler[n - 1] - 1) as u64;
                let next_rank = current_rank + skip;

                if next_rank <= end_rank {
                    current_rank = next_rank;
                    local_evals += skip;
                    ruler[n - 2] += ruler[n - 1] - 1;
                    ruler[n - 1] = 1;
                } else {
                    // Cannot skip beyond the assigned range; cap at end_rank
                    let actual_skip = end_rank - current_rank;
                    local_evals += actual_skip;
                    ctx.eval_counter
                        .fetch_add(local_evals - last_eval_update, Ordering::Relaxed);
                    return (found_rulers, true);
                }
            }
        }

        local_evals += 1;
        if local_evals - last_eval_update >= COUNTER_UPDATE_INTERVAL {
            ctx.eval_counter
                .fetch_add(local_evals - last_eval_update, Ordering::Relaxed);
            last_eval_update = local_evals;
        }

        if current_rank >= end_rank {
            break;
        }

        // Permutation Logic (Odometer-style)
        if n > 2 && ruler[n - 1] > 1 {
            ruler[n - 1] -= 1;
            ruler[n - 2] += 1;
        } else {
            let mut all_attempted = false;
            for i in (1..n - 1).rev() {
                if i == 1 {
                    all_attempted = true;
                    break;
                }
                if ruler[i] == 1 {
                    continue;
                } else {
                    ruler[i] = 1;
                    ruler[i - 1] += 1;
                    let current_sum: u16 = ruler[0..n - 1].iter().map(|&x| x as u16).sum();
                    ruler[n - 1] = ctx.length - (current_sum as u8);
                    break;
                }
            }
            if all_attempted || n < 3 {
                break;
            }
        }
    }

    ctx.eval_counter
        .fetch_add(local_evals - last_eval_update, Ordering::Relaxed);
    (found_rulers, current_rank >= end_rank)
}

fn default_version() -> String {
    "0.0.0".to_string()
}

struct CompactArrayFormatter<'a> {
    inner: PrettyFormatter<'a>,
    depth: usize,
}

impl<'a> CompactArrayFormatter<'a> {
    fn new() -> Self {
        Self {
            inner: PrettyFormatter::with_indent(b"  "),
            depth: 0,
        }
    }
}

impl<'a> Formatter for CompactArrayFormatter<'a> {
    fn begin_array<W: ?Sized + Write>(&mut self, writer: &mut W) -> io::Result<()> {
        self.depth += 1;
        if self.depth > 1 {
            writer.write_all(b"[")
        } else {
            self.inner.begin_array(writer)
        }
    }

    fn end_array<W: ?Sized + Write>(&mut self, writer: &mut W) -> io::Result<()> {
        let res = if self.depth > 1 {
            writer.write_all(b"]")
        } else {
            self.inner.end_array(writer)
        };
        self.depth -= 1;
        res
    }

    fn begin_array_value<W: ?Sized + Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if self.depth > 1 {
            if !first {
                writer.write_all(b", ")?;
            }
            Ok(())
        } else {
            self.inner.begin_array_value(writer, first)
        }
    }

    fn end_array_value<W: ?Sized + Write>(&mut self, writer: &mut W) -> io::Result<()> {
        if self.depth > 1 {
            Ok(())
        } else {
            self.inner.end_array_value(writer)
        }
    }

    fn begin_object<W: ?Sized + Write>(&mut self, writer: &mut W) -> io::Result<()> {
        self.inner.begin_object(writer)
    }

    fn end_object<W: ?Sized + Write>(&mut self, writer: &mut W) -> io::Result<()> {
        self.inner.end_object(writer)
    }

    fn begin_object_key<W: ?Sized + Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        self.inner.begin_object_key(writer, first)
    }

    fn end_object_key<W: ?Sized + Write>(&mut self, writer: &mut W) -> io::Result<()> {
        self.inner.end_object_key(writer)
    }

    fn begin_object_value<W: ?Sized + Write>(&mut self, writer: &mut W) -> io::Result<()> {
        self.inner.begin_object_value(writer)
    }

    fn end_object_value<W: ?Sized + Write>(&mut self, writer: &mut W) -> io::Result<()> {
        self.inner.end_object_value(writer)
    }
}

fn save_state(state: &State, path: Option<&str>) -> std::io::Result<()> {
    if let Some(path) = path {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        let mut ser = Serializer::with_formatter(writer, CompactArrayFormatter::new());
        state.serialize(&mut ser).map_err(std::io::Error::other)?;
    }
    Ok(())
}

fn print_state(state: &State) -> std::io::Result<()> {
    let mut ser = Serializer::with_formatter(io::stdout(), CompactArrayFormatter::new());
    state.serialize(&mut ser).map_err(std::io::Error::other)?;
    println!();
    Ok(())
}

fn load_state(path: &str) -> Option<State> {
    let file = File::open(path).ok()?;
    let reader = std::io::BufReader::new(file);
    serde_json::from_reader(reader).ok()
}

/// Calculate the lower bound of the required number of segments for a sparse
/// ruler. Where n is the number of segments a sparse ruler has up to
/// n^2 - n + 1 unique lengths--or could potentially create a sparse ruler up to
/// that length. Solving this equation for length (and taking the ceiling of the
/// result) allows for finding the lower bound on the required number of
/// segments for a sparse ruler. In other words,
/// n = ceil((sqrt(4l - 3) + 1) / 2)
fn get_num_segments_lower_bound(length: u8) -> u8 {
    (((length as f64 * 4.0 - 3.0).sqrt() + 1.0) / 2.0).ceil() as u8
}

/// Calculate the total number of possible rulers for a given length and number
/// of segments. This uses the "Stars and Bars". For a ruler of length L with n
/// segments where the first segment is fixed at 1, we are looking for the
/// number of compositions of (L-1) into (n-1) parts, where each part s_i >= 1.
/// The formula is binom((L-1) - 1, (n-1) - 1) = binom(L-2, n-2).
fn calculate_total_space(length: u8, num_segments: u8) -> f64 {
    if num_segments < 2 || length < num_segments {
        return 0.0;
    }
    binomial(length as u64 - 2, num_segments as u64 - 2)
}

/// Calculate the lexicographical rank of a given ruler in the search space.
///
/// This implements a ranking algorithm for compositions. The rank is determined by
/// summing the number of compositions that would appear before the current one
/// in a lexicographical sort.
///
/// For a component s_i at position k, we skip all compositions where the value
/// at this position is j < s_i. The number of such compositions is:
/// Sum_{j=1}^{s_i-1} binom((remaining_sum - j) - 1, (remaining_parts - 1) - 1)
/// Using the Hockey-stick identity, this sum simplifies to:
/// binom(remaining_sum - 1, remaining_parts - 1) - binom(remaining_sum - s_i, remaining_parts - 1)
fn calculate_rank(length: u8, num_segments: u8, segments: &[u8]) -> f64 {
    if num_segments < 2 {
        return 0.0;
    }
    let s = length as u64 - 1;
    let k = num_segments as u64 - 1;
    let mut current_s = s;
    let mut current_k = k;
    let mut rank = 0.0;

    for &val in &segments[1..num_segments as usize - 1] {
        let val = val as u64;
        rank += binomial(current_s - 1, current_k - 1) - binomial(current_s - val, current_k - 1);
        current_s -= val;
        current_k -= 1;
    }
    rank
}

/// Computes the composition (ruler) corresponding to a given lexicographical rank.
///
/// This is the inverse of `calculate_rank`. It reconstructs the sequence of
/// segment lengths by iteratively determining how many compositions start with
/// each possible value for the current segment.
fn unrank(length: u8, num_segments: u8, mut rank: f64) -> Vec<u8> {
    let mut segments = vec![1u8; num_segments as usize];
    segments[0] = 1;
    let mut current_s = length as u64 - 1;
    let mut current_k = num_segments as u64 - 1;

    for segment in segments.iter_mut().take(num_segments as usize - 1).skip(1) {
        let mut val = 1;
        loop {
            if current_s <= val {
                break;
            }
            let n_binom = current_s - val - 1;
            let k_binom = current_k - 2;
            let count = binomial(n_binom, k_binom);

            if rank < count {
                break;
            }
            rank -= count;
            val += 1;
        }
        *segment = val as u8;
        current_s -= val;
        current_k -= 1;
    }
    segments[num_segments as usize - 1] = current_s as u8;
    segments
}

/// Calculate binomial coefficient (n choose k) using the multiplicative formula.
/// res = product_{i=1}^{k} (n - i + 1) / i
fn binomial(n: u64, k: u64) -> f64 {
    if k > n {
        return 0.0;
    }
    if k == 0 || k == n {
        return 1.0;
    }
    let k = k.min(n - k);
    let mut res = 1.0;
    for i in 1..=k {
        res = res * (n - i + 1) as f64 / i as f64;
    }
    res
}

/// Checks if a ruler is "canonical" (the lexicographical minimum among all its rotations and reflections).
///
/// A circular ruler can be represented by many different segment sequences depending on
/// which segment you start with and which direction you go (rotation and reflection).
/// To avoid duplicate solutions, we only accept the lexicographical minimum of all
/// 2N possible representations (N rotations and N reflections).
///
/// Since our search only explores compositions where s_0 = 1, and 1 is the minimum
/// possible segment value, we only need to compare the current ruler with other
/// rotations and reflections that also start with 1.
fn is_canonical(ruler: &[u8]) -> bool {
    let n = ruler.len();
    if n < 2 {
        return true;
    }

    // 1. Check other rotations starting with 1.
    // If any rotation is lexicographically smaller than the current ruler,
    // then the current ruler is not canonical.
    for i in 1..n {
        if ruler[i] == 1 {
            for j in 0..n {
                let a = ruler[j];
                let b = ruler[(i + j) % n];
                if b < a {
                    return false;
                }
                if b > a {
                    break;
                }
            }
        }
    }

    // 2. Check all reflections starting with 1.
    // A reflection starting at index i (going backwards) is compared with the ruler.
    // This handles both the standard reflection (fixing s_0) and reflections
    // starting from other '1' segments.
    for i in 0..n {
        if ruler[i] == 1 {
            for j in 0..n {
                let a = ruler[j];
                let b = ruler[(i + n - j) % n];
                if b < a {
                    return false;
                }
                if b > a {
                    break;
                }
            }
        }
    }

    true
}

/// Verifies if a ruler is "complete" (can measure all lengths from 1 to L) using bitwise operations.
///
/// A ruler is complete if for every distance d in [1, L], there exist marks m1, m2 such that:
/// (m1 - m2) mod L == d OR (m2 - m1) mod L == d
///
/// This implementation uses a bitset (marks) where bit i is set if position i has a mark.
/// The set of all measurable distances is (M | (M << L)) integrated over all mark positions.
///
/// ### Parameters:
/// * `segments`: The relative lengths between consecutive marks.
/// * `u64_blocks`: Pre-calculated `L / 64`, used for block-level bitwise shifts and validation.
/// * `bit_shift`: Pre-calculated `L % 64`, used for bit-level shifts within a block.
/// * `final_mask`: A bitmask (all 1s for the first `L % 64` bits) to check the final partial block.
fn is_complete(segments: &[u8], u64_blocks: usize, bit_shift: usize, final_mask: u64) -> bool {
    let n = segments.len();
    let mut marks = [0u64; 4]; // Supports up to length 256
    let mut current_pos = 0;

    // Step 1: Populate the bitset with mark positions
    marks[0] |= 1; // First mark is always at 0
    for &s in &segments[..n - 1] {
        current_pos += s as usize;
        if current_pos < 256 {
            // Divide current_pos by 64 to index into the proper u64 in the
            // array, then set the bit at the proper position in that u64.
            marks[current_pos >> 6] |= 1 << (current_pos & 63);
        }
    }

    // Step 2: Create a virtual bitset m2 representing (M | (M << L))
    // This virtual bitset is crucial for handling circular/modular wrap-around.
    // By duplicating the marks at position L (via shifting), any modular distance
    // (m1 - m2) mod L can be found by a simple linear shift in the next step.
    // m2 is double the size of marks (8 * 64 bits) to accommodate the shift.
    let mut m2 = [0u64; 8];
    for i in 0..4 {
        // Copy original marks
        m2[i] |= marks[i];

        // Shift and OR the marks into the higher blocks of m2 to represent (M << L)
        if bit_shift == 0 {
            // Optimization: L is a perfect multiple of 64
            if i + u64_blocks < 8 {
                m2[i + u64_blocks] |= marks[i];
            }
        } else {
            // General case: L shift spans across block boundaries
            if i + u64_blocks < 8 {
                m2[i + u64_blocks] |= marks[i] << bit_shift;
            }
            if i + u64_blocks + 1 < 8 {
                m2[i + u64_blocks + 1] |= marks[i] >> (64 - bit_shift);
            }
        }
    }

    // Step 3: Calculate measurable distances (differences between marks)
    // We compute the union of the bitset shifted by every mark position.
    // Mathematically, if bit 'd' is set in the result, it means there exists
    // a pair of marks (m1, m2) such that (m1 - m2) mod L == d.
    // By shifting the virtual bitset m2 (which contains duplicated marks at +L),
    // we correctly capture wrap-around distances in a single linear pass.
    let mut diffs = [0u64; 4];
    current_pos = 0;

    // Shift for mark at position 0 (Identity shift)
    for j in 0..4 {
        diffs[j] |= m2[j];
    }

    // Shift for all other mark positions
    for &s in &segments[..n - 1] {
        current_pos += s as usize;
        let u_shift = current_pos >> 6; // Number of full 64-bit blocks to shift
        let b_shift = current_pos & 63; // Number of bits to shift within a block

        if b_shift == 0 {
            // Optimization: Aligning on a 64-bit boundary allows a direct OR
            for j in 0..4 {
                diffs[j] |= m2[j + u_shift];
            }
        } else {
            // General case: The shift spans across two adjacent u64 blocks in m2.
            // We extract the relevant bits from the 'current' and 'next' blocks.
            for j in 0..4 {
                diffs[j] |= (m2[j + u_shift] >> b_shift) | (m2[j + u_shift + 1] << (64 - b_shift));
            }
        }
    }

    // Step 4: Verify that all bits from 1 to L-1 are set
    // A ruler is complete if every bit corresponding to a distance [1, L-1] is 1.
    // We first check all full 64-bit blocks for speed (must be all 1s).
    if diffs.iter().take(u64_blocks).any(|&d| d != !0) {
        return false;
    }

    // Finally, check the last partial block if L is not a multiple of 64.
    // The final_mask ensures we only look at bits up to L-1 and ignore padding.
    !(final_mask > 0 && (diffs[u64_blocks] & final_mask) != final_mask)
}

/// Orchestrates the execution of the sparse ruler search across a range of lengths.
fn execute(
    mut state: State,
    save_path: Option<String>,
    start_length: u8,
    end_length: u8,
    output_json: bool,
    quiet: bool,
    gpu: Option<&GpuContext>,
) {
    let mp = MultiProgress::with_draw_target(if quiet {
        ProgressDrawTarget::hidden()
    } else {
        ProgressDrawTarget::stdout()
    });

    let status_pb = mp.add(ProgressBar::new_spinner());
    status_pb.enable_steady_tick(Duration::from_millis(100));
    status_pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} Length {msg}")
            .unwrap(),
    );

    let progress_pb = mp.add(ProgressBar::new(0));
    progress_pb.enable_steady_tick(Duration::from_millis(100));
    progress_pb.set_style(
        ProgressStyle::default_bar()
            .template("  [{bar:40.cyan/blue}] {pos}/{len} ({percent}%)")
            .unwrap()
            .progress_chars("#>-"),
    );

    let stats_pb = mp.add(ProgressBar::new(0));
    stats_pb.enable_steady_tick(Duration::from_millis(100));
    stats_pb.set_style(
        ProgressStyle::default_spinner()
            .template("  {elapsed_precise} [{evals_per_sec} evals/s]{msg}")
            .unwrap()
            .with_key(
                "evals_per_sec",
                |state: &ProgressState, w: &mut dyn std::fmt::Write| {
                    let n = state.per_sec() as u64;
                    let s = n.to_string();
                    let len = s.len();
                    for (i, c) in s.chars().enumerate() {
                        if i > 0 && (len - i) % 3 == 0 {
                            let _ = w.write_char(',');
                        }
                        let _ = w.write_char(c);
                    }
                },
            ),
    );

    let progress = SearchProgress {
        status_pb,
        progress_pb,
        stats_pb,
        latest_ruler: Arc::new(Mutex::new(None)),
    };

    let interrupt = Arc::new(AtomicBool::new(true));
    let r = interrupt.clone();

    ctrlc::set_handler(move || {
        eprintln!("\nInterrupt signal received. Gracefully shutting down...");
        r.store(false, Ordering::SeqCst);
    })
    .expect("Error setting Ctrl-C handler");

    for i in start_length..=end_length {
        if !interrupt.load(Ordering::SeqCst) {
            break;
        }
        if state.solutions.get(&i).is_some_and(|s| s.completed) {
            continue;
        }

        let mut num_segments = state
            .solutions
            .get(&i)
            .map(|s| s.num_segments)
            .unwrap_or_else(|| get_num_segments_lower_bound(i));

        loop {
            if !interrupt.load(Ordering::SeqCst) {
                break;
            }
            let ctx = SearchContext {
                length: i,
                num_segments,
                save_path: save_path.as_deref(),
                interrupt: &interrupt,
                progress: &progress,
                gpu,
            };

            let status = state.find_rulers(&ctx);
            match status {
                SearchStatus::Finished => {
                    let (found, segments, time) = {
                        let solution = state.solutions.get_mut(&i).unwrap();
                        if !solution.rulers.is_empty() {
                            solution.completed = true;
                            (
                                solution.rulers_found,
                                solution.num_segments,
                                solution.total_clock_time,
                            )
                        } else {
                            (0, 0, Duration::ZERO)
                        }
                    };

                    if found > 0 {
                        state.recalculate_global_metrics();
                        if !quiet {
                            mp.suspend(|| {
                                println!(
                                    "✔ Length {}: Found {} {} with {} {} in {:?}",
                                    i,
                                    found,
                                    if found == 1 { "ruler" } else { "rulers" },
                                    segments,
                                    if segments == 1 { "segment" } else { "segments" },
                                    time
                                );
                            });
                        }
                        break;
                    }
                    num_segments += 1;
                }
                SearchStatus::Interrupted => {
                    progress.status_pb.finish_and_clear();
                    progress.progress_pb.finish_and_clear();
                    progress.stats_pb.finish_and_clear();
                    if !quiet {
                        println!("\nReceived interrupt, saving and exiting...");
                    }
                    if let Err(e) = save_state(&state, save_path.as_deref()) {
                        let destination = save_path.as_deref().unwrap_or("stdout");
                        eprintln!("Error: Failed to save state to '{}': {}", destination, e);
                    }
                    std::process::exit(0);
                }
            }
        }
    }

    if !interrupt.load(Ordering::SeqCst) {
        progress.status_pb.finish_and_clear();
        progress.progress_pb.finish_and_clear();
        progress.stats_pb.finish_and_clear();
        if !quiet {
            println!("\nSearch interrupted. Saving final state...");
        }
    } else {
        progress.status_pb.finish_and_clear();
        progress.progress_pb.finish_and_clear();
        progress.stats_pb.finish_and_clear();
    }

    state.recalculate_global_metrics();
    if let Err(e) = save_state(&state, save_path.as_deref()) {
        let destination = save_path.as_deref().unwrap_or("stdout");
        eprintln!("Error: Failed to save state to '{}': {}", destination, e);
        std::process::exit(1);
    }

    if output_json {
        let _ = print_state(&state);
    }
}

fn main() {
    let cli = Cli::parse();

    // Configure the global thread pool
    rayon::ThreadPoolBuilder::new()
        .num_threads(cli.threads)
        .build_global()
        .expect("Failed to initialize thread pool");

    let load_path = cli.load.or(cli.resume.clone());
    let save_path = cli.save.or(cli.resume);

    let current_version = env!("CARGO_PKG_VERSION");

    let mut state = if let Some(path) = load_path {
        let mut loaded_state = load_state(&path).unwrap_or_else(|| {
            eprintln!(
                "Error: Failed to load state from '{}'. The file may not exist or is invalid.",
                path
            );
            std::process::exit(1);
        });

        if loaded_state.version != current_version {
            eprintln!(
                "Warning: The save file version ({}) differs from the program version ({}).",
                loaded_state.version, current_version
            );
            eprintln!("Save files from different versions may not be compatible.");
            print!("Would you like to proceed with loading? (y/N): ");
            io::stdout().flush().unwrap();

            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();
            if !input.trim().eq_ignore_ascii_case("y") {
                println!("Loading cancelled.");
                std::process::exit(0);
            }
            loaded_state.version = current_version.to_string();
        }
        loaded_state
    } else {
        State {
            version: current_version.to_string(),
            lengths_solved: 0,
            total_rulers_found: 0,
            total_rulers_evaluated: 0,
            total_clock_time: Duration::ZERO,
            total_cpu_time: Duration::ZERO,
            solutions: HashMap::new(),
        }
    };

    state.recalculate_global_metrics();

    let gpu_ctx = if cli.gpu {
        let ctx = pollster::block_on(GpuContext::new(65536));
        if ctx.is_none() {
            eprintln!(
                "Error: Failed to initialize GPU. Ensure you have a compatible graphics card and drivers installed."
            );
            std::process::exit(1);
        }
        ctx
    } else {
        None
    };

    let start_length = if cli.start >= 1 {
        cli.start
    } else {
        // If not specified, start from the first uncompleted length
        let mut i = 1;
        while state.solutions.get(&i).is_some_and(|s| s.completed) {
            i += 1;
        }
        i
    };

    execute(
        state,
        save_path,
        start_length,
        cli.end,
        cli.json,
        cli.quiet,
        gpu_ctx.as_ref(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_canonical() {
        // [1, 1, 3, 6] is canonical
        assert!(is_canonical(&[1, 1, 3, 6]));
        // [1, 1, 6, 3] is a reflection, NOT canonical (it should be [1, 1, 3, 6])
        assert!(!is_canonical(&[1, 1, 6, 3]));
        // [1, 3, 6, 1] is a rotation, NOT canonical
        assert!(!is_canonical(&[1, 3, 6, 1]));

        // Symmetric rulers should be canonical if they are the lexicographical minimum
        assert!(is_canonical(&[1, 1, 2, 2])); // n=4, L=6
        assert!(!is_canonical(&[1, 2, 2, 1])); // [1, 1, 2, 2] is smaller
        assert!(is_canonical(&[1, 2, 1, 2])); // n=4, L=6

        // Test with multiple 1s
        assert!(is_canonical(&[1, 1, 2, 1, 1, 3]));
        assert!(!is_canonical(&[1, 1, 3, 1, 1, 2])); // Reflection is smaller
    }

    #[test]
    fn test_binomial() {
        assert_eq!(binomial(5, 2), 10.0);
        assert_eq!(binomial(10, 5), 252.0);
        assert_eq!(binomial(10, 0), 1.0);
        assert_eq!(binomial(10, 10), 1.0);
    }

    #[test]
    fn test_is_complete() {
        let length = 6;
        let u64_blocks = length >> 6;
        let bit_shift = length & 63;
        let final_mask = (1u64 << (length & 63)) - 1;

        // Length 6, 3 segments: [1, 2, 3] is a ruler for length 6
        assert!(is_complete(&[1, 2, 3], u64_blocks, bit_shift, final_mask));

        // Length 6, [2, 2, 2] is not complete
        assert!(!is_complete(&[2, 2, 2], u64_blocks, bit_shift, final_mask));

        let length = 13;
        let u64_blocks = length >> 6;
        let bit_shift = length & 63;
        let final_mask = (1u64 << (length & 63)) - 1;
        // L=13, k=4 ruler: [1, 3, 2, 7]
        // Marks: 0, 1, 4, 6.
        // Diff 1-0=1, 4-0=4, 6-0=6, 4-1=3, 6-1=5, 6-4=2.
        // Modular diffs: 0-6=7, 0-4=9, 0-1=12, 1-6=8, 4-6=11, 1-4=10.
        // All present!
        assert!(is_complete(
            &[1, 3, 2, 7],
            u64_blocks,
            bit_shift,
            final_mask
        ));
    }

    #[test]
    fn test_rank_unrank() {
        let length = 10;
        let num_segments = 4;
        let segments = vec![1, 2, 3, 4];
        let rank = calculate_rank(length, num_segments, &segments);
        let unranked = unrank(length, num_segments, rank);
        assert_eq!(segments, unranked);

        let length = 20;
        let num_segments = 6;
        let segments = vec![1, 1, 1, 1, 1, 15];
        let rank = calculate_rank(length, num_segments, &segments);
        let unranked = unrank(length, num_segments, rank);
        assert_eq!(segments, unranked);
    }

    #[test]
    fn test_get_num_segments_lower_bound() {
        assert_eq!(get_num_segments_lower_bound(6), 3);
        assert_eq!(get_num_segments_lower_bound(13), 4);
        assert_eq!(get_num_segments_lower_bound(31), 6);
    }
}
