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
mod search;
use gpu::{GpuContext, SearchParams};
use search::{Dfs, Search};

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
    GpuFailed,
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

/// The partial rulers (prefixes of `depth` marks each) that split one search.
struct WorkContext<'a> {
    search: &'a Search,
    depth: usize,
    prefixes: &'a [u8],
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

        let (checkpoint, start_evaluated, mut found_rulers, base_clock_time, base_cpu_time) =
            self.initialize_solution(ctx);
        let start_found_count = found_rulers.len() as u64;

        let length_clock_start = Instant::now();
        let length_cpu_start = ProcessTime::now();

        // Split the search tree into partial rulers (prefixes) that are searched independently.
        let search = Search::new(ctx.length, ctx.num_segments);
        let n = ctx.num_segments as usize;
        let use_gpu = ctx.gpu.is_some() && n >= 3;
        let target = if use_gpu { 1 << 21 } else { 1 << 16 };
        let (depth, prefixes) = search.prefixes(target, (n - 2).max(2), 1 << 24);
        let total_items = (prefixes.len() / depth) as u64;
        let start_item = checkpoint.map_or(0, |marks| {
            prefixes
                .chunks(depth)
                .collect::<Vec<_>>()
                .partition_point(|p| {
                    let m = p.len().min(marks.len());
                    p[..m] < marks[..m]
                }) as u64
        });

        if !found_rulers.is_empty() {
            let mut latest = ctx.progress.latest_ruler.lock().unwrap();
            *latest = Some(found_rulers.last().unwrap().clone());
        } else {
            let mut latest = ctx.progress.latest_ruler.lock().unwrap();
            *latest = None;
        }

        self.setup_progress_bars(
            ctx,
            start_item,
            total_items,
            found_rulers.len(),
            base_clock_time,
        );

        let eval_counter = Arc::new(AtomicU64::new(start_evaluated));
        let done_counter = Arc::new(AtomicU64::new(start_item));
        let found_counter = Arc::new(AtomicU64::new(start_found_count));
        let mut last_checkpoint = Instant::now();

        const UI_UPDATE_INTERVAL: Duration = Duration::from_millis(50);
        const CHECKPOINT_INTERVAL: Duration = Duration::from_secs(10);

        let mut current_item = start_item;
        let mut status = SearchStatus::Finished;
        let mut gpu_work = ctx
            .gpu
            .filter(|_| use_gpu)
            .map(|gpu| GpuWork::new(gpu, start_item, total_items));

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
        let done_counter_ui = done_counter.clone();
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

                progress_pb_ui.set_position(done_counter_ui.load(Ordering::Relaxed));
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

        let work = WorkContext {
            search: &search,
            depth,
            prefixes: &prefixes,
            interrupt: ctx.interrupt,
            eval_counter: &eval_counter,
            found_counter: &found_counter,
            latest_ruler: &ctx.progress.latest_ruler,
        };

        while current_item < total_items {
            if !ctx.interrupt.load(Ordering::SeqCst) {
                status = SearchStatus::Interrupted;
                break;
            }

            let all_completed = if let (Some(gpu), Some(gpu_work)) = (ctx.gpu, &mut gpu_work) {
                if let Err(e) = gpu_work.pass(&work, gpu, &mut found_rulers) {
                    ctx.progress.status_pb.suspend(|| eprintln!("\nError: {e}"));
                    status = SearchStatus::GpuFailed;
                    break;
                }
                current_item = gpu_work.unfinished;
                // Partial rulers finish out of order; show how many are done.
                done_counter.store(gpu_work.completed, Ordering::Relaxed);
                true
            } else {
                let batch = rayon::current_num_threads() as u64 * 64;
                let end = (current_item + batch).min(total_items);
                match cpu_search_items(&work, current_item, end, &mut found_rulers) {
                    Some(c) => {
                        current_item = c;
                        false
                    }
                    None => {
                        current_item = end;
                        true
                    }
                }
            };
            if gpu_work.is_none() {
                done_counter.store(current_item, Ordering::Relaxed);
            }

            if !all_completed {
                status = SearchStatus::Interrupted;
                break;
            }

            let now = Instant::now();
            if now.duration_since(last_checkpoint) >= CHECKPOINT_INTERVAL {
                self.save_checkpoint(
                    ctx,
                    work.checkpoint(current_item),
                    eval_counter.load(Ordering::Relaxed),
                    &found_rulers,
                    &timing,
                );
                last_checkpoint = now;
            }
        }

        ui_stop.store(true, Ordering::Relaxed);
        let _ = ui_thread.join();

        // A resumed search may repeat part of a prefix, so drop duplicates.
        found_rulers.sort();
        found_rulers.dedup();

        self.finalize_solution(
            ctx,
            work.checkpoint(current_item),
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
    ) -> (Option<Vec<u8>>, u64, Vec<Vec<u8>>, Duration, Duration) {
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

        // The checkpoint holds the segments of the first unfinished partial ruler; turn it
        // back into mark positions.
        let checkpoint = solution.checkpoint_ruler.take().map(|segments| {
            let mut marks = vec![0u8];
            for s in segments {
                marks.push(marks.last().unwrap().saturating_add(s));
            }
            marks
        });

        (
            checkpoint,
            solution.total_rulers_evaluated,
            std::mem::take(&mut solution.rulers),
            solution.total_clock_time,
            solution.total_cpu_time,
        )
    }

    fn setup_progress_bars(
        &self,
        ctx: &SearchContext,
        start_item: u64,
        total_items: u64,
        found_count: usize,
        base_clock_time: Duration,
    ) {
        ctx.progress.progress_pb.set_length(total_items);
        ctx.progress.progress_pb.set_position(start_item);

        ctx.progress.status_pb.set_message(format!(
            "{}: Found {} rulers with {} segments in {:?}",
            ctx.length, found_count, ctx.num_segments, base_clock_time
        ));
    }

    fn save_checkpoint(
        &mut self,
        ctx: &SearchContext,
        checkpoint: Option<Vec<u8>>,
        total_evaluated: u64,
        found_rulers: &[Vec<u8>],
        timing: &SearchTiming,
    ) {
        let (found, time) = {
            let solution = self.solutions.get_mut(&ctx.length).unwrap();
            solution.checkpoint_ruler = checkpoint;
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
        checkpoint: Option<Vec<u8>>,
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

        solution.checkpoint_ruler = if status == SearchStatus::Interrupted {
            checkpoint
        } else {
            None
        };

        self.recalculate_global_metrics();
    }
}

impl WorkContext<'_> {
    fn prefix(&self, item: u64) -> &[u8] {
        let i = item as usize * self.depth;
        &self.prefixes[i..i + self.depth]
    }

    /// The segments of the first unfinished partial ruler, saved so a resumed search skips
    /// every partial ruler before it.
    fn checkpoint(&self, item: u64) -> Option<Vec<u8>> {
        (item < (self.prefixes.len() / self.depth) as u64)
            .then(|| self.prefix(item).windows(2).map(|w| w[1] - w[0]).collect())
    }

    fn record(&self, rulers: &[Vec<u8>]) {
        if let Some(last) = rulers.last() {
            self.found_counter
                .fetch_add(rulers.len() as u64, Ordering::Relaxed);
            if let Ok(mut latest) = self.latest_ruler.try_lock() {
                *latest = Some(last.clone());
            }
        }
    }
}

/// Searches partial rulers `start..end` on the thread pool, appending the rulers found to
/// `found_rulers`. Returns the first item that was not finished if the search was interrupted.
fn cpu_search_items(
    ctx: &WorkContext,
    start: u64,
    end: u64,
    found_rulers: &mut Vec<Vec<u8>>,
) -> Option<u64> {
    let results: Vec<(Vec<Vec<u8>>, bool)> = (start..end)
        .into_par_iter()
        .map(|item| {
            let mut dfs = Dfs::new(ctx.search, ctx.interrupt, ctx.eval_counter);
            dfs.run(&ctx.search.node_from_marks(ctx.prefix(item)));
            dfs.flush();
            ctx.record(&dfs.found);
            (dfs.found, !dfs.interrupted)
        })
        .collect();

    for (offset, (mut found, completed)) in results.into_iter().enumerate() {
        if !completed {
            return Some(start + offset as u64);
        }
        found_rulers.append(&mut found);
    }
    None
}

/// Partial rulers being searched on the GPU. Each occupies a GPU slot, which keeps its search
/// state between passes, until it is fully searched; then the slot takes the next one.
///
/// A few partial rulers have far larger subtrees than the rest, so once every partial ruler
/// has started and most slots are idle, the slots still busy are split: what each has left to
/// search is handed out again as deeper partial rulers, which fill the idle slots.
struct GpuWork {
    /// The original partial ruler (item) that each slot's partial ruler came from.
    slot_item: Vec<Option<u64>>,
    free_slots: Vec<u32>,
    active_slots: Vec<u32>,
    next_item: u64,
    /// Partial rulers split off busy slots, waiting for a slot, with their items.
    split: Vec<(u64, Vec<u8>)>,
    /// Partial rulers of each item still in a slot or waiting for one.
    outstanding: Vec<u32>,
    done: Vec<bool>,
    /// The first partial ruler not fully searched yet.
    unfinished: u64,
    /// Partial rulers fully searched, counting those before the starting one.
    completed: u64,
    node_budget: u32,
}

impl GpuWork {
    fn new(gpu: &GpuContext, start: u64, total: u64) -> Self {
        GpuWork {
            slot_item: vec![None; gpu.max_slots as usize],
            free_slots: (0..gpu.max_slots).rev().collect(),
            active_slots: Vec::new(),
            next_item: start,
            split: Vec::new(),
            outstanding: vec![0; total as usize],
            done: vec![false; total as usize],
            unfinished: start,
            completed: start,
            node_budget: 1 << 12,
        }
    }

    /// Records that one partial ruler of `item` was fully searched (or split).
    fn finish(&mut self, item: u64) {
        self.outstanding[item as usize] -= 1;
        if self.outstanding[item as usize] == 0 {
            self.done[item as usize] = true;
            self.completed += 1;
        }
    }

    /// Fills free slots with the next partial rulers, runs one GPU pass, and collects the
    /// rulers it found.
    fn pass(
        &mut self,
        ctx: &WorkContext,
        gpu: &GpuContext,
        found_rulers: &mut Vec<Vec<u8>>,
    ) -> Result<(), String> {
        let n = ctx.search.num_segments();
        let length = ctx.search.length();
        let total = self.done.len() as u64;

        let mut work = Vec::new();
        for &slot in &self.active_slots {
            work.extend([slot, gpu::RESUME]);
        }
        // Each new partial ruler is its depth followed by its marks.
        let mut prefixes: Vec<u32> = Vec::new();
        while !self.free_slots.is_empty() {
            let (item, marks) = if let Some((item, marks)) = self.split.pop() {
                (item, marks)
            } else if self.next_item < total {
                self.outstanding[self.next_item as usize] = 1;
                self.next_item += 1;
                (self.next_item - 1, ctx.prefix(self.next_item - 1).to_vec())
            } else {
                break;
            };
            let slot = self.free_slots.pop().unwrap();
            self.slot_item[slot as usize] = Some(item);
            self.active_slots.push(slot);
            work.extend([slot, prefixes.len() as u32]);
            prefixes.push(marks.len() as u32);
            prefixes.extend(marks.iter().map(|&m| m as u32));
        }

        let params = SearchParams {
            length: length as u32,
            num_segments: n as u32,
            num_work: (work.len() / 2) as u32,
            _padding0: 0,
            words: (length as u32).div_ceil(32),
            max_results: gpu.max_results,
            node_budget: self.node_budget,
            _padding: 0,
        };
        let start = Instant::now();
        let pass = gpu.run_pass(&params, &work, &prefixes)?;
        // Keep each pass short (well under the driver's GPU timeout) but long enough that
        // launching it is cheap.
        let elapsed = start.elapsed();
        if elapsed < Duration::from_millis(100) {
            self.node_budget = (self.node_budget * 2).min(1 << 22);
        } else if elapsed > Duration::from_millis(500) {
            self.node_budget = (self.node_budget / 2).max(1 << 8);
        }

        ctx.eval_counter.fetch_add(pass.nodes, Ordering::Relaxed);
        // The GPU only applies the s_1 <= s_{n-1} symmetry rule; keep the canonical rulers.
        let mut found: Vec<Vec<u8>> = pass
            .rulers
            .chunks(n)
            .map(|pos| {
                let pos: Vec<u8> = pos.iter().map(|&p| p as u8).collect();
                ctx.search.segments(&pos)
            })
            .filter(|r| is_canonical(r))
            .collect();
        ctx.record(&found);
        found_rulers.append(&mut found);

        for slot in pass.finished {
            let item = self.slot_item[slot as usize].take().unwrap();
            self.finish(item);
            self.free_slots.push(slot);
        }
        let slot_item = &self.slot_item;
        self.active_slots
            .retain(|&slot| slot_item[slot as usize].is_some());

        if self.next_item == total
            && self.split.is_empty()
            && !self.active_slots.is_empty()
            && self.active_slots.len() <= gpu.max_slots as usize / 8
        {
            self.split_active(ctx, gpu)?;
        }

        while self.unfinished < total && self.done[self.unfinished as usize] {
            self.unfinished += 1;
        }
        Ok(())
    }

    /// Replaces the partial ruler in every busy slot with the partial rulers it has left to
    /// search, and frees the slots.
    fn split_active(&mut self, ctx: &WorkContext, gpu: &GpuContext) -> Result<(), String> {
        let states = gpu.read_states(&self.active_slots)?;
        for (slot, state) in std::mem::take(&mut self.active_slots)
            .into_iter()
            .zip(states.chunks(gpu::STATE_STRIDE))
        {
            let item = self.slot_item[slot as usize].take().unwrap();
            let k = state[0] as usize;
            let pos: Vec<u8> = state[1..=k].iter().map(|&p| p as u8).collect();
            let root = state[gpu::STATE_STRIDE - 1] as usize;
            ctx.search.remaining(&pos, root, &state[33..65], |marks| {
                self.split.push((item, marks.to_vec()));
                self.outstanding[item as usize] += 1;
            });
            self.finish(item);
            self.free_slots.push(slot);
        }
        Ok(())
    }
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

    let mut last_save = Instant::now();
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
                        // Save finished lengths every so often, so a crash loses little work.
                        if last_save.elapsed() >= Duration::from_secs(10) {
                            if let Err(e) = save_state(&state, save_path.as_deref()) {
                                eprintln!("Warning: failed to save state: {e}");
                            }
                            last_save = Instant::now();
                        }
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
                SearchStatus::Interrupted | SearchStatus::GpuFailed => {
                    progress.status_pb.finish_and_clear();
                    progress.progress_pb.finish_and_clear();
                    progress.stats_pb.finish_and_clear();
                    let failed = status == SearchStatus::GpuFailed;
                    if failed {
                        eprintln!("\nGPU search failed, saving and exiting...");
                    } else if !quiet {
                        println!("\nReceived interrupt, saving and exiting...");
                    }
                    if let Err(e) = save_state(&state, save_path.as_deref()) {
                        let destination = save_path.as_deref().unwrap_or("stdout");
                        eprintln!("Error: Failed to save state to '{}': {}", destination, e);
                    }
                    std::process::exit(if failed { 1 } else { 0 });
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
        let ctx = pollster::block_on(GpuContext::new(1 << 18, 1 << 18));
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
    fn test_get_num_segments_lower_bound() {
        assert_eq!(get_num_segments_lower_bound(6), 3);
        assert_eq!(get_num_segments_lower_bound(13), 4);
        assert_eq!(get_num_segments_lower_bound(31), 6);
    }
}
