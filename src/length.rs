//! The search for every canonical ruler of one length and segment count: splitting it into
//! prefixes, searching them on the CPU threads or the GPU, and checkpointing.

use crate::gpu::{self, GpuContext};
use crate::progress::Progress;
use crate::ruler::{is_canonical, num_segments_lower_bound};
use crate::search::{Dfs, MAX_MARKS, Search};
use crate::state::{FinishedItems, Solution, State};
use cpu_time::ProcessTime;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

pub const CHECKPOINT_INTERVAL: Duration = Duration::from_secs(10);
/// Prefixes to split a search into: at least this many for the CPU or the GPU, at most `MAX`.
const CPU_PREFIXES: usize = 1 << 16;
const GPU_PREFIXES: usize = 1 << 21;
const MAX_PREFIXES: usize = 1 << 24;
/// GPU node budget per thread and pass: initial value and range.
const START_BUDGET: u32 = 1 << 12;
const MIN_BUDGET: u32 = 1 << 8;
const MAX_BUDGET: u32 = 1 << 22;

/// How a search ended.
#[derive(PartialEq, Copy, Clone)]
pub enum SearchStatus {
    Finished,
    Interrupted,
    GpuFailed,
    /// The segment count is outside what the search supports (2 to `MAX_MARKS`).
    Unsupported,
}

/// What a search of one length needs from the caller.
pub struct SearchContext<'a> {
    pub length: u8,
    pub num_segments: u8,
    pub save_path: Option<&'a str>,
    /// Cleared on Ctrl-C.
    pub running: &'a AtomicBool,
    pub progress: &'a Progress,
    pub gpu: Option<&'a GpuContext>,
}

/// Counts shared by the workers and read by the display.
pub struct Counters {
    pub evaluated: AtomicU64,
    pub found: AtomicU64,
    /// Prefixes fully searched.
    pub done: AtomicU64,
}

/// Clock and CPU time of a search, including the runs before it was resumed.
struct Timing {
    base_clock: Duration,
    base_cpu: Duration,
    clock_start: Instant,
    cpu_start: ProcessTime,
}

impl Timing {
    /// Total clock time.
    fn clock(&self) -> Duration {
        self.base_clock + self.clock_start.elapsed()
    }

    /// Total CPU time.
    fn cpu(&self) -> Duration {
        self.base_cpu + self.cpu_start.elapsed()
    }
}

/// Searches one length with one segment count, resuming from its checkpoint, and stores the
/// result (or the checkpoint, if interrupted) in `state`.
pub fn search_length(state: &mut State, ctx: &SearchContext) -> SearchStatus {
    let n = ctx.num_segments as usize;
    if ctx.length == 1 && n == 1 {
        let solution = state
            .solutions
            .entry(1)
            .or_insert_with(|| Solution::new(1, 1));
        solution.completed = true;
        solution.rulers = vec![vec![1]];
        solution.rulers_found = 1;
        solution.total_rulers_evaluated = 1;
        state.recalculate_global_metrics();
        return SearchStatus::Finished;
    }
    if ctx.length < 2 || !(2..=MAX_MARKS).contains(&n) {
        return SearchStatus::Unsupported;
    }

    let resume = take_resume_point(state, ctx);
    let solution = &state.solutions[&ctx.length];
    let timing = Timing {
        base_clock: solution.total_clock_time,
        base_cpu: solution.total_cpu_time,
        clock_start: Instant::now(),
        cpu_start: ProcessTime::now(),
    };

    let search = Search::new(ctx.length, ctx.num_segments);
    let gpu = ctx.gpu.filter(|_| n >= 3);
    let target = if gpu.is_some() {
        GPU_PREFIXES
    } else {
        CPU_PREFIXES
    };
    let (depth, prefixes) = search.prefixes(target, (n - 2).max(2), MAX_PREFIXES);
    let counters = Counters {
        evaluated: AtomicU64::new(resume.evaluated),
        found: AtomicU64::new(resume.rulers.len() as u64),
        done: AtomicU64::new(0),
    };
    let work = Work {
        search: &search,
        depth,
        prefixes: &prefixes,
        running: ctx.running,
        counters: &counters,
        latest_ruler: &ctx.progress.latest_ruler,
    };
    let first = resume
        .checkpoint
        .map_or(0, |marks| work.first_unfinished(&marks));
    let mut finished = Finished::new(work.items(), first, resume.finished);
    counters.done.store(finished.count, Ordering::Relaxed);
    *ctx.progress.latest_ruler.lock().unwrap() = resume.rulers.last().cloned();
    ctx.progress.start(finished.count, finished.total());

    let mut found = resume.rulers;
    let ui_stop = AtomicBool::new(false);
    let status = std::thread::scope(|scope| {
        scope.spawn(|| {
            ctx.progress.update_until(
                &ui_stop,
                (ctx.length, ctx.num_segments),
                &counters,
                timing.base_clock,
                timing.clock_start,
            )
        });
        let mut checkpoint = |finished: &Finished, found: &[Vec<u8>]| {
            save_checkpoint(state, ctx, &work, finished, found, &timing);
        };
        let status = match gpu {
            Some(gpu) => gpu_search(&work, gpu, &mut finished, &mut found, &mut checkpoint)
                .unwrap_or_else(|e| {
                    ctx.progress.suspend(|| eprintln!("\nError: {e}"));
                    SearchStatus::GpuFailed
                }),
            None => cpu_search(&work, &mut finished, &mut found, &mut checkpoint),
        };
        ui_stop.store(true, Ordering::Relaxed);
        status
    });

    // A resumed search may repeat part of a prefix, so drop duplicates.
    found.sort();
    found.dedup();

    let solution = state.solutions.get_mut(&ctx.length).unwrap();
    solution.total_rulers_evaluated = counters.evaluated.load(Ordering::Relaxed);
    solution.rulers_found = found.len() as u64;
    solution.rulers = found;
    solution.total_clock_time = timing.clock();
    solution.total_cpu_time = timing.cpu();
    (solution.checkpoint_ruler, solution.checkpoint_finished) =
        if status == SearchStatus::Interrupted {
            work.checkpoint(&finished)
        } else {
            (None, None)
        };
    state.recalculate_global_metrics();
    status
}

/// Where a saved search of this length left off.
struct ResumePoint {
    /// Marks of the first prefix not fully searched.
    checkpoint: Option<Vec<u8>>,
    finished: Option<FinishedItems>,
    evaluated: u64,
    rulers: Vec<Vec<u8>>,
}

/// Takes the saved progress of this length out of `state`, creating its solution if needed.
fn take_resume_point(state: &mut State, ctx: &SearchContext) -> ResumePoint {
    let lower_bound = num_segments_lower_bound(ctx.length);
    let solution = state
        .solutions
        .entry(ctx.length)
        .or_insert_with(|| Solution::new(lower_bound, ctx.num_segments));

    // Progress saved for another segment count does not apply.
    if !solution.completed && solution.num_segments != ctx.num_segments {
        solution.num_segments = ctx.num_segments;
        solution.total_rulers_evaluated = 0;
        solution.checkpoint_ruler = None;
        solution.checkpoint_finished = None;
    }

    let checkpoint = solution.checkpoint_ruler.take().map(|segments| {
        let mut marks = vec![0u8];
        for s in segments {
            marks.push(marks.last().unwrap().saturating_add(s));
        }
        marks
    });
    ResumePoint {
        checkpoint,
        finished: solution.checkpoint_finished.take(),
        evaluated: solution.total_rulers_evaluated,
        rulers: std::mem::take(&mut solution.rulers),
    }
}

/// Stores the search's progress in `state` and saves it in the background.
fn save_checkpoint(
    state: &mut State,
    ctx: &SearchContext,
    work: &Work,
    finished: &Finished,
    found: &[Vec<u8>],
    timing: &Timing,
) {
    let solution = state.solutions.get_mut(&ctx.length).unwrap();
    (solution.checkpoint_ruler, solution.checkpoint_finished) = work.checkpoint(finished);
    solution.total_rulers_evaluated = work.counters.evaluated.load(Ordering::Relaxed);
    solution.rulers_found = found.len() as u64;
    solution.total_clock_time = timing.clock();
    solution.total_cpu_time = timing.cpu();
    state.recalculate_global_metrics();
    if let Err(e) = state.save(ctx.save_path, Some((ctx.length, found)), false) {
        ctx.progress
            .suspend(|| eprintln!("\nWarning: failed to save state: {e}"));
    }
}

/// One search split into prefixes of `depth` marks, and what its workers share.
struct Work<'a> {
    search: &'a Search,
    depth: usize,
    prefixes: &'a [u8],
    running: &'a AtomicBool,
    counters: &'a Counters,
    latest_ruler: &'a Mutex<Option<Vec<u8>>>,
}

impl Work<'_> {
    /// Number of prefixes.
    fn items(&self) -> u64 {
        (self.prefixes.len() / self.depth) as u64
    }

    /// The marks of prefix `item`.
    fn prefix(&self, item: u64) -> &[u8] {
        let i = item as usize * self.depth;
        &self.prefixes[i..i + self.depth]
    }

    /// The first prefix not before `marks` (prefixes are sorted).
    fn first_unfinished(&self, marks: &[u8]) -> u64 {
        let prefixes: Vec<&[u8]> = self.prefixes.chunks(self.depth).collect();
        prefixes.partition_point(|p| {
            let m = p.len().min(marks.len());
            p[..m] < marks[..m]
        }) as u64
    }

    /// The segments of the first unfinished prefix and the prefixes finished after it.
    fn checkpoint(&self, finished: &Finished) -> (Option<Vec<u8>>, Option<FinishedItems>) {
        let first = (finished.first < finished.total()).then(|| {
            let prefix = self.prefix(finished.first);
            prefix.windows(2).map(|w| w[1] - w[0]).collect()
        });
        (first, finished.saved())
    }

    /// Counts newly found rulers and shows the last one.
    fn record(&self, rulers: &[Vec<u8>]) {
        if let Some(last) = rulers.last() {
            self.counters
                .found
                .fetch_add(rulers.len() as u64, Ordering::Relaxed);
            if let Ok(mut latest) = self.latest_ruler.try_lock() {
                *latest = Some(last.clone());
            }
        }
    }
}

/// Which prefixes of a search are fully searched.
struct Finished {
    done: Vec<bool>,
    /// The first prefix not fully searched.
    first: u64,
    /// Prefixes fully searched.
    count: u64,
}

impl Finished {
    /// Every prefix before `first`, and those in `saved`, are fully searched.
    fn new(total: u64, first: u64, saved: Option<FinishedItems>) -> Self {
        let mut finished = Finished {
            done: vec![false; total as usize],
            first: 0,
            count: 0,
        };
        for item in 0..first {
            finished.mark(item);
        }
        for [a, b] in saved
            .filter(|saved| saved.items == total)
            .map_or(vec![], |saved| saved.ranges)
        {
            for item in a.max(first)..b.min(total) {
                finished.mark(item);
            }
        }
        finished
    }

    /// Number of prefixes.
    fn total(&self) -> u64 {
        self.done.len() as u64
    }

    /// Whether every prefix is fully searched.
    fn is_complete(&self) -> bool {
        self.first == self.total()
    }

    /// Marks prefix `item` fully searched.
    fn mark(&mut self, item: u64) {
        if !self.done[item as usize] {
            self.done[item as usize] = true;
            self.count += 1;
        }
        while self.first < self.total() && self.done[self.first as usize] {
            self.first += 1;
        }
    }

    /// The prefixes fully searched after the first unfinished one, as ranges.
    fn saved(&self) -> Option<FinishedItems> {
        let mut ranges: Vec<[u64; 2]> = Vec::new();
        for item in self.first..self.total() {
            if self.done[item as usize] {
                match ranges.last_mut() {
                    Some(last) if last[1] == item => last[1] += 1,
                    _ => ranges.push([item, item + 1]),
                }
            }
        }
        (!ranges.is_empty()).then(|| FinishedItems {
            items: self.total(),
            ranges,
        })
    }
}

/// Searches the unfinished prefixes on the thread pool, with the narrowest bitsets that hold
/// the length.
fn cpu_search(
    work: &Work,
    finished: &mut Finished,
    found: &mut Vec<Vec<u8>>,
    checkpoint: &mut impl FnMut(&Finished, &[Vec<u8>]),
) -> SearchStatus {
    match work.search.length() {
        ..=64 => cpu_search_with::<1, 1>(work, finished, found, checkpoint),
        ..=127 => cpu_search_with::<2, 1>(work, finished, found, checkpoint),
        _ => cpu_search_with::<4, 2>(work, finished, found, checkpoint),
    }
}

/// Each thread takes the next prefix as soon as it finishes its last; this thread collects
/// the results and checkpoints.
fn cpu_search_with<const W: usize, const H: usize>(
    work: &Work,
    finished: &mut Finished,
    found: &mut Vec<Vec<u8>>,
    checkpoint: &mut impl FnMut(&Finished, &[Vec<u8>]),
) -> SearchStatus {
    let search = Search::<W, H>::new(work.search.length() as u8, work.search.num_segments() as u8);
    let todo: Vec<u64> = (finished.first..finished.total())
        .filter(|&item| !finished.done[item as usize])
        .collect();
    let next = AtomicUsize::new(0);
    let (tx, rx) = mpsc::channel::<(u64, Vec<Vec<u8>>, bool)>();
    std::thread::scope(|scope| {
        let (search, todo, next) = (&search, &todo, &next);
        scope.spawn(move || {
            rayon::broadcast(|_| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= todo.len() || !work.running.load(Ordering::Relaxed) {
                        break;
                    }
                    let mut dfs = Dfs::new(search, work.running, &work.counters.evaluated);
                    dfs.run(&search.node_from_marks(work.prefix(todo[i])));
                    dfs.flush();
                    work.record(&dfs.found);
                    let _ = tx.send((todo[i], dfs.found, !dfs.interrupted));
                }
            });
        });

        let mut last_checkpoint = Instant::now();
        loop {
            match rx.recv_timeout(Duration::from_millis(100)) {
                Ok((item, mut rulers, completed)) => {
                    found.append(&mut rulers);
                    if completed {
                        finished.mark(item);
                        work.counters.done.store(finished.count, Ordering::Relaxed);
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
            if last_checkpoint.elapsed() >= CHECKPOINT_INTERVAL {
                checkpoint(finished, found);
                last_checkpoint = Instant::now();
            }
        }
    });
    if finished.is_complete() {
        SearchStatus::Finished
    } else {
        SearchStatus::Interrupted
    }
}

/// Runs GPU passes until every prefix is searched or the search is interrupted.
fn gpu_search(
    work: &Work,
    gpu: &GpuContext,
    finished: &mut Finished,
    found: &mut Vec<Vec<u8>>,
    checkpoint: &mut impl FnMut(&Finished, &[Vec<u8>]),
) -> Result<SearchStatus, String> {
    let mut slots = GpuSlots::new(gpu, finished);
    let mut last_checkpoint = Instant::now();
    loop {
        if finished.is_complete() {
            return Ok(SearchStatus::Finished);
        }
        if !work.running.load(Ordering::SeqCst) {
            return Ok(SearchStatus::Interrupted);
        }
        slots.pass(work, gpu, finished, found)?;
        work.counters.done.store(finished.count, Ordering::Relaxed);
        if last_checkpoint.elapsed() >= CHECKPOINT_INTERVAL {
            checkpoint(finished, found);
            last_checkpoint = Instant::now();
        }
    }
}

/// Prefixes being searched on the GPU. Each occupies a GPU slot, which keeps its search state
/// between passes, until it is fully searched; then the slot takes the next one.
///
/// Once every prefix has started and most slots are idle, the busy slots are split: what each
/// has left to search is handed out again as deeper prefixes, which fill the idle slots.
struct GpuSlots {
    /// The original prefix (item) that each slot's prefix came from.
    slot_item: Vec<Option<u64>>,
    free: Vec<u32>,
    active: Vec<u32>,
    next_item: u64,
    /// Prefixes split off busy slots, waiting for a slot, with their items.
    split: Vec<(u64, Vec<u8>)>,
    /// Prefixes of each item still in a slot or waiting for one.
    outstanding: Vec<u32>,
    node_budget: u32,
}

impl GpuSlots {
    /// Every slot free, starting at the first unfinished prefix.
    fn new(gpu: &GpuContext, finished: &Finished) -> Self {
        GpuSlots {
            slot_item: vec![None; gpu.max_slots as usize],
            free: (0..gpu.max_slots).rev().collect(),
            active: Vec::new(),
            next_item: finished.first,
            split: Vec::new(),
            outstanding: vec![0; finished.total() as usize],
            node_budget: START_BUDGET,
        }
    }

    /// Records that one prefix of `item` was fully searched (or split).
    fn finish(&mut self, finished: &mut Finished, item: u64) {
        self.outstanding[item as usize] -= 1;
        if self.outstanding[item as usize] == 0 {
            finished.mark(item);
        }
    }

    /// Fills free slots, runs one GPU pass, and collects the rulers it found.
    fn pass(
        &mut self,
        work: &Work,
        gpu: &GpuContext,
        finished: &mut Finished,
        found: &mut Vec<Vec<u8>>,
    ) -> Result<(), String> {
        let n = work.search.num_segments();
        let (entries, prefixes) = self.fill_slots(work, finished);

        let start = Instant::now();
        let pass = gpu.run_pass(
            work.search.length() as u32,
            n as u32,
            self.node_budget,
            &entries,
            &prefixes,
        )?;
        // Keep passes well under the driver's GPU timeout but long enough that launching
        // them is cheap.
        let elapsed = start.elapsed();
        if elapsed < Duration::from_millis(100) {
            self.node_budget = (self.node_budget * 2).min(MAX_BUDGET);
        } else if elapsed > Duration::from_millis(500) {
            self.node_budget = (self.node_budget / 2).max(MIN_BUDGET);
        }

        work.counters
            .evaluated
            .fetch_add(pass.nodes, Ordering::Relaxed);
        // The GPU applies only part of the symmetry rules; keep the canonical rulers.
        let mut rulers: Vec<Vec<u8>> = pass
            .rulers
            .chunks(n)
            .map(|pos| {
                let pos: Vec<u8> = pos.iter().map(|&p| p as u8).collect();
                work.search.segments(&pos)
            })
            .filter(|r| is_canonical(r))
            .collect();
        work.record(&rulers);
        found.append(&mut rulers);

        for slot in pass.finished {
            let item = self.slot_item[slot as usize].take().unwrap();
            self.finish(finished, item);
            self.free.push(slot);
        }
        let slot_item = &self.slot_item;
        self.active
            .retain(|&slot| slot_item[slot as usize].is_some());

        if self.next_item == finished.total()
            && self.split.is_empty()
            && !self.active.is_empty()
            && self.active.len() <= gpu.max_slots as usize / 8
        {
            self.split_active(work, gpu, finished)?;
        }
        Ok(())
    }

    /// Puts split and then new prefixes in the free slots. Returns the pass's work entries
    /// (a slot and a prefix offset, or `gpu::RESUME`, per thread) and the new prefixes (each
    /// its depth followed by its marks).
    fn fill_slots(&mut self, work: &Work, finished: &Finished) -> (Vec<u32>, Vec<u32>) {
        let total = finished.total();
        let mut entries = Vec::new();
        for &slot in &self.active {
            entries.extend([slot, gpu::RESUME]);
        }
        let mut prefixes: Vec<u32> = Vec::new();
        while !self.free.is_empty() {
            // Skip prefixes a resumed search already finished.
            while self.next_item < total && finished.done[self.next_item as usize] {
                self.next_item += 1;
            }
            let (item, marks) = if let Some(split) = self.split.pop() {
                split
            } else if self.next_item < total {
                let item = self.next_item;
                self.outstanding[item as usize] = 1;
                self.next_item += 1;
                (item, work.prefix(item).to_vec())
            } else {
                break;
            };
            let slot = self.free.pop().unwrap();
            self.slot_item[slot as usize] = Some(item);
            self.active.push(slot);
            entries.extend([slot, prefixes.len() as u32]);
            prefixes.push(marks.len() as u32);
            prefixes.extend(marks.iter().map(|&m| m as u32));
        }
        (entries, prefixes)
    }

    /// Replaces the prefix in every busy slot with the prefixes it has left to search, and
    /// frees the slots.
    fn split_active(
        &mut self,
        work: &Work,
        gpu: &GpuContext,
        finished: &mut Finished,
    ) -> Result<(), String> {
        let states = gpu.read_states(&self.active)?;
        for (slot, state) in std::mem::take(&mut self.active)
            .into_iter()
            .zip(states.chunks(gpu::STATE_STRIDE))
        {
            let item = self.slot_item[slot as usize].take().unwrap();
            let state = gpu::SlotState(state);
            let pos: Vec<u8> = state.pos().iter().map(|&p| p as u8).collect();
            work.search
                .remaining(&pos, state.root(), state.next(), |marks| {
                    self.split.push((item, marks.to_vec()));
                    self.outstanding[item as usize] += 1;
                });
            self.finish(finished, item);
            self.free.push(slot);
        }
        Ok(())
    }
}
