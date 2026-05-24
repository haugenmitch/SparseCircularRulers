use clap::Parser;
use cpu_time::ProcessTime;
use fixedbitset::FixedBitSet;
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use serde::{Deserialize, Serialize};
use serde_json::ser::{Formatter, PrettyFormatter, Serializer};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

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
}

#[derive(Serialize, Deserialize)]
struct Solution {
    #[serde(default)]
    completed: bool,
    num_segments: u8,
    rulers: Vec<Vec<u8>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    checkpoint_ruler: Option<Vec<u8>>,
    rulers_found: u64,
    total_rulers_evaluated: u64,
    total_clock_time: Duration,
    total_cpu_time: Duration,
}

#[derive(PartialEq)]
enum SearchStatus {
    Finished,
    Interrupted,
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
            if solution.completed {
                self.total_rulers_evaluated += solution.total_rulers_evaluated;
                self.total_rulers_found += solution.rulers_found;
                self.total_clock_time += solution.total_clock_time;
                self.total_cpu_time += solution.total_cpu_time;
                self.lengths_solved += 1;
            }
        }
    }

    /// Systematically generates and evaluates all possible sparse ruler configurations
    /// for a given length and number of segments.
    ///
    /// This method uses an iterative partition-generation algorithm (similar to an odometer).
    /// It keeps the first segment fixed at 1 and explores all other combinations of segment
    /// lengths that sum up to the target length.
    fn find_rulers(
        &mut self,
        length: u8,
        num_segments: u8,
        save_path: Option<&str>,
        interrupt: &AtomicBool,
        status_pb: &ProgressBar,
        progress_pb: &ProgressBar,
        stats_pb: &ProgressBar,
    ) -> SearchStatus {
        let n = num_segments as usize;

        // Ensure we have a solution entry and get the starting configuration
        let mut ruler = {
            let solution = self.solutions.entry(length).or_insert(Solution {
                completed: false,
                num_segments,
                rulers: vec![],
                checkpoint_ruler: None,
                rulers_found: 0,
                total_rulers_evaluated: 0,
                total_clock_time: Duration::ZERO,
                total_cpu_time: Duration::ZERO,
            });

            if let Some(r) = solution.checkpoint_ruler.take() {
                // Synchronize progress count with the checkpoint ruler position
                solution.total_rulers_evaluated = calculate_rank(length, num_segments, &r) as u64;
                r
            } else {
                let mut r = vec![1u8; n];
                r[n - 1] = length - (n as u8 - 1);
                r
            }
        };

        let total_space = calculate_total_space(length, num_segments);
        progress_pb.set_length(total_space as u64);
        stats_pb.set_length(total_space as u64);
        {
            let solution = self.solutions.get(&length).unwrap();
            progress_pb.set_position(solution.total_rulers_evaluated);
            stats_pb.set_position(solution.total_rulers_evaluated);
            status_pb.set_message(format!(
                "{}: Found {} rulers with {} segments in {:?}",
                length, solution.rulers_found, num_segments, solution.total_clock_time
            ));
        }

        let mut local_evals = 0;
        const CHECKPOINT_INTERVAL: u64 = 1_000_000;
        let mut length_clock_start = Instant::now();
        let mut length_cpu_start = ProcessTime::now();

        loop {
            // Check for external interrupt (Ctrl-C)
            if !interrupt.load(Ordering::SeqCst) {
                let solution = self.solutions.get_mut(&length).unwrap();
                solution.checkpoint_ruler = Some(ruler);
                solution.total_clock_time += length_clock_start.elapsed();
                solution.total_cpu_time += length_cpu_start.elapsed();
                self.recalculate_global_metrics();
                return SearchStatus::Interrupted;
            }

            // Perform evaluation
            {
                let solution = self.solutions.get_mut(&length).unwrap();
                solution.total_rulers_evaluated += 1;
                if is_complete(&ruler, length) {
                    solution.rulers.push(ruler.clone());
                    solution.rulers_found = solution.rulers.len() as u64;

                    let elapsed = solution.total_clock_time + length_clock_start.elapsed();
                    status_pb.set_message(format!(
                        "{}: Found {} rulers with {} segments in {:?}",
                        length, solution.rulers_found, num_segments, elapsed
                    ));
                }
            }

            local_evals += 1;

            // Periodic Checkpoint - Save while continuing
            if local_evals >= CHECKPOINT_INTERVAL {
                let (rulers_found, total_evaluated, elapsed) = {
                    let solution = self.solutions.get_mut(&length).unwrap();
                    solution.checkpoint_ruler = Some(ruler.clone());
                    solution.total_clock_time += length_clock_start.elapsed();
                    solution.total_cpu_time += length_cpu_start.elapsed();
                    (
                        solution.rulers_found,
                        solution.total_rulers_evaluated,
                        solution.total_clock_time,
                    )
                };

                self.recalculate_global_metrics();
                let _ = save_state(self, save_path);

                progress_pb.set_position(total_evaluated);
                stats_pb.set_position(total_evaluated);
                status_pb.set_message(format!(
                    "{}: Found {} rulers with {} segments in {:?}",
                    length, rulers_found, num_segments, elapsed
                ));

                // Reset chunk timers and local counter
                length_clock_start = Instant::now();
                length_cpu_start = ProcessTime::now();
                local_evals = 0;
            }

            // --- Permutation Logic (Moving from right to left) ---

            // If the last segment has "weight" to give, move it to the neighbor on the left
            if n > 2 && ruler[n - 1] > 1 {
                ruler[n - 1] -= 1;
                ruler[n - 2] += 1;
            } else {
                // If the last segment is 1, we must "carry" the weight further left.
                // We look for the first segment from the right (excluding index 0) that is > 1.
                let mut all_attempted = false;
                for i in (1..n - 1).rev() {
                    if i == 1 {
                        // If we reach index 1 and it's already exhausted (can't be incremented
                        // without affecting index 0), then all permutations are done.
                        all_attempted = true;
                        break;
                    }
                    if ruler[i] == 1 {
                        // Keep moving left
                        continue;
                    } else {
                        // Found a segment to decrement.
                        // Reset this segment to 1, increment its left neighbor,
                        // and put all remaining weight back into the far right segment.
                        ruler[i] = 1;
                        ruler[i - 1] += 1;
                        let current_sum: u16 = ruler[0..n - 1].iter().map(|&x| x as u16).sum();
                        ruler[n - 1] = length - (current_sum as u8);
                        break;
                    }
                }

                if all_attempted || n < 3 {
                    let solution = self.solutions.get_mut(&length).unwrap();
                    solution.checkpoint_ruler = None;
                    solution.total_clock_time += length_clock_start.elapsed();
                    solution.total_cpu_time += length_cpu_start.elapsed();
                    return SearchStatus::Finished;
                }
            }
        }
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

fn is_complete(segments: &[u8], total_length: u8) -> bool {
    let n = segments.len();
    let mut measurable_lengths = FixedBitSet::with_capacity(total_length as usize);
    let mut sums = vec![0u16; n];

    // Iteratively calculate sums of k segments (from k=1 to n-1)
    for k in 1..n {
        for i in 0..n {
            // Update the sum at index i to include the next successive segment
            sums[i] += segments[(i + k - 1) % n] as u16;

            measurable_lengths.insert(sums[i] as usize);
        }

        // Early Exit Optimization:
        // Since all segments are >= 1, any contiguous sum of k segments must be
        // at least k. If we haven't found length 'k' by now, we never will.
        if !measurable_lengths.contains(k) {
            return false;
        }
    }

    // Final check to ensure every length from 1 to total_length-1 is measurable.
    measurable_lengths.count_ones(..) == total_length as usize - 1
}

fn execute(
    mut state: State,
    save_path: Option<String>,
    start_length: u8,
    end_length: u8,
    output_json: bool,
    quiet: bool,
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
            .template("  {elapsed_precise} [{per_sec} evals]")
            .unwrap(),
    );

    let interrupt = Arc::new(AtomicBool::new(true));
    let r = interrupt.clone();

    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
    })
    .expect("Error setting Ctrl-C handler");

    for i in start_length..=end_length {
        if state.solutions.get(&i).is_some_and(|s| s.completed) {
            continue;
        }

        let mut num_segments = state
            .solutions
            .get(&i)
            .map(|s| s.num_segments)
            .unwrap_or_else(|| get_num_segments_lower_bound(i));

        loop {
            let status = state.find_rulers(
                i,
                num_segments,
                save_path.as_deref(),
                &interrupt,
                &status_pb,
                &progress_pb,
                &stats_pb,
            );

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
                    state.solutions.remove(&i);
                }
                SearchStatus::Interrupted => {
                    status_pb.finish_and_clear();
                    progress_pb.finish_and_clear();
                    stats_pb.finish_and_clear();
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

    status_pb.finish_and_clear();
    progress_pb.finish_and_clear();
    stats_pb.finish_and_clear();

    state.recalculate_global_metrics();
    if let Err(e) = save_state(&state, save_path.as_deref()) {
        let destination = save_path.as_deref().unwrap_or("stdout");
        eprintln!("Error: Failed to save state to '{}': {}", destination, e);
        std::process::exit(1);
    }

    if output_json && !quiet {
        let _ = print_state(&state);
    }
}

fn main() {
    let cli = Cli::parse();

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

    execute(state, save_path, start_length, cli.end, cli.json, cli.quiet);
}
