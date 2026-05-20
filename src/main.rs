use clap::Parser;
use cpu_time::ProcessTime;
use fixedbitset::FixedBitSet;
use serde::{Deserialize, Serialize};
use serde_json::ser::{Formatter, PrettyFormatter, Serializer};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
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
}

#[derive(Serialize, Deserialize)]
struct Solution {
    #[serde(default)]
    completed: bool,
    num_segments: u8,
    rulers: Vec<Vec<u8>>,
    rulers_found: u64,
    total_rulers_evaluated: u64,
    total_clock_time: Duration,
    total_cpu_time: Duration,
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
    } else {
        let mut ser = Serializer::with_formatter(io::stdout(), CompactArrayFormatter::new());
        state.serialize(&mut ser).map_err(std::io::Error::other)?;
        println!();
    }
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

impl Solution {
    /// Systematically generates and evaluates all possible sparse ruler configurations
    /// for a given length and number of segments.
    ///
    /// This method uses an iterative partition-generation algorithm (similar to an odometer).
    /// It keeps the first segment fixed at 1 and explores all other combinations of segment
    /// lengths that sum up to the target length.
    fn find_rulers(&mut self, length: u8, starting_ruler: Option<Vec<u8>>, interrupt: &AtomicBool) {
        let n = self.num_segments as usize;

        // Initialize the ruler configuration.
        // If no starting ruler is provided, start with the most "right-heavy" configuration:
        // [1, 1, 1, ..., (length - (n-1))]
        let mut ruler = if let Some(r) = starting_ruler {
            r
        } else {
            let mut r = vec![1u8; n];
            r[n - 1] = length - (n as u8 - 1);
            r
        };

        loop {
            // Check for external interrupt (Ctrl-C)
            if !interrupt.load(Ordering::SeqCst) {
                return;
            }

            self.total_rulers_evaluated += 1;

            // Check if current configuration can measure all lengths
            if is_complete(&ruler, length) {
                self.rulers.push(ruler.clone());
            }

            // --- Permutation Logic (Moving from right to left) ---

            // If the last segment has "weight" to give, move it to the neighbor on the left
            if ruler[n - 1] > 1 {
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

                if all_attempted {
                    break;
                }
            }
        }
    }
}

fn execute(mut state: State, save_path: Option<String>, start_length: u8, end_length: u8) {
    let interrupt = Arc::new(AtomicBool::new(true));
    let r = interrupt.clone();

    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
        println!("\nReceived interrupt, saving and exiting...");
    })
    .expect("Error setting Ctrl-C handler");

    for i in start_length..=end_length {
        if state.solutions.get(&i).is_some_and(|s| s.completed) {
            continue;
        }

        println!("Solving for length: {}", i);
        let mut num_segments = get_num_segments_lower_bound(i);
        let mut solution;

        let length_clock_start = Instant::now();
        let length_cpu_start = ProcessTime::now();

        loop {
            solution = Solution {
                completed: false,
                num_segments,
                rulers: vec![],
                rulers_found: 0,
                total_rulers_evaluated: 0,
                total_clock_time: Duration::ZERO,
                total_cpu_time: Duration::ZERO,
            };

            solution.find_rulers(i, None, &interrupt);

            if !interrupt.load(Ordering::SeqCst) {
                break;
            }

            if !solution.rulers.is_empty() {
                solution.completed = true;
                solution.rulers_found = solution.rulers.len() as u64;
                break;
            }

            num_segments += 1;
        }

        if !interrupt.load(Ordering::SeqCst) {
            break;
        }

        solution.total_clock_time = length_clock_start.elapsed();
        solution.total_cpu_time = length_cpu_start.elapsed();

        state.solutions.insert(i, solution);
        state.recalculate_global_metrics();
    }

    if let Err(e) = save_state(&state, save_path.as_deref()) {
        let destination = save_path.as_deref().unwrap_or("stdout");
        eprintln!("Error: Failed to save state to '{}': {}", destination, e);
        std::process::exit(1);
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

    execute(state, save_path, start_length, cli.end);
}
