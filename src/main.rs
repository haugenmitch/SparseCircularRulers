use clap::Parser;
use cpu_time::ProcessTime;
use fixedbitset::FixedBitSet;
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
    // Create a bit set of ruler length - 1
    let mut measurable_lengths = FixedBitSet::with_capacity(total_length as usize);

    // Add each possible length to the bit set
    for i in 0..segments.len() {
        let mut len: u16 = segments[i] as u16;
        measurable_lengths.insert(len as usize);
        for j in 1..(segments.len() - 1) {
            len += segments[(i + j) % segments.len()] as u16;
            measurable_lengths.insert(len as usize);
        }
    }

    // Check if every length has been set
    measurable_lengths.count_ones(..) == total_length as usize - 1
}

impl Solution {
    fn find_rulers(
        &mut self,
        length: u8,
        current_segments: &mut [u8],
        current_index: usize,
        remaining_sum: usize,
        interrupt: &AtomicBool,
    ) {
        if !interrupt.load(Ordering::SeqCst) {
            return;
        }

        // Remaining segments, not including the current segment
        let remaining_segments = self.num_segments.saturating_sub(current_index as u8 + 1);

        // Check if this is the last segment
        if remaining_segments == 0 {
            self.total_rulers_evaluated += 1;
            current_segments[current_index] = remaining_sum as u8;
            if is_complete(current_segments, length) {
                self.rulers.push(current_segments.to_vec());
            }
            return;
        }

        // Leave at least 1 for each remaining segment (the last segment should be at least 2)
        // This is because there should be no trailing ones. Any ruler with trailing ones
        // could be shifted so the first trailing one becomes the first one of the ruler.
        let max_curr_segment_size = remaining_sum.saturating_sub(remaining_segments as usize + 1);

        for i in 1..=max_curr_segment_size {
            current_segments[current_index] = i as u8;
            self.find_rulers(
                length,
                current_segments,
                current_index + 1,
                remaining_sum - i,
                interrupt,
            );

            if !interrupt.load(Ordering::SeqCst) {
                break;
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
        if let Some(sol) = state.solutions.get(&i) {
            if sol.completed {
                continue;
            }
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

            let mut starting_ruler = vec![0; num_segments as usize];
            starting_ruler[0] = 1;
            solution.find_rulers(i, &mut starting_ruler, 1, (i - 1) as usize, &interrupt);

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
        while state.solutions.get(&i).map_or(false, |s| s.completed) {
            i += 1;
        }
        i
    };

    execute(state, save_path, start_length, cli.end);
}
