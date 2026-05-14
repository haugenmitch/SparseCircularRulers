use fixedbitset::FixedBitSet;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufWriter;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

#[derive(Serialize, Deserialize)]
struct Solution {
    num_segments: u8,
    rulers: Vec<Vec<u8>>,
    total_rulers_evaluated: u64,
    total_clock_time: Duration,
    total_cpu_time: Duration,
}

#[derive(Serialize, Deserialize)]
struct State {
    rulers_solved: u8,
    total_rulers_evaluated: u64,
    total_clock_time: Duration,
    total_cpu_time: Duration,
    checkpoint_ruler: Vec<u8>,
    solutions: HashMap<u8, Solution>,
}

use serde_json::ser::{Formatter, PrettyFormatter, Serializer};
use std::io::{self, Write};

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

fn save_state(state: &State, path: &str) -> std::io::Result<()> {
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    let mut ser = Serializer::with_formatter(writer, CompactArrayFormatter::new());
    state.serialize(&mut ser).map_err(std::io::Error::other)?;
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

fn find_rulers(
    length: u8,
    num_segments: u8,
    current_segments: &mut [u8],
    current_index: usize,
    remaining_sum: usize,
    interrupt: &Arc<AtomicBool>,
) -> Vec<Vec<u8>> {
    let mut found_rulers = Vec::new();

    if !interrupt.load(Ordering::SeqCst) {
        return found_rulers;
    }

    // Remaining segments, not including the current segment
    let remaining_segments = num_segments.saturating_sub(current_index as u8 + 1);

    // Check if this is the last segment
    if remaining_segments == 0 {
        current_segments[current_index] = remaining_sum as u8;
        if is_complete(current_segments, length) {
            println!("{:?}", current_segments);
            found_rulers.push(current_segments.to_vec());
        }
        return found_rulers;
    }

    // Leave at least 1 for each remaining segment (the last segment should be at least 2)
    // This is because there should be no trailing ones. Any ruler with trailing ones
    // could be shifted so the first trailing one becomes the first one of the ruler.
    let max_curr_segment_size = remaining_sum.saturating_sub(remaining_segments as usize + 1);

    for i in 1..=max_curr_segment_size {
        current_segments[current_index] = i as u8;
        found_rulers.extend(find_rulers(
            length,
            num_segments,
            current_segments,
            current_index + 1,
            remaining_sum - i,
            interrupt,
        ));

        if !interrupt.load(Ordering::SeqCst) {
            break;
        }
    }

    found_rulers
}

fn execute(mut state: State) {
    let interrupt = Arc::new(AtomicBool::new(true));
    let r = interrupt.clone();

    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
        println!("\nReceived interrupt, saving and exiting...");
    })
    .expect("Error setting Ctrl-C handler");

    for i in (state.rulers_solved + 1)..=255 {
        println!("Solving for length: {}", i);
        let mut num_segments = get_num_segments_lower_bound(i);
        let mut solution = Solution {
            num_segments: 0,
            rulers: vec![],
            total_rulers_evaluated: 0,
            total_clock_time: Duration::ZERO,
            total_cpu_time: Duration::ZERO,
        };

        loop {
            let mut starting_ruler = vec![0; num_segments as usize];
            starting_ruler[0] = 1;
            solution.rulers = find_rulers(
                i,
                num_segments,
                &mut starting_ruler,
                1,
                (i - 1) as usize,
                &interrupt,
            );

            if !solution.rulers.is_empty() {
                solution.num_segments = num_segments;
                break;
            }

            if !interrupt.load(Ordering::SeqCst) {
                break;
            }

            num_segments += 1;
        }

        if !interrupt.load(Ordering::SeqCst) {
            break;
        }

        state.solutions.insert(i, solution);
        state.rulers_solved = i;
    }

    save_state(&state, "results.json").expect("Failed to save state");
}

fn main() {
    let state_data = load_state("results.json");

    if let Some(state) = state_data {
        execute(state);
    } else {
        let state = State {
            rulers_solved: 2,
            total_rulers_evaluated: 0,
            total_clock_time: Duration::ZERO,
            total_cpu_time: Duration::ZERO,
            checkpoint_ruler: vec![],
            solutions: HashMap::new(),
        };
        execute(state);
    }
}
