//! The saved state: each length's solution, the totals, and checkpoints, as a JSON file.

use serde::{Deserialize, Serialize};
use serde_json::ser::{Formatter, PrettyFormatter, Serializer};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Write};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

/// Everything saved: the solution of each length and the totals.
#[derive(Serialize, Deserialize)]
pub struct State {
    #[serde(default = "default_version")]
    pub version: String,
    pub lengths_solved: u8,
    pub total_rulers_found: u64,
    pub total_rulers_evaluated: u64,
    pub total_clock_time: Duration,
    pub total_cpu_time: Duration,
    pub solutions: HashMap<u8, Solution>,
    #[serde(skip)]
    saved: SaveCache,
}

/// The version of save files written before versions were saved.
fn default_version() -> String {
    "0.0.0".to_string()
}

/// The rulers of one length, or the progress of its unfinished search.
#[derive(Serialize, Deserialize)]
pub struct Solution {
    #[serde(default)]
    pub completed: bool,
    pub lower_bound_num_segments: u8,
    pub num_segments: u8,
    pub rulers: Vec<Vec<u8>>,
    /// Segments of the first prefix not fully searched.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checkpoint_ruler: Option<Vec<u8>>,
    /// Prefixes fully searched after `checkpoint_ruler`, so a resumed search skips them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint_finished: Option<FinishedItems>,
    pub rulers_found: u64,
    pub total_rulers_evaluated: u64,
    pub total_clock_time: Duration,
    pub total_cpu_time: Duration,
}

/// Prefixes fully searched, by index, out of the `items` the search was split into (they
/// only apply to a search split the same way).
#[derive(Serialize, Deserialize, Clone)]
pub struct FinishedItems {
    pub items: u64,
    /// Half-open ranges of indices.
    pub ranges: Vec<[u64; 2]>,
}

impl Solution {
    /// A solution not yet searched.
    pub fn new(lower_bound_num_segments: u8, num_segments: u8) -> Self {
        Solution {
            completed: false,
            lower_bound_num_segments,
            num_segments,
            rulers: vec![],
            checkpoint_ruler: None,
            checkpoint_finished: None,
            rulers_found: 0,
            total_rulers_evaluated: 0,
            total_clock_time: Duration::ZERO,
            total_cpu_time: Duration::ZERO,
        }
    }
}

/// JSON kept from earlier saves, so a save only serializes what changed, and the save being
/// written in the background.
#[derive(Default)]
struct SaveCache {
    /// Each finished length's solution; it never changes again.
    finished: HashMap<u8, Arc<Vec<u8>>>,
    /// The length being searched, how many of its rulers are serialized, and their JSON.
    rulers: (u8, usize, Vec<Arc<Vec<u8>>>),
    writer: Option<JoinHandle<io::Result<()>>>,
}

impl State {
    /// An empty state.
    pub fn new(version: &str) -> Self {
        State {
            version: version.to_string(),
            lengths_solved: 0,
            total_rulers_found: 0,
            total_rulers_evaluated: 0,
            total_clock_time: Duration::ZERO,
            total_cpu_time: Duration::ZERO,
            solutions: HashMap::new(),
            saved: SaveCache::default(),
        }
    }

    /// Reads a save file, or `None` if it is missing or invalid.
    pub fn load(path: &str) -> Option<Self> {
        let file = File::open(path).ok()?;
        serde_json::from_reader(BufReader::new(file)).ok()
    }

    /// Prints the state as JSON to stdout.
    pub fn print_json(&self) -> io::Result<()> {
        let mut ser = Serializer::with_formatter(io::stdout(), CompactArrayFormatter::new());
        self.serialize(&mut ser).map_err(io::Error::other)?;
        println!();
        Ok(())
    }

    /// Recomputes the totals from the solutions.
    pub fn recalculate_global_metrics(&mut self) {
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

    /// Saves the state to `path` on a background thread. `current` holds the rulers found so
    /// far for the length being searched, which its solution does not hold yet. Unless `wait`
    /// is set, the save is skipped while the previous one is still being written; with `wait`,
    /// it returns once this save is written.
    pub fn save(
        &mut self,
        path: Option<&str>,
        current: Option<(u8, &[Vec<u8>])>,
        wait: bool,
    ) -> io::Result<()> {
        let Some(path) = path else { return Ok(()) };
        if let Some(writer) = &self.saved.writer
            && !wait
            && !writer.is_finished()
        {
            return Ok(());
        }
        if let Some(writer) = self.saved.writer.take() {
            writer.join().unwrap()?;
        }

        let pieces = self.json_pieces(current);
        // Write to a temporary file and rename it, so the save file is never half written.
        let path = path.to_string();
        let writer = std::thread::spawn(move || {
            let temp = format!("{path}.tmp");
            let mut file = BufWriter::new(File::create(&temp)?);
            for piece in pieces {
                file.write_all(&piece)?;
            }
            file.into_inner().map_err(|e| e.into_error())?.sync_all()?;
            std::fs::rename(temp, path)
        });
        if wait {
            writer.join().unwrap()
        } else {
            self.saved.writer = Some(writer);
            Ok(())
        }
    }

    /// The save file's JSON, in pieces. Finished lengths are serialized once and the current
    /// length's rulers as they are found, so only what changed is serialized again.
    fn json_pieces(&mut self, current: Option<(u8, &[Vec<u8>])>) -> Vec<Arc<Vec<u8>>> {
        // Serialize the state without its solutions, then put them where the empty map is.
        let solutions = std::mem::take(&mut self.solutions);
        let header = to_json(self);
        self.solutions = solutions;
        let at = split_after(&header, b"\"solutions\": {}", true);
        let mut pieces = vec![Arc::new(header[..at].to_vec())];

        let mut lengths: Vec<u8> = self.solutions.keys().copied().collect();
        lengths.sort();
        for (i, length) in lengths.into_iter().enumerate() {
            let comma = if i > 0 { "," } else { "" };
            pieces.push(Arc::new(
                format!("{comma}\n    \"{length}\": ").into_bytes(),
            ));
            let solution = &self.solutions[&length];
            match current {
                _ if solution.completed => pieces.push(
                    self.saved
                        .finished
                        .entry(length)
                        .or_insert_with(|| Arc::new(indent(to_json(solution))))
                        .clone(),
                ),
                Some((current, rulers))
                    if current == length && solution.rulers.is_empty() && !rulers.is_empty() =>
                {
                    // Put the found rulers inside the solution's empty rulers array.
                    let json = indent(to_json(solution));
                    let at = split_after(&json, b"\"rulers\": []", false);
                    pieces.push(Arc::new(json[..at].to_vec()));
                    pieces.extend(self.saved.serialize_rulers(length, rulers));
                    pieces.push(Arc::new(b"\n      ".to_vec()));
                    pieces.push(Arc::new(json[at..].to_vec()));
                }
                _ => pieces.push(Arc::new(indent(to_json(solution)))),
            }
        }
        pieces.push(Arc::new(b"\n  }\n}".to_vec()));
        pieces
    }
}

/// Index of the closing bracket of the first (or, with `last`, the last) `empty` in `json`.
fn split_after(json: &[u8], empty: &[u8], last: bool) -> usize {
    let mut windows = json.windows(empty.len());
    let start = if last {
        windows.rposition(|w| w == empty)
    } else {
        windows.position(|w| w == empty)
    };
    start.unwrap() + empty.len() - 1
}

impl SaveCache {
    /// The JSON of `rulers` (as the elements of the rulers array), serializing only those
    /// found since the last save of the same length.
    fn serialize_rulers(&mut self, length: u8, rulers: &[Vec<u8>]) -> Vec<Arc<Vec<u8>>> {
        let (cached_length, done, pieces) = &mut self.rulers;
        if *cached_length != length || *done > rulers.len() {
            (*cached_length, *done) = (length, 0);
            pieces.clear();
        }
        let mut json = Vec::new();
        for (i, ruler) in rulers.iter().enumerate().skip(*done) {
            json.extend_from_slice(if i > 0 {
                b",\n        ["
            } else {
                b"\n        ["
            });
            for (j, s) in ruler.iter().enumerate() {
                if j > 0 {
                    json.extend_from_slice(b", ");
                }
                json.extend_from_slice(s.to_string().as_bytes());
            }
            json.push(b']');
        }
        *done = rulers.len();
        if !json.is_empty() {
            pieces.push(Arc::new(json));
        }
        pieces.clone()
    }
}

/// `value` as JSON, formatted like the save file.
fn to_json(value: &impl Serialize) -> Vec<u8> {
    let mut json = Vec::new();
    let mut ser = Serializer::with_formatter(&mut json, CompactArrayFormatter::new());
    value.serialize(&mut ser).unwrap();
    json
}

/// Indents JSON for a solution's place in the save file.
fn indent(json: Vec<u8>) -> Vec<u8> {
    let mut out = Vec::with_capacity(json.len() + json.len() / 8);
    for b in json {
        out.push(b);
        if b == b'\n' {
            out.extend_from_slice(b"    ");
        }
    }
    out
}

/// Pretty-printed JSON with nested arrays (the rulers) kept on one line each.
struct CompactArrayFormatter<'a> {
    inner: PrettyFormatter<'a>,
    depth: usize,
}

impl CompactArrayFormatter<'_> {
    /// Indented by two spaces.
    fn new() -> Self {
        Self {
            inner: PrettyFormatter::with_indent(b"  "),
            depth: 0,
        }
    }
}

impl Formatter for CompactArrayFormatter<'_> {
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
