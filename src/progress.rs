//! Terminal display of the current search: status line, prefixes searched, and throughput.

use crate::length::Counters;
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressState, ProgressStyle};
use std::fmt::Write;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

const REFRESH: Duration = Duration::from_millis(50);

/// The progress bars and the latest ruler they show.
pub struct Progress {
    multi: MultiProgress,
    status: ProgressBar,
    prefixes: ProgressBar,
    stats: ProgressBar,
    /// The most recent ruler found, shown after the throughput.
    pub latest_ruler: Mutex<Option<Vec<u8>>>,
}

impl Progress {
    /// Progress bars on stdout, or hidden if `quiet`.
    pub fn new(quiet: bool) -> Self {
        let multi = MultiProgress::with_draw_target(if quiet {
            ProgressDrawTarget::hidden()
        } else {
            ProgressDrawTarget::stdout()
        });
        let add = |bar: ProgressBar, style: ProgressStyle| {
            let bar = multi.add(bar);
            bar.enable_steady_tick(Duration::from_millis(100));
            bar.set_style(style);
            bar
        };
        let status = add(
            ProgressBar::new_spinner(),
            ProgressStyle::default_spinner()
                .template("{spinner:.green} Length {msg}")
                .unwrap(),
        );
        let prefixes = add(
            ProgressBar::new(0),
            ProgressStyle::default_bar()
                .template("  [{bar:40.cyan/blue}] {pos}/{len} ({percent}%)")
                .unwrap()
                .progress_chars("#>-"),
        );
        let stats = add(
            ProgressBar::new(0),
            ProgressStyle::default_spinner()
                .template("  {elapsed_precise} [{evals_per_sec} evals/s]{msg}")
                .unwrap()
                .with_key(
                    "evals_per_sec",
                    |state: &ProgressState, w: &mut dyn Write| {
                        write_thousands(state.per_sec() as u64, w)
                    },
                ),
        );
        Progress {
            multi,
            status,
            prefixes,
            stats,
            latest_ruler: Mutex::new(None),
        }
    }

    /// Resets the prefix bar for a search of `total` prefixes, `done` of them already searched.
    pub fn start(&self, done: u64, total: u64) {
        self.prefixes.set_length(total);
        self.prefixes.set_position(done);
    }

    /// Refreshes the display from `counters` until `stop` is set. The elapsed time is
    /// `base_time` plus the time since `start`.
    pub fn update_until(
        &self,
        stop: &AtomicBool,
        (length, num_segments): (u8, u8),
        counters: &Counters,
        base_time: Duration,
        start: Instant,
    ) {
        while !stop.load(Ordering::Relaxed) {
            self.prefixes
                .set_position(counters.done.load(Ordering::Relaxed));
            self.stats
                .set_position(counters.evaluated.load(Ordering::Relaxed));
            self.status.set_message(format!(
                "{}: Found {} rulers with {} segments in {:?}",
                length,
                counters.found.load(Ordering::Relaxed),
                num_segments,
                base_time + start.elapsed()
            ));
            let latest = match self.latest_ruler.try_lock().as_deref() {
                Ok(Some(ruler)) => format!(" | Latest: {ruler:?}"),
                _ => String::new(),
            };
            self.stats.set_message(latest);
            std::thread::sleep(REFRESH);
        }
    }

    /// Runs `f` (which prints) with the display hidden.
    pub fn suspend<R>(&self, f: impl FnOnce() -> R) -> R {
        self.multi.suspend(f)
    }

    /// Removes the progress bars.
    pub fn clear(&self) {
        self.status.finish_and_clear();
        self.prefixes.finish_and_clear();
        self.stats.finish_and_clear();
    }
}

/// Writes `n` with commas between groups of three digits.
fn write_thousands(n: u64, w: &mut dyn Write) {
    let s = n.to_string();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            let _ = w.write_char(',');
        }
        let _ = w.write_char(c);
    }
}
