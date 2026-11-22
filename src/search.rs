//! Backtracking search for complete circular rulers.
//!
//! A ruler with `n` segments summing to `L` is described by its marks
//! `0 = m_0 < m_1 < ... < m_{n-1} < L`. Since some segment must be 1 (to measure
//! distance 1), rotating it to the front fixes `m_0 = 0` and `m_1 = 1`.
//!
//! Marks are placed left to right. For every partial ruler we keep a bitset of
//! the covered cyclic distances; covering `d` always covers `L - d`, so the
//! distances form `floor(L / 2)` classes. Two pruning rules cut the tree:
//!
//! * Counting bound: with `k` marks placed and `r` left, at most
//!   `r * k + r * (r - 1) / 2` new pairs (and so classes) can still be covered.
//! * Last mark: the smallest missing distance `c` must be measured from the last
//!   mark `p`, so `p` is one of `m_i + c` or `m_i + L - c`. Only those (at most
//!   `2 * (n - 1)`) positions, narrowed further by the next smallest missing
//!   distance, are checked instead of every position.
//!
//! The only symmetry rule applied during the search is `s_1 <= s_{n-1}`, which
//! every canonical ruler satisfies (it must not exceed its reflection about
//! `m_0`). The full `is_canonical` check runs only on complete rulers.

use crate::is_canonical;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

pub const MAX_MARKS: usize = 32;

/// A 256-bit set; bit `i` stands for position or distance `i`.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub struct Bits(pub [u64; 4]);

impl Bits {
    fn set(&mut self, i: usize) {
        self.0[i >> 6] |= 1 << (i & 63);
    }

    /// Bits `lo..=hi`.
    fn range(lo: usize, hi: usize) -> Bits {
        let mut r = Bits::default();
        if lo > hi {
            return r;
        }
        for (w, word) in r.0.iter_mut().enumerate() {
            let base = w * 64;
            let a = lo.max(base);
            let b = hi.min(base + 63);
            if a <= b {
                let width = b - a + 1;
                let m = if width == 64 { !0 } else { (1u64 << width) - 1 };
                *word = m << (a - base);
            }
        }
        r
    }

    fn shl(self, s: usize) -> Bits {
        let (w, b) = (s >> 6, s & 63);
        let mut r = [0u64; 4];
        for i in w..4 {
            let src = i - w;
            r[i] = self.0[src] << b;
            if b > 0 && src > 0 {
                r[i] |= self.0[src - 1] >> (64 - b);
            }
        }
        Bits(r)
    }

    fn shr(self, s: usize) -> Bits {
        let (w, b) = (s >> 6, s & 63);
        let mut r = [0u64; 4];
        for i in 0..4usize.saturating_sub(w) {
            let src = i + w;
            r[i] = self.0[src] >> b;
            if b > 0 && src + 1 < 4 {
                r[i] |= self.0[src + 1] << (64 - b);
            }
        }
        Bits(r)
    }

    fn or(self, o: Bits) -> Bits {
        Bits(std::array::from_fn(|i| self.0[i] | o.0[i]))
    }

    fn and_not(self, o: Bits) -> Bits {
        Bits(std::array::from_fn(|i| self.0[i] & !o.0[i]))
    }

    fn and(self, o: Bits) -> Bits {
        Bits(std::array::from_fn(|i| self.0[i] & o.0[i]))
    }

    fn count(self) -> u32 {
        self.0.iter().map(|w| w.count_ones()).sum()
    }

    fn is_empty(self) -> bool {
        self.0 == [0; 4]
    }

    fn lowest(self) -> Option<usize> {
        self.0
            .iter()
            .enumerate()
            .find(|(_, w)| **w != 0)
            .map(|(i, w)| i * 64 + w.trailing_zeros() as usize)
    }

    fn for_each_one(self, mut f: impl FnMut(usize)) {
        for (i, &w) in self.0.iter().enumerate() {
            let mut w = w;
            while w != 0 {
                f(i * 64 + w.trailing_zeros() as usize);
                w &= w - 1;
            }
        }
    }
}

/// A partial ruler: the first `k` marks, their bitsets, and the covered distances.
#[derive(Clone, Copy)]
pub struct Node {
    pos: [u8; MAX_MARKS],
    k: usize,
    /// Bit `m` for each mark `m`.
    marks: Bits,
    /// Bit `L - 1 - m` for each mark `m`.
    rev: Bits,
    /// Covered cyclic distances (symmetric: `d` set iff `L - d` set).
    cov: Bits,
}

pub struct Search {
    length: usize,
    n: usize,
    full: Bits,
    half: Bits,
}

impl Search {
    pub fn new(length: u8, num_segments: u8) -> Self {
        let length = length as usize;
        assert!(length >= 2 && (2..=MAX_MARKS).contains(&(num_segments as usize)));
        Search {
            length,
            n: num_segments as usize,
            full: Bits::range(1, length - 1),
            half: Bits::range(1, length / 2),
        }
    }

    /// Distances newly measured from a mark at `p`, which lies after every mark of `node`.
    #[inline]
    fn new_distances(&self, node: &Node, p: usize) -> Bits {
        node.rev
            .shr(self.length - 1 - p)
            .or(node.marks.shl(self.length - p))
    }

    fn place(&self, node: &Node, p: usize, cov: Bits) -> Node {
        let mut child = *node;
        child.pos[node.k] = p as u8;
        child.k += 1;
        child.marks.set(p);
        child.rev.set(self.length - 1 - p);
        child.cov = cov;
        child
    }

    /// Places a mark at `p` unless the counting bound shows the result cannot be completed.
    #[inline]
    fn child(&self, node: &Node, p: usize) -> Option<Node> {
        self.child_with(node, p, self.new_distances(node, p))
    }

    /// `child`, given `new_distances(node, p)`.
    #[inline]
    fn child_with(&self, node: &Node, p: usize, new: Bits) -> Option<Node> {
        let cov = node.cov.or(new);
        let k = node.k + 1;
        let r = self.n - k;
        let missing = self.half.and_not(cov).count() as usize;
        (missing <= r * k + r * r.saturating_sub(1) / 2).then(|| self.place(node, p, cov))
    }

    /// Largest position allowed for mark `j`, from `s_1 <= s_{n-1}` and room for the marks after it.
    fn limit(&self, node: &Node, j: usize) -> usize {
        let (l, n) = (self.length, self.n);
        if j == 2 {
            // s_1 = p - 1 and the last mark (>= p + n - 3) must be <= L - s_1.
            (l + 4 - n) / 2
        } else {
            let s1 = node.pos[2] as usize - 1;
            (l - s1).saturating_sub(n - 1 - j)
        }
    }

    pub fn root(&self) -> Node {
        let mut node = Node {
            pos: [0; MAX_MARKS],
            k: 1,
            marks: Bits::default(),
            rev: Bits::default(),
            cov: Bits::default(),
        };
        node.marks.set(0);
        node.rev.set(self.length - 1);
        let cov = self.new_distances(&node, 1);
        self.place(&node, 1, cov)
    }

    pub fn node_from_marks(&self, marks: &[u8]) -> Node {
        let mut node = self.root();
        for &p in &marks[2..] {
            let cov = node.cov.or(self.new_distances(&node, p as usize));
            node = self.place(&node, p as usize, cov);
        }
        node
    }

    /// Lists the surviving partial rulers (as marks) at the shallowest depth that has at least
    /// `target` of them, without going deeper than `max_depth` marks or holding more than `cap`.
    /// Prefixes are in lexicographic order. Returns the depth and the prefixes flattened.
    pub fn prefixes(&self, target: usize, max_depth: usize, cap: usize) -> (usize, Vec<u8>) {
        let mut depth = 2;
        let mut level = vec![0u8, 1];
        while depth < max_depth.min(self.n) && level.len() / depth < target {
            let mut next = Vec::new();
            for prefix in level.chunks(depth) {
                let node = self.node_from_marks(prefix);
                for p in prefix[depth - 1] as usize + 1..=self.limit(&node, depth) {
                    if let Some(child) = self.child(&node, p) {
                        next.extend_from_slice(&child.pos[..depth + 1]);
                    }
                }
                if next.len() / (depth + 1) > cap {
                    return (depth, level);
                }
            }
            level = next;
            depth += 1;
        }
        (depth, level)
    }

    pub fn length(&self) -> usize {
        self.length
    }

    pub fn num_segments(&self) -> usize {
        self.n
    }

    pub fn segments(&self, pos: &[u8]) -> Vec<u8> {
        let mut segs: Vec<u8> = pos.windows(2).map(|w| w[1] - w[0]).collect();
        segs.push((self.length - *pos.last().unwrap() as usize) as u8);
        segs
    }
}

/// Explores every completion of a partial ruler, collecting complete canonical rulers.
pub struct Dfs<'a> {
    pub search: &'a Search,
    pub interrupt: &'a AtomicBool,
    pub eval_counter: &'a AtomicU64,
    pub found: Vec<Vec<u8>>,
    pub interrupted: bool,
    nodes: u64,
    flushed: u64,
}

impl<'a> Dfs<'a> {
    pub fn new(search: &'a Search, interrupt: &'a AtomicBool, eval_counter: &'a AtomicU64) -> Self {
        Dfs {
            search,
            interrupt,
            eval_counter,
            found: Vec::new(),
            interrupted: false,
            nodes: 0,
            flushed: 0,
        }
    }

    pub fn flush(&mut self) {
        self.eval_counter
            .fetch_add(self.nodes - self.flushed, Ordering::Relaxed);
        self.flushed = self.nodes;
    }

    #[inline]
    fn tick(&mut self) {
        self.nodes += 1;
        if self.nodes & 0xFFFF == 0 {
            self.flush();
            if !self.interrupt.load(Ordering::Relaxed) {
                self.interrupted = true;
            }
        }
    }

    fn emit(&mut self, pos: &[u8]) {
        let ruler = self.search.segments(pos);
        if is_canonical(&ruler) {
            self.found.push(ruler);
        }
    }

    pub fn run(&mut self, node: &Node) {
        let s = self.search;
        let k = node.k;
        if k == s.n {
            if s.full.and_not(node.cov).is_empty() {
                self.emit(&node.pos[..k]);
            }
            return;
        }

        let lo = node.pos[k - 1] as usize + 1;
        let hi = s.limit(node, k);
        if lo > hi {
            return;
        }

        if k == s.n - 1 {
            let missing = s.full.and_not(node.cov);
            let range = Bits::range(lo, hi);
            // The last mark must measure the smallest missing distance, and the next smallest.
            let mut candidates = range;
            let mut rest = missing;
            for _ in 0..2 {
                let Some(c) = rest.lowest() else { break };
                let from_c = node.marks.shl(c).or(node.marks.shl(s.length - c));
                candidates = candidates.and(from_c);
                rest.0[c >> 6] &= !(1 << (c & 63));
                rest.0[(s.length - c) >> 6] &= !(1 << ((s.length - c) & 63));
            }
            let mut pos = node.pos;
            candidates.for_each_one(|p| {
                self.nodes += 1;
                if missing.and_not(s.new_distances(node, p)).is_empty() {
                    pos[k] = p as u8;
                    let ruler = s.segments(&pos[..=k]);
                    if is_canonical(&ruler) {
                        self.found.push(ruler);
                    }
                }
            });
            return;
        }

        // Moving the new mark from p to p + 1 lengthens each distance p - m by one and
        // shortens each L - (p - m) by one, so the two halves of new_distances shift by one bit.
        let mut up = node.rev.shr(s.length - 1 - lo);
        let mut down = node.marks.shl(s.length - lo);
        for p in lo..=hi {
            self.tick();
            if self.interrupted {
                return;
            }
            if let Some(child) = s.child_with(node, p, up.or(down)) {
                self.run(&child);
                if self.interrupted {
                    return;
                }
            }
            up = up.shl(1);
            down = down.shr(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every complete canonical ruler, found by checking all compositions with s_0 = 1.
    fn brute_force(length: u8, n: u8) -> Vec<Vec<u8>> {
        fn complete(r: &[u8]) -> bool {
            let l: usize = r.iter().map(|&x| x as usize).sum();
            let mut seen = vec![false; l];
            for i in 0..r.len() {
                let mut d = 0;
                for j in 0..r.len() {
                    d += r[(i + j) % r.len()] as usize;
                    seen[d % l] = true;
                }
            }
            seen[1..].iter().all(|&s| s)
        }
        fn rec(r: &mut Vec<u8>, left: u8, n: usize, out: &mut Vec<Vec<u8>>) {
            if r.len() == n - 1 {
                r.push(left);
                if complete(r) && is_canonical(r) {
                    out.push(r.clone());
                }
                r.pop();
                return;
            }
            for s in 1..=left - (n - 1 - r.len()) as u8 {
                r.push(s);
                rec(r, left - s, n, out);
                r.pop();
            }
        }
        let mut out = Vec::new();
        rec(&mut vec![1], length - 1, n as usize, &mut out);
        out.sort();
        out
    }

    fn search_all(length: u8, n: u8, target: usize) -> Vec<Vec<u8>> {
        let s = Search::new(length, n);
        let interrupt = AtomicBool::new(true);
        let counter = AtomicU64::new(0);
        let (depth, prefixes) = s.prefixes(target, n as usize - 1, usize::MAX);
        let mut dfs = Dfs::new(&s, &interrupt, &counter);
        for p in prefixes.chunks(depth) {
            dfs.run(&s.node_from_marks(p));
        }
        let mut out = dfs.found;
        out.sort();
        out
    }

    #[test]
    fn matches_brute_force() {
        // Covers the minimal segment count and the one above it (which has many more
        // solutions and exercises the slack in the counting bound), with and without
        // splitting the search into prefixes.
        for length in 2..=30u8 {
            let lb = crate::get_num_segments_lower_bound(length).max(2);
            for n in lb..=(lb + 1).min(length) {
                let expected = brute_force(length, n);
                assert_eq!(search_all(length, n, 1), expected, "L={length} n={n}");
                assert_eq!(search_all(length, n, 50), expected, "L={length} n={n}");
            }
        }
    }
}
