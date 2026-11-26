//! Backtracking search for complete canonical rulers of one length and segment count.
//!
//! Marks are placed left to right from `m_0 = 0, m_1 = 1`, with bitsets for the marks and
//! the covered distances. ALGORITHM.md (sections 2 to 5) explains the pruning and symmetry
//! rules and how the search is split into prefixes.

use crate::ruler::is_canonical;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Most marks (segments) a ruler may have.
pub const MAX_MARKS: usize = 32;

/// A set of `64 * W` bits; bit `i` stands for position or distance `i`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Bits<const W: usize>(pub [u64; W]);

impl<const W: usize> Default for Bits<W> {
    fn default() -> Self {
        Bits([0; W])
    }
}

impl<const W: usize> Bits<W> {
    /// Sets bit `i`.
    fn set(&mut self, i: usize) {
        self.0[i >> 6] |= 1 << (i & 63);
    }

    /// Bits `lo..=hi`.
    fn range(lo: usize, hi: usize) -> Self {
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

    /// Shifts toward higher bits.
    fn shl(self, s: usize) -> Self {
        let (w, b) = (s >> 6, s & 63);
        let mut r = [0u64; W];
        for i in w..W {
            let src = i - w;
            r[i] = self.0[src] << b;
            if b > 0 && src > 0 {
                r[i] |= self.0[src - 1] >> (64 - b);
            }
        }
        Bits(r)
    }

    /// Shifts toward lower bits.
    fn shr(self, s: usize) -> Self {
        let (w, b) = (s >> 6, s & 63);
        let mut r = [0u64; W];
        for i in 0..W.saturating_sub(w) {
            let src = i + w;
            r[i] = self.0[src] >> b;
            if b > 0 && src + 1 < W {
                r[i] |= self.0[src + 1] << (64 - b);
            }
        }
        Bits(r)
    }

    /// Union.
    fn or(self, o: Self) -> Self {
        Bits(std::array::from_fn(|i| self.0[i] | o.0[i]))
    }

    /// Bits not in `o`.
    fn and_not(self, o: Self) -> Self {
        Bits(std::array::from_fn(|i| self.0[i] & !o.0[i]))
    }

    /// Intersection.
    fn and(self, o: Self) -> Self {
        Bits(std::array::from_fn(|i| self.0[i] & o.0[i]))
    }

    /// Number of set bits.
    fn count(self) -> u32 {
        self.0.iter().map(|w| w.count_ones()).sum()
    }

    /// Whether no bit is set.
    fn is_empty(self) -> bool {
        self.0 == [0; W]
    }

    /// The first `H` words (`H <= W`).
    fn low<const H: usize>(self) -> Bits<H> {
        Bits(std::array::from_fn(|i| self.0[i]))
    }

    /// Clears bit `i`.
    fn clear(&mut self, i: usize) {
        self.0[i >> 6] &= !(1 << (i & 63));
    }

    /// The lowest set bit.
    fn lowest(self) -> Option<usize> {
        self.0
            .iter()
            .enumerate()
            .find(|(_, w)| **w != 0)
            .map(|(i, w)| i * 64 + w.trailing_zeros() as usize)
    }

    /// Calls `f` with each set bit, lowest first.
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
///
/// Positions take `W` words; covered distances take `H`, since only distances up to
/// `L / 2` are tracked (measuring `d` also measures `L - d`). Bits of `cov` above `L / 2`
/// are ignored.
#[derive(Clone, Copy)]
pub struct Node<const W: usize, const H: usize> {
    pos: [u8; MAX_MARKS],
    k: usize,
    /// Bit `m` for each mark `m`.
    marks: Bits<W>,
    /// Bit `L - 1 - m` for each mark `m`.
    rev: Bits<W>,
    /// Covered cyclic distances up to `L / 2`.
    cov: Bits<H>,
    /// Bit `i` for each segment `s_i = 1` whose rotation still equals the ruler's start:
    /// `s_i, s_{i+1}, ...` matches `s_0, s_1, ...` up to the last placed segment.
    tied: u32,
}

impl<const W: usize, const H: usize> Node<W, H> {
    /// Segment `i`, between marks `i` and `i + 1`.
    fn seg(&self, i: usize) -> usize {
        (self.pos[i + 1] - self.pos[i]) as usize
    }
}

/// The search for rulers of one length and segment count. `W` and `H` are the words of its
/// position and distance bitsets: `L <= 64 * W` and `L / 2 < 64 * H`.
pub struct Search<const W: usize = 4, const H: usize = 2> {
    length: usize,
    n: usize,
    half: Bits<H>,
}

impl<const W: usize, const H: usize> Search<W, H> {
    /// The search for rulers of `length` with `num_segments` segments.
    pub fn new(length: u8, num_segments: u8) -> Self {
        let length = length as usize;
        assert!(length >= 2 && (2..=MAX_MARKS).contains(&(num_segments as usize)));
        assert!(length <= 64 * W && length / 2 < 64 * H);
        Search {
            length,
            n: num_segments as usize,
            half: Bits::range(1, length / 2),
        }
    }

    /// Distances newly measured from a mark at `p`, which lies after every mark of `node`.
    #[inline]
    fn new_distances(&self, node: &Node<W, H>, p: usize) -> Bits<W> {
        node.rev
            .shr(self.length - 1 - p)
            .or(node.marks.shl(self.length - p))
    }

    /// Smallest position the next mark may take: the segment it ends must be at least the
    /// one that follows the start of the ruler in each tied rotation.
    fn lowest(&self, node: &Node<W, H>) -> usize {
        let t = node.k - 1;
        let mut lb = 1;
        let mut tied = node.tied;
        while tied != 0 {
            let i = tied.trailing_zeros() as usize;
            tied &= tied - 1;
            lb = lb.max(node.seg(t - i));
        }
        node.pos[t] as usize + lb
    }

    /// The tied rotations after a mark at `p`, or `None` if a rotation or reflection of the
    /// ruler would then be smaller.
    fn ties(&self, node: &Node<W, H>, p: usize) -> Option<u32> {
        let t = node.k - 1;
        let s = p - node.pos[t] as usize;
        let mut tied = 0;
        let mut old = node.tied;
        while old != 0 {
            let i = old.trailing_zeros() as usize;
            old &= old - 1;
            let a = node.seg(t - i);
            if s < a {
                return None;
            }
            if s == a {
                tied |= 1 << i;
            }
        }
        // s_0 itself is the ruler's start, not a rotation.
        if s == 1 && t > 0 {
            tied |= 1 << t;
            // The reflection s_t, s_{t-1}, ..., s_0 against s_0, s_1, ..., s_t (s_t = s_0 = 1).
            for j in 1..t {
                let (a, b) = (node.seg(t - j), node.seg(j));
                if a != b {
                    if a < b {
                        return None;
                    }
                    break;
                }
            }
        }
        Some(tied)
    }

    /// `node` with a mark at `p`, given the covered distances that result.
    fn place(&self, node: &Node<W, H>, p: usize, cov: Bits<H>) -> Node<W, H> {
        let mut child = *node;
        child.pos[node.k] = p as u8;
        child.k += 1;
        child.marks.set(p);
        child.rev.set(self.length - 1 - p);
        child.cov = cov;
        child.tied = self.ties(node, p).unwrap_or(0);
        child
    }

    /// Places a mark at `p` unless the counting bound shows the result cannot be completed or
    /// the symmetry rules show it cannot be canonical.
    #[inline]
    fn child(&self, node: &Node<W, H>, p: usize) -> Option<Node<W, H>> {
        self.child_with(node, p, self.new_distances(node, p))
    }

    /// `child`, given `new_distances(node, p)`.
    #[inline]
    fn child_with(&self, node: &Node<W, H>, p: usize, new: Bits<W>) -> Option<Node<W, H>> {
        let cov = node.cov.or(new.low());
        let k = node.k + 1;
        let r = self.n - k;
        let missing = self.half.and_not(cov).count() as usize;
        if missing > r * k + r * r.saturating_sub(1) / 2 {
            return None;
        }
        let tied = self.ties(node, p)?;
        let mut child = self.place(node, p, cov);
        child.tied = tied;
        Some(child)
    }

    /// Largest position allowed for mark `j`, from `s_1 <= s_{n-1}` and room for the marks after it.
    fn limit(&self, node: &Node<W, H>, j: usize) -> usize {
        let (l, n) = (self.length, self.n);
        if j == 2 {
            // s_1 = p - 1 and the last mark (>= p + n - 3) must be <= L - s_1.
            (l + 4 - n) / 2
        } else {
            let s1 = node.pos[2] as usize - 1;
            (l - s1).saturating_sub(n - 1 - j)
        }
    }

    /// The partial ruler with marks 0 and 1, where every search starts.
    pub fn root(&self) -> Node<W, H> {
        let mut node = Node {
            pos: [0; MAX_MARKS],
            k: 1,
            marks: Bits::default(),
            rev: Bits::default(),
            cov: Bits::default(),
            tied: 0,
        };
        node.marks.set(0);
        node.rev.set(self.length - 1);
        let cov = self.new_distances(&node, 1).low();
        self.place(&node, 1, cov)
    }

    /// The partial ruler with `marks` (starting 0, 1).
    pub fn node_from_marks(&self, marks: &[u8]) -> Node<W, H> {
        let mut node = self.root();
        for &p in &marks[2..] {
            let cov = node.cov.or(self.new_distances(&node, p as usize).low());
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
                for p in self.lowest(&node)..=self.limit(&node, depth) {
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

    /// Calls `f` with the marks of each partial ruler left to search by a search that started
    /// from a partial ruler of `root` marks, has reached the marks `pos`, and tries position
    /// `next[i]` next for mark `i` (`root <= i <= pos.len()`). Together they cover exactly the
    /// rest of that search.
    pub fn remaining(&self, pos: &[u8], root: usize, next: &[u32], mut f: impl FnMut(&[u8])) {
        let k = pos.len();
        if k == self.n - 1 {
            // The last mark is placed all at once, so this node is searched again whole.
            f(pos);
        }
        let mut marks = [0u8; MAX_MARKS];
        marks[..k].copy_from_slice(pos);
        for i in root..=k.min(self.n - 2) {
            let node = self.node_from_marks(&pos[..i]);
            for p in next[i] as usize..=self.limit(&node, i) {
                if self.child(&node, p).is_some() {
                    marks[i] = p as u8;
                    f(&marks[..=i]);
                }
            }
            marks[i] = pos.get(i).copied().unwrap_or(0);
        }
    }

    /// The ruler length.
    pub fn length(&self) -> usize {
        self.length
    }

    /// The segment count.
    pub fn num_segments(&self) -> usize {
        self.n
    }

    /// The segments of a complete ruler with marks `pos`.
    pub fn segments(&self, pos: &[u8]) -> Vec<u8> {
        let mut segs: Vec<u8> = pos.windows(2).map(|w| w[1] - w[0]).collect();
        segs.push((self.length - *pos.last().unwrap() as usize) as u8);
        segs
    }
}

/// Explores every completion of a partial ruler, collecting complete canonical rulers.
pub struct Dfs<'a, const W: usize, const H: usize> {
    pub search: &'a Search<W, H>,
    /// Cleared to stop the search.
    pub running: &'a AtomicBool,
    pub eval_counter: &'a AtomicU64,
    pub found: Vec<Vec<u8>>,
    /// Whether the search stopped before finishing.
    pub interrupted: bool,
    nodes: u64,
    flushed: u64,
}

impl<'a, const W: usize, const H: usize> Dfs<'a, W, H> {
    /// A search that adds its node count to `eval_counter`.
    pub fn new(
        search: &'a Search<W, H>,
        running: &'a AtomicBool,
        eval_counter: &'a AtomicU64,
    ) -> Self {
        Dfs {
            search,
            running,
            eval_counter,
            found: Vec::new(),
            interrupted: false,
            nodes: 0,
            flushed: 0,
        }
    }

    /// Adds the nodes visited since the last flush to `eval_counter`.
    pub fn flush(&mut self) {
        self.eval_counter
            .fetch_add(self.nodes - self.flushed, Ordering::Relaxed);
        self.flushed = self.nodes;
    }

    /// Counts a node; every 65,536 nodes, flushes and checks whether to stop.
    #[inline]
    fn tick(&mut self) {
        self.nodes += 1;
        if self.nodes & 0xFFFF == 0 {
            self.flush();
            if !self.running.load(Ordering::Relaxed) {
                self.interrupted = true;
            }
        }
    }

    /// Keeps the complete ruler with marks `pos` if it is canonical.
    fn emit(&mut self, pos: &[u8]) {
        let ruler = self.search.segments(pos);
        if is_canonical(&ruler) {
            self.found.push(ruler);
        }
    }

    /// Searches every completion of `node`.
    pub fn run(&mut self, node: &Node<W, H>) {
        let s = self.search;
        let k = node.k;
        if k == s.n {
            if s.half.and_not(node.cov).is_empty() {
                self.emit(&node.pos[..k]);
            }
            return;
        }

        let lo = s.lowest(node);
        let hi = s.limit(node, k);
        if lo > hi {
            return;
        }

        if k == s.n - 1 {
            let missing = s.half.and_not(node.cov);
            // The last mark must measure the smallest missing distance, and the next smallest.
            let mut candidates = Bits::<W>::range(lo, hi);
            let mut rest = missing;
            for _ in 0..2 {
                let Some(c) = rest.lowest() else { break };
                let from_c = node.marks.shl(c).or(node.marks.shl(s.length - c));
                candidates = candidates.and(from_c);
                rest.clear(c);
            }
            let mut pos = node.pos;
            candidates.for_each_one(|p| {
                self.nodes += 1;
                if missing.and_not(s.new_distances(node, p).low()).is_empty() {
                    pos[k] = p as u8;
                    self.emit(&pos[..=k]);
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

    /// Every ruler the search finds, splitting it into about `target` prefixes.
    fn search_all<const W: usize, const H: usize>(
        length: u8,
        n: u8,
        target: usize,
    ) -> Vec<Vec<u8>> {
        let s = Search::<W, H>::new(length, n);
        let running = AtomicBool::new(true);
        let counter = AtomicU64::new(0);
        let (depth, prefixes) = s.prefixes(target, n as usize - 1, usize::MAX);
        let mut dfs = Dfs::new(&s, &running, &counter);
        for p in prefixes.chunks(depth) {
            dfs.run(&s.node_from_marks(p));
        }
        let mut out = dfs.found;
        out.sort();
        out
    }

    /// Splitting a search after it descended along its first path, at every depth, leaves
    /// exactly the rulers of the whole search (nothing before that path was searched).
    #[test]
    fn remaining_covers_rest_of_search() {
        let running = AtomicBool::new(true);
        let counter = AtomicU64::new(0);
        for (length, n) in [(24u8, 6u8), (30, 7), (31, 7)] {
            let s = <Search>::new(length, n);
            let mut whole = Dfs::new(&s, &running, &counter);
            whole.run(&s.root());
            whole.found.sort();
            assert!(!whole.found.is_empty());

            // The first path: at each mark, the first position the search descends into.
            let mut node = s.root();
            let mut next = [0u32; MAX_MARKS];
            while node.k < s.n {
                let k = node.k;
                next[k] = node.pos[k - 1] as u32 + 1;
                let mut parts = Vec::new();
                s.remaining(&node.pos[..k], 2, &next, |m| parts.push(m.to_vec()));
                let mut split = Dfs::new(&s, &running, &counter);
                for part in &parts {
                    split.run(&s.node_from_marks(part));
                }
                split.found.sort();
                assert_eq!(split.found, whole.found, "L={length} n={n} k={k}");
                if k == s.n - 1 {
                    break;
                }
                let p = (next[k] as usize..=s.limit(&node, k))
                    .find(|&p| s.child(&node, p).is_some())
                    .unwrap();
                next[k] = p as u32 + 1;
                node = s.child(&node, p).unwrap();
            }
        }
    }

    /// The widest bitsets find a complete ruler of the longest supported length, whose
    /// marks and distances reach the top bits: 15 segments of 1, then 15 of 16 (L = 255).
    #[test]
    fn finds_ruler_of_length_255() {
        let ruler: Vec<u8> = [vec![1; 15], vec![16; 15]].concat();
        let s = <Search>::new(255, 30);
        let mut marks = vec![0u8];
        for &seg in &ruler[..29] {
            marks.push(marks.last().unwrap() + seg);
        }
        let running = AtomicBool::new(true);
        let counter = AtomicU64::new(0);
        let mut dfs = Dfs::new(&s, &running, &counter);
        dfs.run(&s.node_from_marks(&marks[..26]));
        assert!(dfs.found.contains(&ruler));
    }

    #[test]
    fn matches_brute_force() {
        // Covers the minimal segment count and the one above it (which has many more
        // solutions and exercises the slack in the counting bound), with and without
        // splitting the search into prefixes.
        for length in 2..=30u8 {
            let lb = crate::ruler::num_segments_lower_bound(length).max(2);
            for n in lb..=(lb + 1).min(length) {
                let expected = brute_force(length, n);
                assert_eq!(
                    search_all::<1, 1>(length, n, 1),
                    expected,
                    "L={length} n={n}"
                );
                assert_eq!(
                    search_all::<1, 1>(length, n, 50),
                    expected,
                    "L={length} n={n}"
                );
                assert_eq!(
                    search_all::<4, 2>(length, n, 50),
                    expected,
                    "L={length} n={n}"
                );
            }
        }
    }
}
