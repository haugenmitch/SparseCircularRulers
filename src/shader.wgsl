// Backtracking search for complete circular rulers; each thread searches one partial ruler,
// in steps of at most node_budget nodes per pass.
// This mirrors `Dfs::run` in search.rs (see there for the pruning rules), except that the
// rotation and reflection rules are only applied to the partial rulers handed to the GPU:
// checking them here needs per-depth state indexed by each thread's own depth, which cost
// more GPU time than the nodes they cut saved.
//
// WORDS (32-bit words per bitset), HWORDS, and MARKS are substituted when the pipeline is
// built, so every bitset loop has a constant bound and unrolls. Words are only ever indexed
// by constants (`word` picks a runtime index with selects), which keeps the bitsets in
// registers. The per-depth arrays (marks and covered distances) are kept in registers too
// and indexed by each thread's own depth, which costs code for every entry, so they hold only
// MARKS (num_segments) entries: sized for 32 marks, the shader used twice the registers and
// ran about 1.4x slower at length 130.
//
// Some forms crash the radv (Mesa 26) shader compiler and are avoided: early returns
// inside loops, trailing-zero builtins inside loops, clamped constant indices such as
// a[max(i, 1u) - 1u], and short loops ending in `break` that the compiler unrolls.

const WORDS: u32 = __WORDS__u;
// Words that hold distances up to length / 2, the only ones tracked (measuring d also
// measures length - d); covered distances are stored per depth in this many words.
const HWORDS: u32 = __HWORDS__u;
// Marks per ruler (num_segments): the per-depth arrays hold no more.
const MARKS: u32 = __MARKS__u;

struct Params {
    length: u32,
    num_segments: u32,
    num_work: u32,
    _p1: u32,
    words: u32,
    max_results: u32,
    node_budget: u32,
    _p0: u32,
};

const RESUME: u32 = 0xFFFFFFFFu;
// Saved search state per slot: depth k, then pos[32], then nextp[32], then the depth d of
// the slot's partial ruler.
const STATE_STRIDE: u32 = 66u;

@group(0) @binding(0) var<uniform> params: Params;
// Per thread: the slot it works in, and the offset in `prefixes` of the partial ruler to start
// there (its depth, then its marks) or RESUME to continue the slot's saved search.
@group(0) @binding(1) var<storage, read> work: array<u32>;
@group(0) @binding(2) var<storage, read> prefixes: array<u32>;
// Marks of complete rulers, num_segments per ruler.
@group(0) @binding(3) var<storage, read_write> results: array<u32>;
// [0] rulers found, [1] and [2] low and high words of the node count, [3] slots finished.
@group(0) @binding(4) var<storage, read_write> counters: array<atomic<u32>, 4>;
@group(0) @binding(5) var<storage, read_write> state: array<u32>;
// Slots whose partial ruler was fully searched in this pass.
@group(0) @binding(6) var<storage, read_write> finished: array<u32>;

alias Bits = array<u32, WORDS>;
alias HBits = array<u32, HWORDS>;

// The first HWORDS words of a.
fn narrow(a: Bits) -> HBits {
    var r: HBits;
    for (var i = 0u; i < HWORDS; i++) { r[i] = a[i]; }
    return r;
}

fn widen(a: HBits) -> Bits {
    var r: Bits;
    for (var i = 0u; i < HWORDS; i++) { r[i] = a[i]; }
    return r;
}

// Word j of a, or 0 if j is out of range.
fn word(a: Bits, j: u32) -> u32 {
    var v = 0u;
    for (var t = 0u; t < WORDS; t++) { v = select(v, a[t], t == j); }
    return v;
}

fn hbits_or(a: HBits, b: HBits) -> HBits {
    var r: HBits;
    for (var i = 0u; i < HWORDS; i++) { r[i] = a[i] | b[i]; }
    return r;
}

// Bits of `half` not in `c`.
fn hbits_missing(half: HBits, c: HBits) -> u32 {
    var n = 0u;
    for (var i = 0u; i < HWORDS; i++) { n += countOneBits(half[i] & ~c[i]); }
    return n;
}

fn bits_or(a: Bits, b: Bits) -> Bits {
    var r: Bits;
    for (var i = 0u; i < WORDS; i++) { r[i] = a[i] | b[i]; }
    return r;
}

fn bits_and(a: Bits, b: Bits) -> Bits {
    var r: Bits;
    for (var i = 0u; i < WORDS; i++) { r[i] = a[i] & b[i]; }
    return r;
}

fn bits_and_not(a: Bits, b: Bits) -> Bits {
    var r: Bits;
    for (var i = 0u; i < WORDS; i++) { r[i] = a[i] & ~b[i]; }
    return r;
}

fn bits_count(a: Bits) -> u32 {
    var c = 0u;
    for (var i = 0u; i < WORDS; i++) { c += countOneBits(a[i]); }
    return c;
}

fn bits_is_empty(a: Bits) -> bool {
    var any = 0u;
    for (var i = 0u; i < WORDS; i++) { any |= a[i]; }
    return any == 0u;
}

// Lowest set bit, or 0xFFFFFFFF if none.
fn bits_lowest(a: Bits) -> u32 {
    var res = 0xFFFFFFFFu;
    for (var i = WORDS; i > 0u; i--) {
        let x = a[i - 1u];
        res = select(res, (i - 1u) * 32u + countOneBits((x & (0u - x)) - 1u), x != 0u);
    }
    return res;
}

fn bits_shl(a: Bits, s: u32) -> Bits {
    var r: Bits;
    let w = s >> 5u;
    let b = s & 31u;
    for (var i = 0u; i < WORDS; i++) {
        // i - w wraps around to a huge index (so word() gives 0) when i < w.
        let lo = word(a, i - w);
        let hi = word(a, i - w - 1u);
        r[i] = (lo << b) | select(0u, hi >> (32u - b), b > 0u);
    }
    return r;
}

fn bits_shr(a: Bits, s: u32) -> Bits {
    var r: Bits;
    let w = s >> 5u;
    let b = s & 31u;
    for (var i = 0u; i < WORDS; i++) {
        let lo = word(a, i + w);
        let hi = word(a, i + w + 1u);
        r[i] = (lo >> b) | select(0u, hi << (32u - b), b > 0u);
    }
    return r;
}

fn bits_shl1(a: Bits) -> Bits {
    var r: Bits;
    for (var i = 0u; i < WORDS; i++) {
        r[i] = (a[i] << 1u) | (word(a, i - 1u) >> 31u);
    }
    return r;
}

fn bits_shr1(a: Bits) -> Bits {
    var r: Bits;
    for (var i = 0u; i < WORDS; i++) {
        r[i] = (a[i] >> 1u) | (word(a, i + 1u) << 31u);
    }
    return r;
}

// Bits lo..=hi.
fn bits_range(lo: u32, hi: u32) -> Bits {
    var r: Bits;
    for (var i = 0u; i < WORDS; i++) {
        let base = i * 32u;
        let a = max(lo, base);
        let b = min(hi, base + 31u);
        let width = b - a + 1u;
        let m = select((1u << width) - 1u, 0xFFFFFFFFu, width == 32u);
        r[i] = select(0u, m << (a - base), lo <= hi && a <= b);
    }
    return r;
}

fn bits_toggle(a: Bits, i: u32) -> Bits {
    var r: Bits;
    for (var t = 0u; t < WORDS; t++) {
        r[t] = a[t] ^ select(0u, 1u << (i & 31u), t == (i >> 5u));
    }
    return r;
}

// Distances measured from a new mark at p, which lies after every existing mark.
fn new_distances(marks: Bits, rev: Bits, p: u32) -> Bits {
    let l = params.length;
    return bits_or(bits_shr(rev, l - 1u - p), bits_shl(marks, l - p));
}

// Positions m + c and m + L - c for each mark m: where a new mark measures distance c.
fn measuring(marks: Bits, c: u32) -> Bits {
    return bits_or(bits_shl(marks, c), bits_shl(marks, params.length - c));
}

// Largest position allowed for mark j (s_1 <= s_{n-1} and room for later marks).
fn limit(j: u32, pos2: u32) -> u32 {
    let l = params.length;
    let n = params.num_segments;
    if (j == 2u) { return (l + 4u - n) / 2u; }
    let a = l - (pos2 - 1u);
    let b = n - 1u - j;
    return select(0u, a - b, a >= b);
}

// Reserves `count` result entries, all or none, so a ruler is never dropped or reported twice.
// Returns the first entry, or RESUME if they do not fit. A thread that does not fit takes its
// count back; while it is added, every other thread's reservation fails too, so the entries
// handed out stay contiguous.
fn reserve_results(count: u32) -> u32 {
    let first = atomicAdd(&counters[0], count);
    if (first + count > params.max_results) {
        atomicSub(&counters[0], count);
        return RESUME;
    }
    return first;
}

// Each thread searches the partial ruler in its slot for up to node_budget nodes, then saves
// where it got to, so no dispatch runs long enough to trip the driver's GPU timeout.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.num_work) { return; }

    let l = params.length;
    let n = params.num_segments;
    let half = bits_range(1u, l / 2u);
    let slot = work[idx * 2u];
    let prefix = work[idx * 2u + 1u];
    let base = slot * STATE_STRIDE;

    var pos: array<u32, MARKS>;
    // Next position to try for mark k (for marks below k it is one past their position).
    var np: u32;
    // Covered distances at each depth (only those up to l / 2 count).
    var covs: array<HBits, MARKS>;
    var marks: Bits;
    var rev: Bits;
    var k: u32;
    // Depth of the slot's partial ruler: the search backtracks no further.
    var d: u32;

    if (prefix != RESUME) {
        d = prefixes[prefix];
        for (var i = 0u; i < d; i++) { pos[i] = prefixes[prefix + 1u + i]; }
        k = d;
        np = pos[d - 1u] + 1u;
    } else {
        k = state[base];
        d = state[base + 65u];
        for (var i = 0u; i < k; i++) { pos[i] = state[base + 1u + i]; }
        np = state[base + 33u + k];
    }

    // Rebuild the marks and the covered distances at each depth from the marks.
    var cov: Bits;
    for (var i = 0u; i < k; i++) {
        let p = pos[i];
        if (i > 0u) { cov = bits_or(cov, new_distances(marks, rev, p)); }
        marks = bits_toggle(marks, p);
        rev = bits_toggle(rev, l - 1u - p);
        if (i + 1u >= d) { covs[i + 1u] = narrow(cov); }
    }

    var nodes = 0u;
    var done = false;
    loop {
        if (nodes >= params.node_budget) { break; }

        if (k == n - 1u) {
            // Last mark: only positions that measure the two smallest missing distances.
            let lo = pos[k - 1u] + 1u;
            let hi = limit(k, pos[2]);
            if (lo <= hi) {
                let missing = bits_and_not(half, widen(covs[k]));
                var cand = bits_range(lo, hi);
                let d0 = bits_lowest(missing);
                if (d0 != 0xFFFFFFFFu) {
                    cand = bits_and(cand, measuring(marks, d0));
                    let zero = Bits();
                    let pair = bits_or(bits_toggle(zero, d0), bits_toggle(zero, l - d0));
                    let d1 = bits_lowest(bits_and_not(missing, pair));
                    if (d1 != 0xFFFFFFFFu) { cand = bits_and(cand, measuring(marks, d1)); }
                }
                // Explicitly zeroed: a `var` declared in a loop without an initializer is not
                // reset on each iteration here.
                var good = Bits();
                for (var w = 0u; w < WORDS; w++) {
                    var bits = cand[w];
                    loop {
                        if (bits == 0u) { break; }
                        let p = w * 32u + countOneBits((bits & (0u - bits)) - 1u);
                        bits &= bits - 1u;
                        nodes += 1u;
                        if (bits_is_empty(bits_and_not(missing, new_distances(marks, rev, p)))) {
                            good = bits_toggle(good, p);
                        }
                    }
                }
                let count = bits_count(good);
                if (count > 0u) {
                    let first = reserve_results(count);
                    // No room left this pass: stop here and redo this mark next pass.
                    if (first == RESUME) { break; }
                    var slot_i = first;
                    for (var w = 0u; w < WORDS; w++) {
                        var bits = good[w];
                        loop {
                            if (bits == 0u) { break; }
                            let p = w * 32u + countOneBits((bits & (0u - bits)) - 1u);
                            bits &= bits - 1u;
                            for (var i = 0u; i < k; i++) { results[slot_i * n + i] = pos[i]; }
                            results[slot_i * n + k] = p;
                            slot_i += 1u;
                        }
                    }
                }
            }
        } else {
            // Try positions p = np..=limit; moving p by one shifts its new distances by one.
            let ck = covs[k];
            let hhalf = narrow(half);
            let lim = limit(k, pos[2]);
            let k1 = k + 1u;
            let r = n - k1;
            let bound = r * k1 + (r * (r - 1u)) / 2u;
            var p = np;
            var up: Bits;
            var down: Bits;
            if (p <= lim) {
                up = bits_shr(rev, l - 1u - p);
                down = bits_shl(marks, l - p);
            }
            var descended = false;
            loop {
                if (p > lim) { break; }
                nodes += 1u;
                // Only distances up to l / 2 count, so only those words are compared.
                let nc = hbits_or(ck, narrow(bits_or(up, down)));
                if (hbits_missing(hhalf, nc) <= bound) {
                    pos[k] = p;
                    marks = bits_toggle(marks, p);
                    rev = bits_toggle(rev, l - 1u - p);
                    covs[k1] = nc;
                    k = k1;
                    np = p + 1u;
                    descended = true;
                    break;
                }
                up = bits_shl1(up);
                down = bits_shr1(down);
                p += 1u;
            }
            if (descended) { continue; }
        }

        // Backtrack.
        if (k == d) { done = true; break; }
        k -= 1u;
        np = pos[k] + 1u;
        marks = bits_toggle(marks, pos[k]);
        rev = bits_toggle(rev, l - 1u - pos[k]);
    }

    if (done) {
        finished[atomicAdd(&counters[3], 1u)] = slot;
    } else {
        state[base] = k;
        state[base + 65u] = d;
        for (var i = 0u; i < k; i++) { state[base + 1u + i] = pos[i]; }
        // The host reads a next position for every mark from d to k.
        for (var i = d; i < k; i++) { state[base + 33u + i] = pos[i] + 1u; }
        state[base + 33u + k] = np;
    }

    let old = atomicAdd(&counters[1], nodes);
    if (old + nodes < old) { atomicAdd(&counters[2], 1u); }
}
