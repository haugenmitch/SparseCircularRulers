struct Params {
    length: u32,
    num_segments: u32,
    batch_size: u32,
    start_rank_low: u32,
    start_rank_high: u32,
    steps_per_thread: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> binomial_table: array<u32>;
@group(0) @binding(2) var<storage, read_write> results: array<u32>;
@group(0) @binding(3) var<storage, read_write> found_counter: atomic<u32>;

struct u64_ {
    low: u32,
    high: u32,
}

fn u64_add(a: u64_, b: u64_) -> u64_ {
    var res: u64_;
    let low = a.low + b.low;
    let carry = u32(low < a.low);
    res.low = low;
    res.high = a.high + b.high + carry;
    return res;
}

fn u64_sub(a: u64_, b: u64_) -> u64_ {
    var res: u64_;
    let low = a.low - b.low;
    let borrow = u32(a.low < b.low);
    res.low = low;
    res.high = a.high - b.high - borrow;
    return res;
}

fn u64_lt(a: u64_, b: u64_) -> bool {
    if (a.high < b.high) { return true; }
    if (a.high > b.high) { return false; }
    return a.low < b.low;
}

fn get_binomial(n: u32, k: u32) -> u64_ {
    if (k > n || k > 32u) { return u64_(0u, 0u); }
    let index = n * 33u + k;
    return u64_(binomial_table[index * 2u], binomial_table[index * 2u + 1u]);
}

fn unrank(rank: u64_) -> array<u32, 32> {
    var ruler = array<u32, 32>();
    let n = params.num_segments;
    let length = params.length;
    
    ruler[0] = 1u;
    var current_s = length - 1u;
    var current_k = n - 1u;
    var r = rank;

    for (var i: u32 = 1; i < n - 1u; i++) {
        var val = 1u;
        loop {
            if (current_s <= val) { break; }
            let count = get_binomial(current_s - val - 1u, current_k - 2u);
            if (u64_lt(r, count)) { break; }
            r = u64_sub(r, count);
            val += 1u;
        }
        ruler[i] = val;
        current_s -= val;
        current_k -= 1u;
    }
    ruler[n - 1u] = current_s;
    return ruler;
}

fn is_complete(ruler: ptr<function, array<u32, 32>>, length: u32, n: u32) -> bool {
    var marks = array<u32, 8>(0,0,0,0,0,0,0,0);
    var cp: u32 = 0;
    marks[0] |= 1u;
    for (var i: u32 = 1; i < n; i++) {
        cp += (*ruler)[i - 1u];
        if (cp < 256u) { marks[cp >> 5] |= (1u << (cp & 31u)); }
    }

    let u32_blocks = (length + 31u) >> 5;
    let bit_shift = length & 31u;
    
    var m2 = array<u32, 16>(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
    let num_mark_blocks = (length + 31u) >> 5;
    for (var i: u32 = 0; i < num_mark_blocks; i++) {
        m2[i] |= marks[i];
        let shift_blocks = length >> 5;
        let shift_bits = length & 31u;
        if (shift_bits == 0) {
            if (i + shift_blocks < 16u) { m2[i + shift_blocks] |= marks[i]; }
        } else {
            if (i + shift_blocks < 16u) { m2[i + shift_blocks] |= marks[i] << shift_bits; }
            if (i + shift_blocks + 1u < 16u) { m2[i + shift_blocks + 1u] |= marks[i] >> (32u - shift_bits); }
        }
    }

    var diffs = array<u32, 8>(0,0,0,0,0,0,0,0);
    for (var j: u32 = 0; j < num_mark_blocks; j++) { diffs[j] |= m2[j]; }

    cp = 0;
    for (var i: u32 = 1; i < n; i++) {
        cp += (*ruler)[i - 1u];
        let us = cp >> 5;
        let bs = cp & 31u;
        if (bs == 0u) {
            for (var j: u32 = 0; j < num_mark_blocks; j++) { if (j + us < 16u) { diffs[j] |= m2[j + us]; } }
        } else {
            for (var j: u32 = 0; j < num_mark_blocks; j++) {
                let low = m2[j + us];
                var high = 0u;
                if (j + us + 1u < 16u) { high = m2[j + us + 1u]; }
                diffs[j] |= (low >> bs) | (high << (32u - bs));
            }
        }
    }

    let full_blocks = length >> 5;
    for (var i: u32 = 0; i < full_blocks; i++) {
        if (diffs[i] != 0xFFFFFFFFu) { return false; }
    }
    if (bit_shift > 0u) {
        let mask = (1u << bit_shift) - 1u;
        if ((diffs[full_blocks] & mask) != mask) { return false; }
    }
    return true;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_groups: vec3<u32>) {
    let thread_idx = global_id.x + global_id.y * num_groups.x * 256u;
    if (thread_idx >= params.batch_size) { return; }
    
    var current_rank = u64_add(u64_(params.start_rank_low, params.start_rank_high), u64_(thread_idx * params.steps_per_thread, 0u));
    
    var ruler = unrank(current_rank);
    let n = params.num_segments;
    let length = params.length;

    for (var s: u32 = 0; s < params.steps_per_thread; s++) {
        var canon = true;
        if (n >= 3u) {
            if (ruler[1] < ruler[n - 1u]) { canon = true; }
            else if (ruler[1] > ruler[n - 1u]) { canon = false; }
            else {
                let m = (n - 1u) / 2u;
                for (var i: u32 = 2; i <= m; i++) {
                    if (ruler[i] < ruler[n - i]) { canon = true; break; }
                    if (ruler[i] > ruler[n - i]) { canon = false; break; }
                }
            }
        }

        if (canon) {
            if (is_complete(&ruler, length, n)) {
                let idx = atomicAdd(&found_counter, 1u);
                if (idx < 65536u) {
                    results[idx * 2u] = current_rank.low;
                    results[idx * 2u + 1u] = current_rank.high;
                }
            }
        } else {
            // Symmetry-Breaking Skip: If ruler[1] > ruler[n-1], we can skip the remaining
            // combinations for the current s_1, s_2, ..., s_{n-2} values.
            if (n >= 3u && ruler[1] > ruler[n - 1u] && ruler[n - 1u] > 1u) {
                let skip = ruler[n - 1u] - 1u;
                let steps_left = params.steps_per_thread - s - 1u;
                let actual_skip = min(skip, steps_left);
                
                current_rank = u64_add(current_rank, u64_(actual_skip, 0u));
                ruler[n - 2u] += actual_skip;
                ruler[n - 1u] -= actual_skip;
                s += actual_skip;
            }
        }

        // Increment
        current_rank = u64_add(current_rank, u64_(1u, 0u));
        if (n > 2u && ruler[n - 1u] > 1u) {
            ruler[n - 1u] -= 1u;
            ruler[n - 2u] += 1u;
        } else {
            var f = false;
            for (var i: i32 = i32(n) - 2; i >= 2; i--) {
                if (ruler[u32(i)] > 1u) {
                    ruler[u32(i)] = 1u;
                    ruler[u32(i - 1)] += 1u;
                    var sum: u32 = 0;
                    for (var j: u32 = 0; j < n - 1u; j++) { sum += ruler[j]; }
                    ruler[n - 1u] = length - sum;
                    f = true;
                    break;
                }
            }
            if (!f) { break; }
        }
    }
}
