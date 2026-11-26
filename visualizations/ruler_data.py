"""Loading and vectorized analysis of sparse circular ruler results.

Results files store every ruler as a JSON list, so `json.load` on a large file
builds tens of millions of Python objects (the 595 MB v0.10.2 file needs more
than 26 GB of memory that way). This module instead pulls the ruler arrays out
of the raw bytes with a regular expression, parses the numbers with numpy, and
keeps one `uint16` matrix of shape (rulers, segments) per length. The parsed
result is cached as an `.npz` file keyed by the source file's size and mtime,
so later runs load in about a second.

All analysis helpers work on those matrices in chunks, so memory stays bounded
even for lengths with millions of solutions.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field

import numpy as np

CACHE_VERSION = 1
DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")

# A rulers array is a list of flat integer lists; capture everything between
# the outer brackets.
RULERS_RE = re.compile(rb'"rulers"\s*:\s*\[((?:\s*\[[\d,\s]*\]\s*,?)*)\s*\]')
NON_DIGITS = bytes(b if chr(b).isdigit() else ord(" ") for b in range(256))


def _secs(duration: dict | None) -> float:
    """A serialized Rust Duration in seconds."""
    if not duration:
        return 0.0
    return duration.get("secs", 0) + duration.get("nanos", 0) / 1e9


@dataclass
class LengthResult:
    """One length's metadata and rulers."""
    length: int
    num_segments: int
    lower_bound: int
    completed: bool
    rulers: np.ndarray  # (n, k) uint16
    evaluated: int = 0
    clock_time: float = 0.0
    cpu_time: float = 0.0

    @property
    def count(self) -> int:
        """Number of rulers."""
        return int(self.rulers.shape[0])

    @property
    def slack(self) -> int:
        """Arcs beyond what is needed: k(k-1) arcs cover L-1 distances."""
        k = self.num_segments
        return k * (k - 1) - (self.length - 1)


@dataclass
class Results:
    """Every length of one results file."""
    source: str
    version: str | None
    lengths: dict[int, LengthResult] = field(default_factory=dict)
    total_evaluated: int = 0

    def sorted(self) -> list[LengthResult]:
        """Lengths in order."""
        return [self.lengths[L] for L in sorted(self.lengths)]

    def with_rulers(self) -> list[LengthResult]:
        """Lengths in order that have rulers."""
        return [r for r in self.sorted() if r.count]


def _cache_path(path: str, cache_dir: str) -> str:
    """Cache file for `path`, keyed by its path, size, and mtime."""
    st = os.stat(path)
    key = f"{os.path.abspath(path)}|{st.st_size}|{int(st.st_mtime)}|{CACHE_VERSION}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:12]
    name = os.path.splitext(os.path.basename(path))[0]
    return os.path.join(cache_dir, f"{name}-{digest}.npz")


def _parse(path: str, with_rulers: bool) -> tuple[dict, list[np.ndarray | None]]:
    """The file's JSON without rulers, and each rulers array as flat uint16 values."""
    with open(path, "rb") as f:
        raw = f.read()
    blocks: list[bytes] = []

    def stash(match: re.Match) -> bytes:
        """Replace a rulers array with its index in `blocks`."""
        blocks.append(match.group(1))
        return b'"rulers": %d' % (len(blocks) - 1)

    meta = json.loads(RULERS_RE.sub(stash, raw))
    del raw
    arrays: list[np.ndarray | None] = []
    for block in blocks:
        if not with_rulers:
            arrays.append(None)
            continue
        text = block.translate(NON_DIGITS)
        values = np.array(text.split(), dtype=np.uint16) if text.strip() else np.zeros(0, np.uint16)
        arrays.append(values)
    return meta, arrays


def load_results(path: str, cache_dir: str | None = DEFAULT_CACHE_DIR,
                 with_rulers: bool = True, verbose: bool = True) -> Results:
    """Load a results JSON file, using (and filling) the npz cache when allowed."""
    cache = _cache_path(path, cache_dir) if cache_dir and with_rulers else None
    if cache and os.path.exists(cache):
        if verbose:
            print(f"Loading cached rulers from {cache}")
        with np.load(cache, allow_pickle=False) as npz:
            meta = json.loads(str(npz["meta"]))
            arrays = {int(k[1:]): npz[k] for k in npz.files if k.startswith("L")}
        return _build(path, meta, lambda L, idx: arrays.get(L))

    if verbose:
        print(f"Parsing {path} ({os.path.getsize(path) / 1e6:.0f} MB)...")
    meta, blocks = _parse(path, with_rulers)

    def rulers_for(L: int, idx) -> np.ndarray | None:
        """The flat rulers of block `idx`, if loaded."""
        if not with_rulers or not isinstance(idx, int):
            return None
        return blocks[idx]

    results = _build(path, meta, rulers_for)
    if cache:
        os.makedirs(cache_dir, exist_ok=True)
        slim = {k: v for k, v in meta.items() if k != "solutions"}
        slim["solutions"] = {
            L: {k: v for k, v in sol.items() if k != "rulers"}
            for L, sol in meta.get("solutions", {}).items()
        }
        payload = {f"L{r.length}": r.rulers for r in results.sorted()}
        tmp = cache + ".tmp.npz"
        np.savez(tmp, meta=np.array(json.dumps(slim)), **payload)
        os.replace(tmp, cache)
        if verbose:
            print(f"Cached parsed rulers to {cache}")
    return results


def _build(path: str, meta: dict, rulers_for) -> Results:
    """Results from parsed metadata; `rulers_for` gives each length's flat rulers."""
    results = Results(source=path, version=meta.get("version"),
                      total_evaluated=meta.get("total_rulers_evaluated", 0))
    for key, sol in meta.get("solutions", {}).items():
        L = int(key)
        k = sol.get("num_segments", 0)
        flat = rulers_for(L, sol.get("rulers"))
        if flat is None or k == 0:
            rulers = np.zeros((0, max(k, 1)), np.uint16)
        else:
            rulers = flat.reshape(-1, k) if flat.ndim == 1 else flat
        results.lengths[L] = LengthResult(
            length=L,
            num_segments=k,
            lower_bound=sol.get("lower_bound_num_segments", 0),
            completed=bool(sol.get("completed", True)),
            rulers=rulers,
            evaluated=sol.get("total_rulers_evaluated", 0),
            clock_time=_secs(sol.get("total_clock_time")),
            cpu_time=_secs(sol.get("total_cpu_time")),
        )
    return results


# ---------------------------------------------------------------------------
# Vectorized helpers
# ---------------------------------------------------------------------------

def chunks(n: int, size: int):
    """Slices of at most `size` covering range(n)."""
    for start in range(0, n, size):
        yield slice(start, min(start + size, n))


def sample_rows(rulers: np.ndarray, limit: int | None) -> np.ndarray:
    """Evenly spaced deterministic sample of at most `limit` rulers."""
    if not limit or rulers.shape[0] <= limit:
        return rulers
    idx = np.linspace(0, rulers.shape[0] - 1, limit).round().astype(np.int64)
    return rulers[idx]


def arc_sums(rulers: np.ndarray) -> np.ndarray:
    """Sums of every proper arc: result[r, i, m-1] = s_i + ... + s_{i+m-1} (cyclic).

    Shape (n, k, k-1); arcs of m = 1..k-1 consecutive segments starting at i.
    """
    n, k = rulers.shape
    doubled = np.concatenate([rulers, rulers], axis=1).astype(np.int32)
    prefix = np.zeros((n, 2 * k + 1), np.int32)
    np.cumsum(doubled, axis=1, out=prefix[:, 1:])
    starts = np.arange(k)[:, None]
    ends = starts + np.arange(1, k)[None, :]
    return prefix[:, ends] - prefix[:, starts]


def distance_multiplicity(rulers: np.ndarray, length: int, chunk: int = 50_000) -> np.ndarray:
    """Number of arcs measuring each distance; shape (n, length), column d = distance d."""
    n, k = rulers.shape
    out = np.zeros((n, length), np.int32)
    for sl in chunks(n, chunk):
        sums = arc_sums(rulers[sl]).reshape(sl.stop - sl.start, -1)
        rows = np.arange(sums.shape[0])[:, None]
        flat = (rows * length + sums).ravel()
        out[sl] = np.bincount(flat, minlength=sums.shape[0] * length).reshape(-1, length)
    return out


def canonical(rulers: np.ndarray) -> np.ndarray:
    """Lexicographically smallest rotation/reflection of each ruler, (n, k) uint16."""
    n, k = rulers.shape
    if n == 0:
        return rulers.copy()
    idx = np.array([[(s + j) % k for j in range(k)] for s in range(k)]
                   + [[(s - j) % k for j in range(k)] for s in range(k)])
    # The smallest variant starts with the row's smallest value, so only
    # starting positions holding that value need comparing.
    mins = rulers.min(axis=1)
    best = np.full_like(rulers, np.iinfo(rulers.dtype).max)
    for perm in idx:
        rows = np.nonzero(rulers[:, perm[0]] == mins)[0]
        if rows.size == 0:
            continue
        cand = rulers[rows][:, perm]
        cur = best[rows]
        diff = cand != cur
        first = diff.argmax(axis=1)
        r = np.arange(rows.size)
        less = diff[r, first] & (cand[r, first] < cur[r, first])
        best[rows[less]] = cand[less]
    return best


def row_keys(rulers: np.ndarray) -> np.ndarray:
    """Opaque hashable/sortable key per row (void bytes), equal iff rows are equal."""
    arr = np.ascontiguousarray(rulers.astype(np.uint16))
    return arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1]))).ravel()


def parents(rulers: np.ndarray) -> dict[str, np.ndarray]:
    """Candidate rulers one unit shorter that turn into `rulers` by one edit.

    "extend": the parent had one segment one shorter (same segment count).
    "split":  the parent had one segment s that became two segments a, b with
              a + b = s + 1 (one more segment). Reversed, a and b merge into a + b - 1.
    Returns (n, k, k_parent) arrays with rows of zeros where no parent exists.
    """
    n, k = rulers.shape
    ext = np.repeat(rulers[:, None, :], k, axis=1).astype(np.int32)
    ext[:, np.arange(k), np.arange(k)] -= 1
    ext[(ext == 0).any(axis=2)] = 0
    merged = np.zeros((n, k, k - 1), np.int32)
    for i in range(k):
        j = (i + 1) % k
        rest = [(i + 2 + t) % k for t in range(k - 2)]
        merged[:, i, 0] = rulers[:, i].astype(np.int32) + rulers[:, j] - 1
        merged[:, i, 1:] = rulers[:, rest]
    return {"extend": ext.astype(np.uint16), "split": merged.astype(np.uint16)}


def children(rulers: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Rulers one unit longer reachable by one edit (the inverse of `parents`).

    Returns, per relation, (candidates, source_row) with candidates of shape
    (m, k) for "extend" and (m, k + 1) for "split".
    """
    n, k = rulers.shape
    ext = np.repeat(rulers[:, None, :], k, axis=1).astype(np.int32)
    ext[:, np.arange(k), np.arange(k)] += 1
    ext_src = np.repeat(np.arange(n), k)
    split_rows, split_src = [], []
    vmax = int(rulers.max()) if n else 0
    for i in range(k):
        rest = rulers[:, [(i + 1 + t) % k for t in range(k - 1)]].astype(np.int32)
        s = rulers[:, i].astype(np.int32)
        for a in range(1, vmax + 1):
            ok = a <= s
            if not ok.any():
                break
            cand = np.concatenate([np.full((ok.sum(), 1), a), (s[ok] + 1 - a)[:, None], rest[ok]], axis=1)
            split_rows.append(cand)
            split_src.append(np.nonzero(ok)[0])
    split = np.concatenate(split_rows) if split_rows else np.zeros((0, k + 1), np.int32)
    src = np.concatenate(split_src) if split_src else np.zeros(0, np.int64)
    return {"extend": (ext.reshape(-1, k).astype(np.uint16), ext_src),
            "split": (split.astype(np.uint16), src)}


def row_hash(rulers: np.ndarray) -> np.ndarray:
    """64-bit polynomial hash per row (collisions are resolved by `_match`)."""
    h = np.zeros(rulers.shape[0], np.uint64)
    mult = np.uint64(0x9E3779B97F4A7C15)
    for j in range(rulers.shape[1]):
        h = h * mult + rulers[:, j].astype(np.uint64)
    return h


def _index(rulers: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sorted hashes of canonical forms, for `_match`."""
    canon = canonical(rulers)
    h = row_hash(canon)
    order = np.argsort(h)
    return h[order], order, canon


def _match(cand: np.ndarray, index) -> np.ndarray:
    """Index of each candidate ruler in an `_index`ed set, or -1."""
    hashes, order, canon = index
    canon_cand = canonical(cand)
    h = row_hash(canon_cand)
    pos = np.minimum(np.searchsorted(hashes, h), hashes.size - 1)
    hits = order[pos]
    found = (hashes[pos] == h) & (canon[hits] == canon_cand).all(axis=1)
    return np.where(found, hits, -1)


def link_lengths(prev: np.ndarray, cur: np.ndarray) -> dict[str, np.ndarray]:
    """Edges between solutions at L-1 (`prev`) and L (`cur`).

    Searches from whichever side generates fewer candidates: parents of every
    solution at L (k per ruler), or children of every solution at L-1 (about
    k + L per ruler). Returns, per relation, an (m, 2) array of
    (prev_index, cur_index) pairs.
    """
    empty = np.zeros((0, 2), np.int64)
    out = {"extend": empty, "split": empty}
    if prev.shape[0] == 0 or cur.shape[0] == 0:
        return out
    kp, kc = prev.shape[1], cur.shape[1]
    wanted = {"extend": kp == kc, "split": kp + 1 == kc}
    if not any(wanted.values()):
        return out
    L_prev = int(prev[0].sum())
    backward = cur.shape[0] * kc <= prev.shape[0] * (kp + L_prev)
    if backward:
        index = _index(prev)
        found = {name: [] for name in out}
        for sl in chunks(cur.shape[0], 100_000):
            for name, cand in parents(cur[sl]).items():
                if not wanted[name]:
                    continue
                n, k, _ = cand.shape
                flat = cand.reshape(n * k, -1)
                valid = flat.all(axis=1)
                hits = np.full(flat.shape[0], -1, np.int64)
                if valid.any():
                    hits[valid] = _match(flat[valid], index)
                cur_idx = sl.start + np.repeat(np.arange(n), k)
                mask = hits >= 0
                found[name].append(np.stack([hits[mask], cur_idx[mask]], axis=1))
        for name, parts in found.items():
            if parts:
                out[name] = np.unique(np.concatenate(parts), axis=0)
    else:
        index = _index(cur)
        for name, (cand, src) in children(prev).items():
            if not wanted[name] or cand.shape[0] == 0:
                continue
            hits = _match(cand, index)
            mask = hits >= 0
            out[name] = np.unique(np.stack([src[mask], hits[mask]], axis=1), axis=0)
    return out


def arc_index(k: int) -> tuple[np.ndarray, np.ndarray]:
    """Membership of segments in arcs, matching arc_sums' (i, m-1) flattening.

    Returns (seg_in_arc, boundary_inside_arc) boolean matrices of shapes
    (k*(k-1), k) and (k*(k-1), k). Boundary j sits between segment j-1 and j.
    """
    arcs = k * (k - 1)
    seg = np.zeros((arcs, k), bool)
    bnd = np.zeros((arcs, k), bool)
    for i in range(k):
        for m in range(1, k):
            a = i * (k - 1) + (m - 1)
            for t in range(m):
                seg[a, (i + t) % k] = True
            for t in range(1, m):
                bnd[a, (i + t) % k] = True
    return seg, bnd


def usage(rulers: np.ndarray, length: int, chunk: int = 256) -> dict[str, np.ndarray]:
    """Per-ruler segment involvement and boundary crossing counts.

    For each distance d, look at every arc measuring d:
      - segment i is *involved* in d (d <= L/2) if some arc for d contains it,
        and *essential* to d if every arc for d contains it;
      - boundary j is *crossed* for d if some arc for d has j strictly inside,
        and *forced* if every arc for d does (d cannot be measured without
        crossing j).
    Distances d and L-d are complementary arcs, so segment involvement only
    looks at d <= L/2 (otherwise every segment would be involved in every pair).
    Boundary counts use all d in 1..L-1.

    Returns arrays of shape (n, k): involved, essential, crossed, forced.
    """
    n, k = rulers.shape
    half = length // 2
    res = {name: np.zeros((n, k), np.int32) for name in ("involved", "essential", "crossed", "forced")}
    # (arc, j) pairs where segment / boundary j belongs to the arc
    members = [np.nonzero(m) for m in arc_index(k)]
    limits = (half, length - 1)
    names = (("involved", "essential"), ("crossed", "forced"))
    for sl in chunks(n, chunk):
        sums = arc_sums(rulers[sl]).reshape(sl.stop - sl.start, -1)  # (c, A)
        c, A = sums.shape
        flat_d = (np.arange(c)[:, None] * length + sums).ravel()
        count_d = np.bincount(flat_d, minlength=c * length).reshape(c, length)
        for (arc, j), limit, (name_any, name_all) in zip(members, limits, names):
            # per (ruler, distance, j): number of arcs for d that contain j
            idx = (flat_d.reshape(c, A)[:, arc] * k + j).ravel()
            hit = np.bincount(idx, minlength=c * length * k).reshape(c, length, k)[:, 1:limit + 1]
            cnt = count_d[:, 1:limit + 1, None]
            present = cnt > 0
            res[name_any][sl] = (hit > 0).sum(axis=1)
            res[name_all][sl] = ((hit == cnt) & present).sum(axis=1)
    return res
