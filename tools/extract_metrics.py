#!/usr/bin/env python3
"""Write a results file's per-length metrics (counts, times, throughput) to a JSON file.

Usage: extract_metrics.py <results_json> [output_json]
"""
import json
import sys


def canonical(ruler):
    """Lexicographically smallest rotation or reflection of `ruler`."""
    n, rev = len(ruler), ruler[::-1]
    variants = [ruler[i:] + ruler[:i] for i in range(n)] + [rev[i:] + rev[:i] for i in range(n)]
    return min(variants) if ruler else ruler


def seconds(d):
    """A serialized Rust Duration in seconds."""
    return d.get("secs", 0) + d.get("nanos", 0) / 1e9


def format_duration(d):
    """A serialized Rust Duration as readable text."""
    total = seconds(d)
    if total < 0.001:
        return f"{total * 1e6:.2f}us"
    if total < 1:
        return f"{total * 1e3:.2f}ms"
    if total < 60:
        return f"{total:.2f}s"
    return f"{int(total // 60)}m {total % 60:.2f}s"


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <results_json> [output_json]")
        sys.exit(1)
    with open(sys.argv[1]) as f:
        data = json.load(f)

    metrics = {
        "version": data.get("version"),
        "lengths_solved": data.get("lengths_solved"),
        "total_rulers_found": data.get("total_rulers_found"),
        "total_rulers_evaluated": data.get("total_rulers_evaluated"),
        "total_clock_time": format_duration(data.get("total_clock_time", {})),
        "total_cpu_time": format_duration(data.get("total_cpu_time", {})),
        "solutions": {},
    }
    solutions = data.get("solutions", {})
    for length in sorted(solutions, key=int):
        s = solutions[length]
        m = metrics["solutions"][length] = {
            "completed": s.get("completed"),
            "num_segments": s.get("num_segments"),
            "rulers_found": s.get("rulers_found"),
            # Distinct rulers up to rotation and reflection.
            "true_rulers_found": len({tuple(canonical(r)) for r in s.get("rulers") or []}),
            "total_rulers_evaluated": s.get("total_rulers_evaluated"),
            "total_clock_time": format_duration(s.get("total_clock_time", {})),
            "total_cpu_time": format_duration(s.get("total_cpu_time", {})),
        }
        clock = seconds(s.get("total_clock_time", {}))
        if clock > 0:
            m["evals_per_sec"] = f"{s.get('total_rulers_evaluated', 0) / clock:.2f}"

    out_file = sys.argv[2] if len(sys.argv) > 2 else "metrics.json"
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Extracted metrics to {out_file}")


if __name__ == "__main__":
    main()
