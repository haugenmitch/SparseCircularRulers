#!/usr/bin/env python3
import json
import sys
import os

def get_canonical(ruler):
    n = len(ruler)
    if n == 0: return ruler
    
    all_perms = []
    # Rotations
    for i in range(n):
        all_perms.append(ruler[i:] + ruler[:i])
    # Reflections (rotations of reversed)
    rev_ruler = ruler[::-1]
    for i in range(n):
        all_perms.append(rev_ruler[i:] + rev_ruler[:i])
        
    # Rule 1: Longest string of 1s at the beginning
    max_ones = 0
    for p in all_perms:
        ones = 0
        for x in p:
            if x == 1: ones += 1
            else: break
        if ones > max_ones:
            max_ones = ones
            
    candidates = []
    for p in all_perms:
        ones = 0
        for x in p:
            if x == 1: ones += 1
            else: break
        if ones == max_ones:
            candidates.append(p)
            
    # Rule 2: Tie-breaker for prefix palindrome (which strings of 1s are)
    # suffix < suffix[::-1]
    valid_candidates = []
    for c in candidates:
        suffix = c[max_ones:]
        if suffix <= suffix[::-1]:
            valid_candidates.append(c)
            
    if not valid_candidates:
        # Should not happen if Rule 2 is just a filter
        valid_candidates = candidates
        
    valid_candidates.sort()
    return valid_candidates[0]

def format_duration(d):
    secs = d.get("secs", 0)
    nanos = d.get("nanos", 0)
    total = secs + nanos / 1e9
    if total < 0.001:
        return f"{total*1e6:.2f}us"
    if total < 1:
        return f"{total*1e3:.2f}ms"
    if total < 60:
        return f"{total:.2f}s"
    mins = int(total // 60)
    rem_secs = total % 60
    return f"{mins}m {rem_secs:.2f}s"

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <results_json> [output_json]")
        sys.exit(1)

    with open(sys.argv[1], 'r') as f:
        data = json.load(f)

    metrics = {
        "version": data.get("version"),
        "lengths_solved": data.get("lengths_solved"),
        "total_rulers_found": data.get("total_rulers_found"),
        "total_rulers_evaluated": data.get("total_rulers_evaluated"),
        "total_clock_time": format_duration(data.get("total_clock_time", {})),
        "total_cpu_time": format_duration(data.get("total_cpu_time", {})),
        "solutions": {}
    }

    # Sort solutions by length
    solutions = data.get("solutions", {})
    sorted_keys = sorted(solutions.keys(), key=lambda x: int(x))

    for k in sorted_keys:
        s = solutions[k]
        
        # Calculate true (canonical) solution count
        true_rulers_found = 0
        if s.get("rulers"):
            seen = set()
            for r in s["rulers"]:
                canon = tuple(get_canonical(r))
                if canon not in seen:
                    true_rulers_found += 1
                    seen.add(canon)

        metrics["solutions"][k] = {
            "completed": s.get("completed"),
            "num_segments": s.get("num_segments"),
            "rulers_found": s.get("rulers_found"),
            "true_rulers_found": true_rulers_found,
            "total_rulers_evaluated": s.get("total_rulers_evaluated"),
            "total_clock_time": format_duration(s.get("total_clock_time", {})),
            "total_cpu_time": format_duration(s.get("total_cpu_time", {})),
        }
        
        # Calculate evals/s
        secs = s.get("total_clock_time", {}).get("secs", 0)
        nanos = s.get("total_clock_time", {}).get("nanos", 0)
        total_secs = secs + nanos / 1e9
        if total_secs > 0:
            evals_per_sec = s.get("total_rulers_evaluated", 0) / total_secs
            metrics["solutions"][k]["evals_per_sec"] = f"{evals_per_sec:.2f}"

    out_file = sys.argv[2] if len(sys.argv) > 2 else "metrics.json"
    with open(out_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Extracted metrics to {out_file}")

if __name__ == "__main__":
    main()
