#!/usr/bin/env python3
"""Write each length's distinct canonical rulers from a results file to a smaller JSON file.

Usage: extract_solutions.py <results_json> [output_json]
"""
import json
import re
import sys


def canonical(ruler):
    """Lexicographically smallest rotation or reflection of `ruler`."""
    n, rev = len(ruler), ruler[::-1]
    variants = [ruler[i:] + ruler[:i] for i in range(n)] + [rev[i:] + rev[:i] for i in range(n)]
    return min(variants) if ruler else ruler


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <results_json> [output_json]")
        sys.exit(1)
    with open(sys.argv[1]) as f:
        data = json.load(f)

    output = {}
    solutions = data.get("solutions", {})
    for length in sorted(solutions, key=int):
        s = solutions[length]
        if not s.get("rulers"):
            continue
        rulers = sorted({tuple(canonical(r)) for r in s["rulers"]})
        output[length] = {
            "lower_bound_num_segments": s.get("lower_bound_num_segments"),
            "num_segments": s.get("num_segments"),
            "rulers_found": len(rulers),
            "rulers": [list(r) for r in rulers],
        }

    # Pretty-print, with each ruler on one line.
    text = json.dumps(output, indent=2)
    text = re.sub(r"\[[\s\n]+([\d,\s\n]+)\]",
                  lambda m: "[" + ", ".join(x.rstrip(",") for x in m.group(1).split()) + "]", text)
    out_file = sys.argv[2] if len(sys.argv) > 2 else "solutions.json"
    with open(out_file, "w") as f:
        f.write(text + "\n")
    print(f"Extracted solutions to {out_file}")


if __name__ == "__main__":
    main()
