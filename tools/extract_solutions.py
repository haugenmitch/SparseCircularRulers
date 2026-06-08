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

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <results_json> [output_json]")
        sys.exit(1)

    with open(sys.argv[1], 'r') as f:
        data = json.load(f)

    output = {}
    solutions = data.get("solutions", {})
    sorted_keys = sorted(solutions.keys(), key=lambda x: int(x))

    for k in sorted_keys:
        s = solutions[k]
        if not s.get("rulers"):
            continue
            
        canonical_rulers = []
        seen = set()
        for r in s["rulers"]:
            canon = tuple(get_canonical(r))
            if canon not in seen:
                canonical_rulers.append(list(canon))
                seen.add(canon)
        
        canonical_rulers.sort()
        
        output[k] = {
            "lower_bound_num_segments": s.get("lower_bound_num_segments"),
            "num_segments": s.get("num_segments"),
            "rulers_found": len(canonical_rulers),
            "rulers": canonical_rulers
        }

    out_file = sys.argv[2] if len(sys.argv) > 2 else "solutions.json"
    
    # Pretty print with indent=2, then collapse ruler lists to single lines
    import re
    json_str = json.dumps(output, indent=2)
    compact_json = re.sub(r'\[[\s\n]+([\d,\s\n]+)\]', 
                          lambda m: "[" + ", ".join(x.rstrip(',') for x in m.group(1).split()) + "]", 
                          json_str)

    with open(out_file, 'w') as f:
        f.write(compact_json)
        f.write('\n')
    print(f"Extracted solutions to {out_file}")

if __name__ == "__main__":
    main()
