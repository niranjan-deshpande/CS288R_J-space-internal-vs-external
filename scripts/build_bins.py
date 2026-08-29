"""Freeze the experimental design from clean runs: solvable set, difficulty,
bins, base rates. Run ONCE after the clean run completes, before any ablated
result is analyzed. Writes results/design.json.

Definitions (spec + approved changes):
- Pile A = samples 0-3, Pile B = samples 4-7.
- Inclusion: Pile A solves >= 2 of 4. Final.
- Difficulty: median think_tokens over Pile B *successful* attempts;
  fallback (documented): median over all Pile B attempts if B has no success.
- Base solved per problem: Pile B majority (>= 2 of 4)  [metric fix].
- Bins: log-spaced edges chosen from the clean difficulty distribution,
  floor 25 included problems per bin, adjacent bins merged below floor.
"""
import json
import statistics
from collections import defaultdict

RESULTS = "/workspace/jlens-cot/results"


def main():
    rows = defaultdict(dict)   # pid -> sample_idx -> record
    with open(f"{RESULTS}/runs/clean.jsonl") as f:
        for line in f:
            r = json.loads(line)
            rows[r["pid"]][r["sample_idx"]] = r

    design = {"problems": {}, "solvable_ids": []}
    for pid, samples in sorted(rows.items()):
        if len(samples) < 8:
            print(f"WARN {pid}: only {len(samples)} samples")
            continue
        pile_a = [samples[i] for i in range(4)]
        pile_b = [samples[i] for i in range(4, 8)]
        a_solved = sum(r["solved"] for r in pile_a)
        b_solved = sum(r["solved"] for r in pile_b)
        included = a_solved >= 2
        b_succ_toks = [r["think_tokens"] for r in pile_b if r["solved"]]
        b_all_toks = [r["think_tokens"] for r in pile_b]
        difficulty = (statistics.median(b_succ_toks) if b_succ_toks
                      else statistics.median(b_all_toks))
        design["problems"][pid] = {
            "dataset": pile_a[0]["dataset"],
            "included": included,
            "a_solved": a_solved,
            "b_solved": b_solved,
            "base_solved": b_solved >= 2,
            "difficulty": difficulty,
            "difficulty_fallback": not b_succ_toks,
            "difficulty_all_b": statistics.median(b_all_toks),
        }
        if included:
            design["solvable_ids"].append(pid)

    inc = [p for p in design["problems"].values() if p["included"]]
    diffs = sorted(p["difficulty"] for p in inc)
    print(f"{len(inc)}/{len(design['problems'])} included")
    print("difficulty percentiles:",
          {q: diffs[int(q * (len(diffs) - 1))] for q in (0, .1, .25, .5, .75, .9, 1.)})

    # candidate log-spaced edges; merge below floor
    edges = [0, 250, 500, 1000, 2000, 4000, 8000, float("inf")]
    def bin_counts(es):
        counts = [0] * (len(es) - 1)
        for p in inc:
            for b in range(len(es) - 1):
                if es[b] <= p["difficulty"] < es[b + 1]:
                    counts[b] += 1
        return counts
    counts = bin_counts(edges)
    print("raw bins:", list(zip(edges[:-1], counts)))
    while min(counts) < 25 and len(edges) > 2:
        i = counts.index(min(counts))
        # merge with the smaller neighbor
        if i == 0:
            j = 1
        elif i == len(counts) - 1:
            j = i
        else:
            j = i if counts[i - 1] <= counts[i + 1] else i + 1
        del edges[j]
        counts = bin_counts(edges)
    print("final bins:", list(zip(edges[:-1], counts)))
    design["bin_edges"] = [e if e != float("inf") else 1e9 for e in edges]

    # per-bin summary
    for b in range(len(edges) - 1):
        lo, hi = edges[b], edges[b + 1]
        members = [(pid, p) for pid, p in design["problems"].items()
                   if p["included"] and lo <= p["difficulty"] < hi]
        base = sum(p["base_solved"] for _, p in members)
        total_in_range = sum(1 for p in design["problems"].values()
                             if lo <= p.get("difficulty", -1) < hi)
        print(f"bin [{lo}, {hi}): n={len(members)} base_rate={base}/{len(members)}"
              f" inclusion={len(members)}/{total_in_range}")

    with open(f"{RESULTS}/design.json", "w") as f:
        json.dump(design, f, indent=1)
    print("wrote design.json")


if __name__ == "__main__":
    main()
