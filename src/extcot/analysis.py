"""Analysis: retention per bin, logistic B* fits, bootstrap CIs.

Conventions:
- A problem counts as solved in a cell iff >= 2 of 4 samples solved (majority),
  matching the inclusion rule and the (fixed) base-rate estimator.
- B* per (bin, level): fit solved ~ sigmoid(a + b * log(B+1)) over the budget
  grid on per-problem majority outcomes, pooled within the bin; B* is where
  the fitted curve crosses 0.9 * bin base rate. Censored if the fit never
  crosses that threshold on [0, B_MAX].
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

RESULTS = "/workspace/jlens-cot/results"
B_MAX = 16384


def load_runs(tag: str) -> list[dict]:
    out = []
    with open(f"{RESULTS}/runs/{tag}.jsonl") as f:
        for line in f:
            out.append(json.loads(line))
    return out


def load_design() -> dict:
    with open(f"{RESULTS}/design.json") as f:
        return json.load(f)


def majority(records: list[dict]) -> dict[str, bool]:
    """pid -> majority-solved over its samples."""
    by_pid = defaultdict(list)
    for r in records:
        by_pid[r["pid"]].append(r["solved"])
    return {pid: sum(v) >= 2 for pid, v in by_pid.items()}


def bin_of(diff: float, edges: list[float]) -> int:
    for b in range(len(edges) - 1):
        if edges[b] <= diff < edges[b + 1]:
            return b
    return len(edges) - 2


def fit_logistic(logB: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """MLE for y ~ sigmoid(a + b*logB), b >= 0 (monotone in budget)."""

    def nll(theta):
        a, b = theta
        p = expit(a + b * logB).clip(1e-6, 1 - 1e-6)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p)).sum()

    best = None
    for a0 in (-2.0, 0.0, 2.0):
        for b0 in (0.1, 0.5, 2.0):
            r = minimize(nll, [a0, b0], method="L-BFGS-B",
                         bounds=[(-20, 20), (0, 10)])
            if best is None or r.fun < best.fun:
                best = r
    return float(best.x[0]), float(best.x[1])


def bstar_from_fit(a: float, b: float, target: float) -> float | None:
    """Smallest B in [0, B_MAX] with sigmoid(a + b*log(B+1)) >= target.
    None = censored (never crosses)."""
    if expit(a + b * math.log(B_MAX + 1)) < target:
        return None
    if expit(a) >= target:
        return 0.0
    x = (math.log(target / (1 - target)) - a) / b
    return math.exp(x) - 1


def cell_bstar(problems: dict[str, dict], outcomes: dict[int, dict[str, bool]],
               edges: list[float], n_boot: int = 500, seed: int = 0):
    """outcomes: budget -> pid -> majority-solved.
    Returns per-bin dicts: {bstar, lo, hi, censored, base_rate, n,
    retention_at_bmax}."""
    budgets = sorted(outcomes)
    bins = defaultdict(list)   # bin -> pids
    for pid, p in problems.items():
        if p.get("in_sample", p["included"]):
            bins[bin_of(p["difficulty"], edges)].append(pid)

    results = {}
    rng = random.Random(seed)
    for b, pids in sorted(bins.items()):
        base = np.mean([problems[pid]["base_solved"] for pid in pids])
        target = 0.9 * base

        def collect(sample_pids):
            logB, y = [], []
            for pid in sample_pids:
                for B in budgets:
                    if pid in outcomes[B]:
                        logB.append(math.log(B + 1))
                        y.append(outcomes[B][pid])
            return np.array(logB), np.array(y, dtype=float)

        logB, y = collect(pids)
        if len(y) == 0 or base == 0:
            continue
        a_, b_ = fit_logistic(logB, y)
        bstar = bstar_from_fit(a_, b_, target)
        # retention at B_MAX (empirical)
        top = outcomes.get(B_MAX, {})
        at_max = [top[pid] for pid in pids if pid in top]
        ret_max = (np.mean(at_max) / base) if (at_max and base > 0) else None

        boots = []
        for _ in range(n_boot):
            bs = [pids[rng.randrange(len(pids))] for _ in pids]
            base_bs = np.mean([problems[pid]["base_solved"] for pid in bs])
            if base_bs == 0:
                continue
            lB, yy = collect(bs)
            try:
                aa, bb = fit_logistic(lB, yy)
            except Exception:
                continue
            v = bstar_from_fit(aa, bb, 0.9 * base_bs)
            boots.append(B_MAX * 4 if v is None else v)   # censored -> large
        lo = float(np.percentile(boots, 16)) if boots else None
        hi = float(np.percentile(boots, 84)) if boots else None
        results[b] = dict(
            bstar=bstar, censored=bstar is None, lo=lo, hi=hi,
            base_rate=float(base), n=len(pids),
            retention_at_bmax=None if ret_max is None else float(ret_max),
            median_difficulty=float(np.median([problems[pid]["difficulty"]
                                               for pid in pids])),
        )
    return results
