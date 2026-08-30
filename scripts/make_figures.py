#!/usr/bin/env python3
"""Figures for "The Token Cost of Forced Externalization" (SPEC.md).

Figure 1: B* (budget to reach 90% of bin base rate) vs bin median difficulty,
  one line per ablation level, random-control overlay, base budget-sweep line,
  y = x reference (base model's own usage), censored bins as upward arrows at
  the plot top, bootstrap 68% CI bands, inclusion-rate strip underneath.
Figure 2: asymptotic retention (B = 16384) vs difficulty per level + controls,
  same strip.

Works on real results or the synthetic fixture:
  python3 scripts/make_figures.py                                   # real
  python3 scripts/make_figures.py --results-dir <fixture> \
      --out-dir <fixture>/figures --n-boot 200                      # fixture

Estimates come from src/extcot/analysis.py (cell_bstar) unmodified; this
script only loads runs, builds per-problem majority outcomes, and plots.
Missing cells are warned about and skipped, so partial grids still render.
Also writes <out-dir>/bstar_summary.json with every plotted number.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

from extcot import analysis

B_MAX = 16384
BUDGETS = [0, 128, 512, 2048, 16384]

# Colorblind-safe palette (dataviz reference instance): ordered ablation
# strength = one blue ramp light->dark (ordinal); random control = orange
# (slot 2); out-of-band control = aqua (slot 3); base/reference = neutral
# inks. Marker shape + line style carry identity as the secondary channel.
STYLE = {
    "light":  dict(color="#86b6ef", marker="o", ls="-",  label="light (26:30, k=10)"),
    "medium": dict(color="#2a78d6", marker="s", ls="-",  label="medium (22:34, k=10)"),
    "heavy":  dict(color="#104281", marker="^", ls="-",  label="heavy (22:34, k=100)"),
    "random": dict(color="#eb6834", marker="x", ls="--", label="random ctrl (2 seeds)"),
    "base":   dict(color="#52514e", marker="D", ls="-.", label="base (truncated CoT)"),
    "oob":    dict(color="#1baf7a", marker="*", ls=":",  label="out-of-band ctrl"),
}
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"

LEVEL_TAGS = [("light", "jspace-light"), ("medium", "jspace-med"),
              ("heavy", "jspace-heavy")]


def load_runs(results_dir: str, tag: str) -> list[dict] | None:
    path = os.path.join(results_dir, "runs", f"{tag}.jsonl")
    if not os.path.exists(path):
        print(f"  WARN: missing cell {tag} ({path}), skipping")
        return None
    rows = []
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # torn tail line of a live run
    return rows


def majority_checked(rows: list[dict], tag: str) -> dict[str, bool]:
    """analysis.majority (>=2 solved), plus a loud warning on partial cells:
    with <4 samples the fixed >=2 rule biases toward 'unsolved'."""
    n_by_pid = defaultdict(int)
    for r in rows:
        n_by_pid[r["pid"]] += 1
    partial = {p: n for p, n in n_by_pid.items() if n < 4}
    if partial:
        print(f"  WARN [{tag}]: {len(partial)} pids have <4 samples "
              f"(majority '>=2' biases these toward unsolved), e.g. "
              f"{list(partial.items())[:3]}")
    return analysis.majority(rows)


def collect_outcomes(results_dir: str, prefix: str,
                     budgets=BUDGETS) -> dict[int, dict[str, bool]]:
    out = {}
    for B in budgets:
        rows = load_runs(results_dir, f"{prefix}-B{B}")
        if rows:
            out[B] = majority_checked(rows, f"{prefix}-B{B}")
    return out


def base_outcomes(results_dir: str, design: dict) -> dict[int, dict[str, bool]]:
    """Base budget sweep (amendment 5): base-B{0,128,512,2048} runs, plus the
    clean Pile-B majority (design base_solved) as the B=16384 point."""
    out = collect_outcomes(results_dir, "base", [0, 128, 512, 2048])
    out[B_MAX] = {pid: p["base_solved"] for pid, p in design["problems"].items()
                  if p.get("in_sample", p["included"])}
    return out


def retention_ci(problems, outcome_bmax, pids, n_boot=500, seed=1):
    """Bootstrap-over-problems 68% CI for retention at B_MAX."""
    rng = random.Random(seed)
    both = [pid for pid in pids if pid in outcome_bmax]
    if not both:
        return None, None
    vals = []
    for _ in range(n_boot):
        bs = [both[rng.randrange(len(both))] for _ in both]
        den = np.mean([problems[pid]["base_solved"] for pid in bs])
        if den > 0:
            vals.append(np.mean([outcome_bmax[pid] for pid in bs]) / den)
    return (float(np.percentile(vals, 16)), float(np.percentile(vals, 84))) \
        if vals else (None, None)


def inclusion_strip(ax, design, edges, xlim):
    rates, ns = [], []
    for b in range(len(edges) - 1):
        lo, hi = edges[b], edges[b + 1]
        allp = [p for p in design["problems"].values()
                if lo <= p["difficulty"] < hi]
        inc = [p for p in allp if p["included"]]
        rates.append(len(inc) / len(allp) if allp else 0.0)
        ns.append(sum(p.get("in_sample", p["included"]) for p in allp))
    for b, r in enumerate(rates):
        lo = max(edges[b], xlim[0])
        hi = min(edges[b + 1], xlim[1])
        ax.bar(lo, r, width=hi - lo, align="edge", color="#c3c2b7",
               edgecolor="white", linewidth=1.5, alpha=0.75)
        cx = math.sqrt(lo * hi)
        ax.text(cx, min(r + 0.06, 1.02), f"{r:.0%}", ha="center", va="bottom",
                fontsize=6.5, color="#52514e")
        ax.text(cx, 0.09, f"n={ns[b]}", ha="center", va="bottom",
                fontsize=6, color="#52514e")
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0", ".5", "1"], fontsize=7)
    ax.set_ylabel("inclusion", fontsize=7.5)
    ax.set_xlabel("bin difficulty: median base thinking tokens (log)", fontsize=8.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def style_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.tick_params(labelsize=8, colors=INK)


def draw_censor_arrows(ax, xs, color, jitter=1.0):
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    for x in xs:
        ax.annotate("", xy=(x * jitter, 0.99), xytext=(x * jitter, 0.90),
                    xycoords=tr, textcoords=tr,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6),
                    annotation_clip=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="/workspace/jlens-cot/results")
    ap.add_argument("--out-dir", default=None,
                    help="default: <results-dir>/../figures")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rd = args.results_dir.rstrip("/")
    out_dir = args.out_dir or os.path.join(os.path.dirname(rd), "figures")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(rd, "design.json")) as f:
        design = json.load(f)
    problems = design["problems"]
    edges = design["bin_edges"]

    # ------------------------------------------------------------ estimates
    series = {}
    for name, prefix in LEVEL_TAGS:
        oc = collect_outcomes(rd, prefix)
        if oc:
            series[name] = analysis.cell_bstar(problems, oc, edges,
                                               n_boot=args.n_boot,
                                               seed=args.seed)
            series[f"_outcomes_{name}"] = oc
    rand = {}
    for s in (0, 1):
        oc = collect_outcomes(rd, f"random-s{s}")
        if oc:
            rand[s] = analysis.cell_bstar(problems, oc, edges,
                                          n_boot=args.n_boot, seed=args.seed)
            series[f"_outcomes_random-s{s}"] = oc
    oc_base = base_outcomes(rd, design)
    base_res = analysis.cell_bstar(problems, oc_base, edges,
                                   n_boot=args.n_boot, seed=args.seed) \
        if len(oc_base) > 1 else {}
    oob_rows = load_runs(rd, "oob-B16384")
    oob_out = majority_checked(oob_rows, "oob-B16384") if oob_rows else None

    if not series and not base_res:
        sys.exit("No cells found under " + rd)

    # x range from bin medians
    any_res = next(iter([*series.values(), base_res]))
    meds = {b: v["median_difficulty"] for res in [base_res, *
            [series[n] for n, _ in LEVEL_TAGS if n in series]]
            for b, v in (res or {}).items()}
    xlim = (min(meds.values()) / 2.2, max(meds.values()) * 2.6)

    # ------------------------------------------------------------- figure 1
    fig = plt.figure(figsize=(7.0, 5.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[4.1, 1.0], hspace=0.13)
    ax = fig.add_subplot(gs[0])
    axs = fig.add_subplot(gs[1], sharex=ax)

    ax.set_xscale("log")
    ax.set_yscale("log")
    y_floor = 30
    ax.set_xlim(*xlim)
    ax.set_ylim(y_floor, B_MAX * 1.9)

    xs_ref = np.geomspace(*xlim, 50)
    ax.plot(xs_ref, xs_ref, ls=":", color=MUTED, lw=1.2, zorder=1)
    ax.text(xlim[1] * 0.55, xlim[1] * 0.42, "y = x (base usage)", fontsize=7.5,
            color=MUTED, rotation=32, ha="center", va="top",
            rotation_mode="anchor")
    ax.axhline(B_MAX, color=GRID, lw=0.8)
    ax.text(xlim[1] * 0.93, B_MAX * 1.16, "budget cap", fontsize=6.5,
            color=MUTED, ha="right", va="bottom")

    def plot_bstar(res, st, jitter=1.0, band=True, lw=1.7, alpha=1.0,
                   label=None):
        bs = sorted(res)
        x = np.array([res[b]["median_difficulty"] for b in bs])
        y = np.array([res[b]["bstar"] if not res[b]["censored"] else np.nan
                      for b in bs], dtype=float)
        y = np.clip(y, y_floor, None)
        lo = np.array([res[b]["lo"] if res[b]["lo"] is not None else np.nan
                       for b in bs], dtype=float).clip(y_floor, B_MAX * 1.9)
        hi = np.array([res[b]["hi"] if res[b]["hi"] is not None else np.nan
                       for b in bs], dtype=float).clip(y_floor, B_MAX * 1.9)
        ok = ~np.isnan(y)
        if band and ok.any():
            ax.fill_between(x[ok], lo[ok], hi[ok], color=st["color"],
                            alpha=0.13, lw=0, zorder=2)
        ax.plot(x[ok], y[ok], marker=st["marker"], ls=st["ls"], lw=lw,
                ms=5, color=st["color"], alpha=alpha, zorder=3,
                label=label, markerfacecolor="none" if st["marker"] in "o^sD"
                else st["color"], markeredgewidth=1.4)
        cens_x = [res[b]["median_difficulty"] for b in bs if res[b]["censored"]]
        draw_censor_arrows(ax, cens_x, st["color"], jitter)

    for i, (name, _) in enumerate(LEVEL_TAGS):
        if name in series:
            plot_bstar(series[name], STYLE[name], jitter=1.09 ** (i - 1),
                       label=STYLE[name]["label"])
    for j, (s, res) in enumerate(sorted(rand.items())):
        plot_bstar(res, STYLE["random"], jitter=1.09 ** (j + 2), band=False,
                   lw=1.1, alpha=0.85,
                   label=STYLE["random"]["label"] if j == 0 else None)
    if base_res:
        plot_bstar(base_res, STYLE["base"], jitter=1.09 ** -2, band=True,
                   lw=1.3, label=STYLE["base"]["label"])

    ax.set_ylabel("B*: thinking-token budget for 90% of base rate (log)",
                  fontsize=8.5)
    ax.set_title("External budget needed to recover base performance rises "
                 "with difficulty under J-space ablation", fontsize=9.5,
                 color=INK, loc="left", pad=10)
    ax.text(0.995, 1.015, "$\\uparrow$ = censored: fit never reaches 90% "
            "within the cap", transform=ax.transAxes, ha="right", fontsize=7,
            color=MUTED)
    leg = ax.legend(fontsize=7.5, frameon=False, loc="upper left",
                    handlelength=2.6, borderaxespad=0.2)
    for t in leg.get_texts():
        t.set_color(INK)
    style_axes(ax)
    plt.setp(ax.get_xticklabels(), visible=False)

    inclusion_strip(axs, design, edges, xlim)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig1_bstar.{ext}"), dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ------------------------------------------------------------- figure 2
    fig = plt.figure(figsize=(7.0, 4.9))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.4, 1.0], hspace=0.15)
    ax = fig.add_subplot(gs[0])
    axs = fig.add_subplot(gs[1], sharex=ax)
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(0, 1.18)
    ax.axhline(1.0, color=GRID, lw=0.9)
    ax.axhline(0.9, color=MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.text(xlim[0] * 1.1, 0.905, "recovery threshold (0.9)", fontsize=7,
            color=MUTED, va="bottom")

    def plot_ret(res, oc, st, lw=1.7, alpha=1.0, label=None):
        bs = [b for b in sorted(res) if res[b]["retention_at_bmax"] is not None]
        if not bs:
            return
        x = [res[b]["median_difficulty"] for b in bs]
        y = [res[b]["retention_at_bmax"] for b in bs]
        los, his = [], []
        for b in bs:
            pids = [pid for pid, p in problems.items()
                    if p.get("in_sample", p["included"])
                    and analysis.bin_of(p["difficulty"], edges) == b]
            lo, hi = retention_ci(problems, oc.get(B_MAX, {}), pids,
                                  n_boot=args.n_boot)
            los.append(lo if lo is not None else np.nan)
            his.append(hi if hi is not None else np.nan)
        ax.fill_between(x, los, his, color=st["color"], alpha=0.13, lw=0)
        ax.plot(x, y, marker=st["marker"], ls=st["ls"], lw=lw, ms=5,
                color=st["color"], alpha=alpha, label=label,
                markerfacecolor="none" if st["marker"] in "o^sD*"
                else st["color"], markeredgewidth=1.4)

    for name, _ in LEVEL_TAGS:
        if name in series:
            plot_ret(series[name], series[f"_outcomes_{name}"], STYLE[name],
                     label=STYLE[name]["label"])
    for j, (s, res) in enumerate(sorted(rand.items())):
        plot_ret(res, series[f"_outcomes_random-s{s}"], STYLE["random"],
                 lw=1.1, alpha=0.85,
                 label=STYLE["random"]["label"] if j == 0 else None)
    if oob_out:
        oob_res = analysis.cell_bstar(problems, {B_MAX: oob_out}, edges,
                                      n_boot=0)
        plot_ret(oob_res, {B_MAX: oob_out}, STYLE["oob"], lw=1.2,
                 label=STYLE["oob"]["label"])

    ax.set_ylabel("retention at B = 16384\n(ablated / base solve rate)",
                  fontsize=8.5)
    ax.set_title("Asymptotic retention falls with difficulty; frontier moves "
                 "down with ablation strength", fontsize=9.5, color=INK,
                 loc="left", pad=8)
    leg = ax.legend(fontsize=7.5, frameon=False, loc="lower left",
                    handlelength=2.6)
    for t in leg.get_texts():
        t.set_color(INK)
    style_axes(ax)
    plt.setp(ax.get_xticklabels(), visible=False)
    inclusion_strip(axs, design, edges, xlim)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig2_retention.{ext}"), dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ------------------------------------------------------------ summaries
    summary = {"bin_edges": edges}
    for name in [n for n, _ in LEVEL_TAGS if n in series]:
        summary[name] = series[name]
    for s, res in rand.items():
        summary[f"random-s{s}"] = res
    if base_res:
        summary["base"] = base_res
    with open(os.path.join(out_dir, "bstar_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)

    print(f"\nWrote figures + bstar_summary.json to {out_dir}\n")
    hdr = f"{'series':10s} " + " ".join(f"bin{b:>7d}" for b in
                                        sorted(next(iter(
                                            [v for k, v in summary.items()
                                             if k != 'bin_edges']))))
    print(hdr)
    for k, res in summary.items():
        if k == "bin_edges":
            continue
        cells = " ".join(
            f"{'CENS':>10s}" if v["censored"] else f"{v['bstar']:>10.0f}"
            for _, v in sorted(res.items(), key=lambda kv: int(kv[0])))
        print(f"{k:10s} {cells}")


if __name__ == "__main__":
    main()
