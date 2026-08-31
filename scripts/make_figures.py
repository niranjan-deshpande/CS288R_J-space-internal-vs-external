#!/usr/bin/env python3
"""Figures for "The Token Cost of Forced Externalization" (SPEC.md).

Figure 1: B* (budget to reach 90% of bin base rate) vs bin median difficulty,
  one line per ablation level, random-control overlay, base budget-sweep line,
  y = x reference (base model's own usage), censored bins as upward arrows at
  the plot top, bootstrap 68% CI bands, inclusion-rate strip underneath.
Figure 2: asymptotic retention (B = 16384) vs difficulty per level + controls.
Combined: both panels side by side at 2-column paper width (compact, for the
  ~2-page report), inclusion strips beneath each.

  python3 scripts/make_figures.py                                   # real
  python3 scripts/make_figures.py --results-dir <fixture> --n-boot 200

Estimates come from src/extcot/analysis.py (cell_bstar) unmodified; missing
cells are warned about and skipped, so partial grids still render. Also
writes <out-dir>/bstar_summary.json with every plotted number.
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

# Warm print style: cream ground, ink black for the base/reference series,
# a terracotta ramp for ablation strength (ordinal), warm gray-brown dashed
# for the random control. Marker shape + line style are the secondary channel.
BG, INK, MUTED, GRID = "#FAF6EE", "#1A1917", "#8A8072", "#E7E1D2"
STYLE = {
    "light":  dict(color="#E0A87C", marker="o", ls="-",  mfc="none",
                   label="light (26:30, k=10)"),
    "medium": dict(color="#C95F35", marker="s", ls="-",  mfc="none",
                   label="medium (22:34, k=10)"),
    "heavy":  dict(color="#71391D", marker="^", ls="-",  mfc="none",
                   label="heavy (22:34, k=100)"),
    "random": dict(color="#8A7A66", marker="o", ls="--", mfc="none",
                   label="random ctrl (norm-matched)"),
    "base":   dict(color=INK,      marker="o", ls="-",  mfc=INK,
                   label="base (truncated CoT)"),
    "oob":    dict(color="#5E7C8A", marker="D", ls=":",  mfc="none",
                   label="out-of-band ctrl"),
}
LEVEL_TAGS = [("light", "jspace-light"), ("medium", "jspace-med"),
              ("heavy", "jspace-heavy")]

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
    "font.family": "DejaVu Sans", "axes.titleweight": "bold",
})


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
    """analysis.majority (>=2 solved). Cells run with --staged legitimately
    hold 2-3 samples for majority-determined problems; only warn when a
    problem is genuinely undecidable from what's on disk."""
    by_pid = defaultdict(list)
    for r in rows:
        by_pid[r["pid"]].append(bool(r["solved"]))
    undecided = {p: v for p, v in by_pid.items()
                 if len(v) < 4 and sum(v) < 2 and (len(v) - sum(v)) < 3}
    if undecided:
        print(f"  WARN [{tag}]: {len(undecided)} pids undecidable from disk "
              f"(counted unsolved), e.g. {list(undecided)[:3]}")
    return {p: sum(v) >= 2 for p, v in by_pid.items()}


def collect_outcomes(results_dir: str, prefix: str,
                     budgets=BUDGETS) -> dict[int, dict[str, bool]]:
    out = {}
    for B in budgets:
        rows = load_runs(results_dir, f"{prefix}-B{B}")
        if rows:
            out[B] = majority_checked(rows, f"{prefix}-B{B}")
    return out


def base_outcomes(results_dir: str, design: dict) -> dict[int, dict[str, bool]]:
    out = collect_outcomes(results_dir, "base", [0, 128, 512, 2048])
    out[B_MAX] = {pid: p["base_solved"] for pid, p in design["problems"].items()
                  if p.get("in_sample", p["included"])}
    return out


def retention_ci(problems, outcome_bmax, pids, n_boot=500, seed=1):
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


def inclusion_strip(ax, design, edges, xlim, tiny=False):
    fs = 5.0 if tiny else 6.5
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
        ax.bar(lo, r, width=hi - lo, align="edge", color="#D9D2C0",
               edgecolor=BG, linewidth=1.2, alpha=0.9)
        cx = math.sqrt(lo * hi)
        ax.text(cx, 0.12, f"{r:.0%}\nn={ns[b]}", ha="center", va="bottom",
                fontsize=fs, color="#6E6658", linespacing=1.1)
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([])
    ax.set_ylabel("incl.", fontsize=fs + 0.5, color="#6E6658")
    ax.set_xlabel("bin difficulty: median base thinking tokens (log)",
                  fontsize=fs + 1.5)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=fs + 1, length=2)


def style_axes(ax, fs=8):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("bottom", "left"):
        ax.spines[s].set_color("#B9B2A2")
        ax.spines[s].set_linewidth(0.8)
    ax.grid(True, axis="y", which="major", color=GRID, linewidth=0.7)
    ax.grid(False, axis="x")
    ax.tick_params(labelsize=fs, colors=INK, length=2.5)


def styled_legend(ax, fs, loc):
    leg = ax.legend(fontsize=fs, loc=loc, handlelength=2.2, borderaxespad=0.4,
                    frameon=True, fancybox=False, framealpha=0.95,
                    facecolor=BG, edgecolor="#CFC8B6")
    leg.get_frame().set_linewidth(0.8)
    return leg


def draw_censor_arrows(ax, xs, color, jitter=1.0):
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    for x in xs:
        ax.annotate("", xy=(x * jitter, 0.99), xytext=(x * jitter, 0.90),
                    xycoords=tr, textcoords=tr,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4),
                    annotation_clip=False)


def draw_fig1(ax, data, fs=8, compact=False):
    series, rand, base_res, xlim, n_boot = (data["series"], data["rand"],
                                            data["base_res"], data["xlim"],
                                            data["n_boot"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    y_floor = 30
    ax.set_xlim(*xlim)
    ax.set_ylim(y_floor, B_MAX * 2.1)

    xs_ref = np.geomspace(*xlim, 50)
    ax.plot(xs_ref, xs_ref, ls=":", color=MUTED, lw=1.1, zorder=1)
    ax.text(xlim[1] * 0.52, xlim[1] * 0.36, "y = x (base usage)",
            fontsize=fs - 1.5, color=MUTED, rotation=30, ha="center", va="top",
            rotation_mode="anchor")
    ax.axhline(B_MAX, color="#CFC8B6", lw=0.8)
    ax.text(xlim[1] * 0.92, B_MAX * 1.18, "budget cap", fontsize=fs - 2,
            color=MUTED, ha="right", va="bottom")

    def plot_bstar(res, st, jitter=1.0, band=True, lw=1.6, label=None):
        bs = sorted(res)
        x = np.array([res[b]["median_difficulty"] for b in bs])
        y = np.array([res[b]["bstar"] if not res[b]["censored"] else np.nan
                      for b in bs], dtype=float)
        y = np.clip(y, y_floor, None)
        lo = np.array([res[b]["lo"] if res[b]["lo"] is not None else np.nan
                       for b in bs], dtype=float).clip(y_floor, B_MAX * 2.1)
        hi = np.array([res[b]["hi"] if res[b]["hi"] is not None else np.nan
                       for b in bs], dtype=float).clip(y_floor, B_MAX * 2.1)
        ok = ~np.isnan(y)
        if band and ok.any():
            ax.fill_between(x[ok], lo[ok], hi[ok], color=st["color"],
                            alpha=0.14, lw=0, zorder=2)
        ax.plot(x[ok], y[ok], marker=st["marker"], ls=st["ls"], lw=lw,
                ms=4.2 if compact else 5, color=st["color"], zorder=3,
                label=label, markerfacecolor=st["mfc"], markeredgewidth=1.2)
        draw_censor_arrows(ax, [res[b]["median_difficulty"] for b in bs
                                if res[b]["censored"]], st["color"], jitter)

    for i, (name, _) in enumerate(LEVEL_TAGS):
        if name in series:
            plot_bstar(series[name], STYLE[name], jitter=1.09 ** (i - 1),
                       label=STYLE[name]["label"])
    for j, (s, res) in enumerate(sorted(rand.items())):
        plot_bstar(res, STYLE["random"], jitter=1.09 ** (j + 2), band=False,
                   lw=1.1, label=STYLE["random"]["label"] if j == 0 else None)
    if base_res:
        plot_bstar(base_res, STYLE["base"], jitter=1.09 ** -2, band=True,
                   lw=1.4, label=STYLE["base"]["label"])

    ax.set_ylabel("B*: budget for 90% of base rate (log)", fontsize=fs)
    ax.set_title("B* rises faster than base usage;\ncensored ($\\uparrow$) = "
                 "no recovery within cap" if compact else
                 "External budget needed to recover base performance rises "
                 "with difficulty", fontsize=fs + 1, color=INK, loc="left",
                 pad=8)
    # monospace inset: medium B* as a multiple of base B* (prediction 1)
    if "medium" in series and base_res:
        lines = ["med. B*/base B*"]
        for b, v in sorted(series["medium"].items()):
            bb = base_res.get(b)
            if v["censored"] or bb is None or bb["censored"]:
                r = "cens." if v["censored"] else "-"
            else:
                r = f"{v['bstar'] / max(bb['bstar'], 1):.1f}x"
            lines.append(f" bin{b}: {r}")
        ax.text(0.985, 0.03, "\n".join(lines), transform=ax.transAxes,
                fontsize=fs - 2.2, family="monospace", ha="right", va="bottom",
                color="#544C3E",
                bbox=dict(facecolor=BG, edgecolor="#CFC8B6", lw=0.8,
                          boxstyle="square,pad=0.45"))
    styled_legend(ax, fs - 1.5, "upper left")
    style_axes(ax, fs)


def draw_fig2(ax, data, problems, edges, fs=8, compact=False):
    series, rand, oob_out, xlim, n_boot = (data["series"], data["rand"],
                                           data["oob_out"], data["xlim"],
                                           data["n_boot"])
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(0, 1.16)
    ax.axhline(1.0, color="#CFC8B6", lw=0.9)
    ax.axhline(0.9, color=MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.text(xlim[0] * 1.08, 0.905, "recovery threshold (0.9)",
            fontsize=fs - 2, color=MUTED, va="bottom")

    def plot_ret(res, oc, st, lw=1.6, label=None):
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
                                  n_boot=n_boot)
            los.append(lo if lo is not None else np.nan)
            his.append(hi if hi is not None else np.nan)
        ax.fill_between(x, los, his, color=st["color"], alpha=0.14, lw=0)
        ax.plot(x, y, marker=st["marker"], ls=st["ls"], lw=lw,
                ms=4.2 if compact else 5, color=st["color"], label=label,
                markerfacecolor=st["mfc"], markeredgewidth=1.2)

    for name, _ in LEVEL_TAGS:
        if name in series:
            plot_ret(series[name], data[f"oc_{name}"], STYLE[name],
                     label=STYLE[name]["label"])
    for j, (s, res) in enumerate(sorted(rand.items())):
        plot_ret(res, data[f"oc_random-s{s}"], STYLE["random"], lw=1.1,
                 label=STYLE["random"]["label"] if j == 0 else None)
    if oob_out:
        oob_res = analysis.cell_bstar(problems, {B_MAX: oob_out}, edges,
                                      n_boot=0)
        plot_ret(oob_res, {B_MAX: oob_out}, STYLE["oob"], lw=1.2,
                 label=STYLE["oob"]["label"])

    ax.set_ylabel("retention at B = 16384", fontsize=fs)
    ax.set_title("Asymptotic retention falls with difficulty" if compact else
                 "Asymptotic retention (B = 16384) falls with difficulty",
                 fontsize=fs + 1, color=INK, loc="left", pad=8)
    styled_legend(ax, fs - 1.5, "lower left")
    style_axes(ax, fs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="/workspace/jlens-cot/results")
    ap.add_argument("--out-dir", default=None)
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
    series, data = {}, {}
    MIN_BUDGETS = 3   # a series needs a real budget axis to be a curve
    for name, prefix in LEVEL_TAGS:
        oc = collect_outcomes(rd, prefix)
        if len(oc) >= MIN_BUDGETS:
            series[name] = analysis.cell_bstar(problems, oc, edges,
                                               n_boot=args.n_boot,
                                               seed=args.seed)
            data[f"oc_{name}"] = oc
        elif oc:
            print(f"  NOTE: {prefix} has only {len(oc)} budget point(s); "
                  f"omitting from figures")
    rand = {}
    for s in (0, 1):
        oc = collect_outcomes(rd, f"random-s{s}")
        if len(oc) >= MIN_BUDGETS:
            rand[s] = analysis.cell_bstar(problems, oc, edges,
                                          n_boot=args.n_boot, seed=args.seed)
            data[f"oc_random-s{s}"] = oc
        elif oc:
            print(f"  NOTE: random-s{s} has only {len(oc)} budget point(s); "
                  f"omitting from figures")
    oc_base = base_outcomes(rd, design)
    base_res = analysis.cell_bstar(problems, oc_base, edges,
                                   n_boot=args.n_boot, seed=args.seed) \
        if len(oc_base) > 1 else {}
    oob_rows = load_runs(rd, "oob-B16384")
    oob_out = majority_checked(oob_rows, "oob-B16384") if oob_rows else None

    if not series and not base_res:
        sys.exit("No cells found under " + rd)

    meds = {b: v["median_difficulty"]
            for res in [base_res, *[series[n] for n, _ in LEVEL_TAGS
                                    if n in series]]
            for b, v in (res or {}).items()}
    xlim = (min(meds.values()) / 2.2, max(meds.values()) * 2.6)
    data.update(series=series, rand=rand, base_res=base_res, oob_out=oob_out,
                xlim=xlim, n_boot=args.n_boot)

    # ------------------------------------------- individual (full-size) figs
    fig = plt.figure(figsize=(7.0, 5.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[4.4, 0.8], hspace=0.12)
    ax, axs = fig.add_subplot(gs[0]), None
    axs = fig.add_subplot(gs[1], sharex=ax)
    draw_fig1(ax, data, fs=8.5)
    plt.setp(ax.get_xticklabels(), visible=False)
    inclusion_strip(axs, design, edges, xlim)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig1_bstar.{ext}"), dpi=300,
                    bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(7.0, 4.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.6, 0.8], hspace=0.14)
    ax = fig.add_subplot(gs[0])
    axs = fig.add_subplot(gs[1], sharex=ax)
    draw_fig2(ax, data, problems, edges, fs=8.5)
    plt.setp(ax.get_xticklabels(), visible=False)
    inclusion_strip(axs, design, edges, xlim)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig2_retention.{ext}"), dpi=300,
                    bbox_inches="tight")
    plt.close(fig)

    # --------------------------------- combined compact (2-page report) fig
    fig = plt.figure(figsize=(7.05, 3.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[5.2, 0.9], hspace=0.1,
                          wspace=0.24)
    ax1 = fig.add_subplot(gs[0, 0])
    st1 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2 = fig.add_subplot(gs[0, 1])
    st2 = fig.add_subplot(gs[1, 1], sharex=ax2)
    draw_fig1(ax1, data, fs=6.4, compact=True)
    draw_fig2(ax2, data, problems, edges, fs=6.4, compact=True)
    for a in (ax1, ax2):
        plt.setp(a.get_xticklabels(), visible=False)
    inclusion_strip(st1, design, edges, xlim, tiny=True)
    inclusion_strip(st2, design, edges, xlim, tiny=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig_combined.{ext}"), dpi=300,
                    bbox_inches="tight")
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

    print(f"\nWrote fig1, fig2, fig_combined + bstar_summary.json to {out_dir}")
    for k, res in summary.items():
        if k == "bin_edges":
            continue
        cells = " ".join(
            f"{'CENS':>8s}" if v["censored"] else f"{v['bstar']:>8.0f}"
            for _, v in sorted(res.items(), key=lambda kv: int(kv[0])))
        print(f"{k:10s} {cells}")


if __name__ == "__main__":
    main()
