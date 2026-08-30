---
name: jlens-externalization-project
description: Standing user instructions and approved design decisions for the J-lens forced-externalization experiment (CS 2881 HW0)
metadata: 
  node_type: memory
  pinned: true
  originSessionId: 1fad82f2-692c-465b-8150-9f33013a371d
  modified: 2026-08-30T05:21:48.120Z
---

# Project: Token Cost of Forced Externalization (CS 2881 HW0)

The user (ndeshpande@college.harvard.edu, Harvard) is implementing an experiment measuring how many external CoT tokens an ablated Qwen3-4B needs to recover base performance, based on Anthropic's J-lens / global-workspace paper (transformer-circuits.pub/2026/workspace, companion code github.com/anthropics/jacobian-lens). The full spec lives in the project repo.

## Standing user instructions

- **Keep everything important on `/workspace`** (the RunPod network volume, mounted at `/workspace` — NOT `/workplace`, which is an empty local dir). The GPU pod's local 40GB disk is ephemeral; the user will not keep the GPU on indefinitely. Scripts, results, transcripts, and the repo must live on /workspace. Only rebuildable caches (HF model weights) may live locally.
- **Check in periodically** — especially with time budgets, ETAs, and judgment calls being made. The user may steer; absent steering, continue autonomously until all experiments are done.
- Work autonomously through the whole experiment; the user approved this explicitly (re-confirmed 2026-08-29: "assume you have my permission to go ahead and use your best judgement").
- **Send push notifications (PushNotification tool) at important junctures** — e.g. calibration locked, long runs finished, a decision genuinely needs the user — and immediately if anything goes massively wrong. Routine progress stays in the terminal, not notifications.

## Approved spec changes (user signed off 2026-08-29)

1. MATH-500 → 250 problems, stratified 50 per level (saves ~35% of total compute).
2. Exemption-disabled variant: cut entirely.
3. Faithfulness spot check: keep, at 2 grid points (medium ablation × 16384 and base × 16384), reusing existing transcripts.
4. Retention metric fix: base rate per bin uses majority-of-4 on Pile B (not pooled sample-level rate), matching the ablated outcome estimator and the inclusion rule.
5. Added: base-model budget sweep at {128, 512, 2048} through the same truncation protocol (16K reused from clean runs) as a stronger reference line for the slope>1 claim.
6. Added: one adaptive budget point per bin in a second pass, near each bin's preliminary B* crossing.
7. Added free sensitivity checks: (a) difficulty axis recomputed as median-over-all-attempts vs median-over-successes; (b) quantify verbosity confound by comparing ablated vs base token usage at B=16384.

Do-not-cut items confirmed: random control (2 seeds, full budget grid), 16K cells, 25-problem bin floor.

## Session continuity (GPU swap 2026-08-29)

The user swapped the RTX 4090 pod for an A100 mid-project. On a fresh pod:
run `bash /workspace/jlens-cot/setup.sh`, then **read
/workspace/jlens-cot/HANDOFF.md** — it carries the full experiment state,
validated findings, next steps, and hard-won gotchas (OOM/memory settings,
seed hygiene, forced answer prefix). SPEC.md in the repo holds the user's
spec verbatim with all approved amendments.

## Usage-budget awareness (user, 2026-08-30, clarified)

The user has ample credits on a 5-hour resetting limit, but running out of
credits mid-monitoring is the failure mode to avoid: **while the user is
asleep/away, default to monitoring-only** (cheap ~50-min checks, no
subagents) so the watch never stops. Targeted efficiency/quality subagents
are welcome when the user is around — **at most one at a time**, never a
large fan-out (user, 2026-08-30).

## Deadline and cost constraints (user, 2026-08-30)

HW0 is due **Monday 2026-08-31 5pm (ET)**; the user wants experiments done
well before then to leave time for the writeup, and is cost-sensitive about
GPU spend. Prioritize throughput optimizations and validity-preserving
design cuts; flag cost/time tradeoffs (e.g. a second pod) as user decisions.

## Key resources

- Pre-fitted Qwen3-4B Jacobian lens exists at HF `neuronpedia/jacobian-lens` under `qwen3-4b/jlens/Salesforce-wikitext/` — no need to estimate J_l ourselves.
- Hardware: RTX 4090 24GB, 48 cores, 251GB RAM. HW0 requires: ~2-page report.pdf + private reproducible GitHub repo.
