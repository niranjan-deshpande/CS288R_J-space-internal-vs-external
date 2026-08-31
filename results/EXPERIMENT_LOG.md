# Experiment record: The Token Cost of Forced Externalization

CS 2881 HW0 — J-space ablation on Qwen3-4B (Anthropic workspace paper protocol).
Full spec + approved amendments: `SPEC.md`. Level calibration: `results/calibration.md`.
Hardware: 1× A100 80GB (RunPod), preceded by RTX 4090 pilot phase. All runs 2026-08-29/31.

## 1. Design (frozen in `results/design.json`, 2026-08-31 ~21:30 UTC)

- **Model**: Qwen/Qwen3-4B, bf16, HF transformers 5.16.1, thinking mode, forced
  answer prefix `Final answer: \boxed{` in every condition.
- **Problems**: GSM8K test n=150 (seed 0), MATH-500 stratified 50/level (243),
  AIME 2024 (30) → 423. Clean run: 8 samples/problem at B=16384 (3,384 records).
- **Inclusion**: Pile A (samples 0–3) solves ≥2/4 → 396/423 included (94%).
- **Difficulty**: median think tokens over Pile B (4–7) successes.
- **Bins** (edges 0/1K/2K/4K/8K/∞): 49 / 150 / 110 / 58 / 29 included problems;
  per-bin subsample cap 45 → **209 problems per grid cell**. Pile-B base rates
  ~ceiling: 49/49, 150/150, 108/110, 58/58, 28/29 (majority-of-4).
- **Ablation levels** (calibrated; extraction gate passed at all strengths):
  light = band 26:30 k=10 · medium = band 22:34 k=10 (paper protocol) ·
  heavy = band 22:34 k=100. Exemption: clean-pass top-10, always on.
- **Estimator**: per problem per cell, majority of 4 samples (staged early-stop
  once determined — identical outcome by construction); retention = ablated
  majority rate / Pile-B base rate over identical pids; B\* = logistic fit of
  majority vs log B, threshold 0.9 × base; bootstrap-over-problems CIs.

## 2. Cells run (all in `results/runs/<tag>.jsonl`, on the /workspace volume; not in git)

| condition | budgets | problems | provenance |
|---|---|---|---|
| clean (base, 8 samples) | 16384 | all 423 | fresh |
| base sweep | 0 / 128 / 512 / 2048 | 209 solvable | fresh, staged |
| jspace-med (k=10, 22:34) | 0 / 128 / 512 / 2048 / 16384 | 209 | 16K fresh+staged; sub-16K derived by prefix reuse + backfill |
| random-s0 (norm-matched, medium band/k) | 0 / 128 / 512 / 2048 / 16384 | 209 | same as medium |
| jspace-heavy (k=100, 22:34) | 0 / 128 / 512 / 2048 / 16384 | **25/bin, bins 0–2 only** (16K & derived); B=0 on all 209; B=2048 also fresh on bins 3–4 | deadline triage, see §5 |
| jspace-light (k=10, 26:30) | 0 | 209 | B>0 cells not run (see §5) |
| random-s1 | 0 | 209 | B>0 cells not run (see §5) |
| out-of-band (4:16, k=10) | — | — | not run (see §5) |
| selectivity | 0 | MMLU 500 + SST-2 300 | clean + medium, 1 sample |
| faithfulness (trunc @50%) | — | 50/point | clean (Pile-B sample) + medium-16K |
| adaptive budget points | 1024 & 4096 | bins 0–2 | medium + random-s0, derived |

Prefix-reuse derivation is exact under the spec's budget-as-truncation definition
(generation is budget-oblivious until the forcing check); sub-16K budget cells share
think prefixes with their 16K source (common-random-numbers pairing — lower-variance
budget curves, unbiased marginals). Staged sampling runs samples {0,1} then only
undecided problems get samples 2/3; per-sample statistics use samples {0,1} only.

## 3. Headline results

- **B\* vs difficulty (medium)**: 891 / 4,467 / 3,677 / censored / censored across
  bins vs clean usage 834 / 1,390 / 2,512 / 6,029 / 9,393 → externalization costs
  more tokens than the base model's own usage (ratio 1.07→3.2), and **recovery
  fails entirely above ~4K difficulty** (retention @16K: 0.89 / 0.82; every
  bootstrap replicate censored). Predictions 1 & 2: supported.
- **Random control (norm-matched, seed 0)**: B\* = 1,016 / 3,585 / 5,067 /
  censored / censored; retention @16K 0.89 / 0.89 in the top bins —
  **statistically indistinguishable from medium jspace**. Prediction 3 fails at
  paper-protocol strength: the math token-cost is perturbation-magnitude-driven,
  not J-space-direction-specific. Caveat that rescues specificity: the control's
  norm-matching removes ~3× more norm than jspace does on its own trajectory
  (removal fraction ~0.14–0.17 vs ~0.05), so per unit norm removed, J-space
  directions are ~3× more damaging.
- **Selectivity battery (medium)**: MMLU 69.0%→67.6%, SST-2 86.3%→84.0% —
  extraction/selection intact. Pilots: closed-book recall 20/20→10–12/20 while
  in-context extraction stays 18–20/20 at every k (dissociation widens with k).
- **B=0 row**: base 27.6% vs all ablated levels 16–19% (random ≈ jspace here too).
- **Faithfulness (truncate trace @50%)**: clean 48/50 → 44/50. Medium: see
  `results/runs/faith-jspace-med-B16384-f0.5.jsonl` (prediction 4: ablated should
  drop more).
- **Heavy (k=100)**: bins 0–2 at full budget grid — see figures; bins 3–4 marked
  censored a fortiori (medium already censored there; k=100 ⊃ k=10 damage),
  evidenced by heavy-B2048 on bins 3–4.

## 4. Performance engineering (all changes bit-exact or distribution-exact, validated)

| change | measured effect |
|---|---|
| GQA fast-decode attention (fold groups into q_len; avoids repeat_kv under mask) | 2.5–3.6× decode; bit-identical logits |
| PreallocCache (block-grown KV, no per-step cat) | eliminates 41–77 GB/step copy traffic; step 118→82ms |
| BandFork (share layers < band between clean/ablated passes) | 1.2–1.3×; 30% KV saved; bit-identical |
| Prefix reuse for sub-16K cells | each ~2h cell → minutes |
| Staged sampling (early-stop determined majorities) | 16–45% fewer samples per cell |
| Length-sorted bucketed scheduling (windows 1024·2^b) | ~1.5× effective throughput vs naive batching |
| sync-free `solve_ex`, on-GPU stats, vectorized sampler | hooks 43→24ms (k=10); pick-identical sampling |
| rejected: restricted-vocab lens scoring | candidate sets unfaithful at deep layers (8% overlap @L34) |
| rejected: Cholesky solve, torch.compile, StaticCache | not bit-exact |

## 5. Deviations from SPEC (all deadline-driven, user-approved 2026-08-31)

1. **Heavy 16K restricted to bins 0–2 at 25/bin** (spec bin floor): the k=100 cell
   was on pace for ~14h (nothing solves → everything generates to the 16K cap).
   Bins 3–4 are censored a fortiori and evidenced at B=2048.
2. **Light level, random seed 1, out-of-band control: B>0 cells not run** (B=0
   row exists for light/random-s1). The random control therefore has 1 seed, not 2.
3. Faithfulness at 2 grid points per amendment 3 (not a deviation, noted for completeness).

## 6. Reproduction

`bash setup.sh` (fresh pod) → `python3 scripts/run_cell.py --tag clean --mode none
--budget 16384 --samples 8 --problems all` → `python3 scripts/build_bins.py` →
`bash scripts/run_grid.sh` (or `final_push4.sh` for the triaged version) →
`python3 scripts/make_figures.py && python3 scripts/make_tables.py`.
Seeds: zlib.crc32(pid|sample|tag) — every record independently reproducible.
Engine invariant: per-sequence RNG ⇒ results independent of batch composition.
