# Specification: The Token Cost of Forced Externalization

(User's original spec, verbatim. Approved amendments listed at the bottom.)

## Hypothesis

J-space ablation removes the model's internal scratchpad. External chain of
thought can substitute for it, but only for storing intermediates, not for the
per-step computation that produces them. Predictions:

1. B*, the external token budget needed to recover 90% of base performance,
   rises with difficulty, and rises faster than the base model's own token
   usage (slope > 1 on the main plot).
2. At some difficulty, recovery fails at any budget: asymptotic retention
   falls below 0.9. This frontier moves to lower difficulty as ablation
   strength rises.
3. Matched-norm random ablation shows flat B* and no frontier.
4. Ablation increases the causal dependence of the answer on the written
   trace (truncation hurts the ablated model more than the base model).

The design makes the opposite results equally visible: if CoT fully
compensates at all difficulties, B* stays near the base model's token usage
and no frontier appears.

## Motivation

This measures the price, in external tokens and residual capability, of
forcing computation out of the latent workspace and into the monitorable
channel — relevant to CoT monitorability, since a monitor is only useful if
the trace is load-bearing and not too long to read. The deployment story is
speculative; say so in the analysis section, one paragraph.

## Model and data

- Model: Qwen3-4B, HF transformers, bf16, forward hooks for ablation. Not
  vLLM (no residual-stream access).
- Datasets: GSM8K (test split, n=150 random with fixed seed), MATH-500 (all
  500; drop to 250 stratified by level if compute-bound), AIME 2024
  (`HuggingFaceH4/aime_2024`, all 30 — state this source in the report).
- Answer extraction: force a fixed answer format; grade with exact match
  after normalization (sympy-based equivalence for MATH/AIME).

## Definitions

- **Base model:** unablated, thinking mode on, budget capped at 16,384
  thinking tokens (this is "uncapped").
- **Budget B:** max thinking tokens. Enforce by truncating the think block at
  B tokens, appending `</think>` and the answer prefix. B = 0 means no think
  block (direct answering). Log tokens actually used in every run.
- **Clean samples:** n = 8 per problem, temperature per Qwen3 recommended
  sampling settings, fixed seeds. Split into Pile A (4) and Pile B (4).
- **Inclusion (solvable set):** a problem is included iff Pile A solves >= 2
  of 4. Decided once, final. Pile A quantities are never reused for
  estimation (transcripts may be reused qualitatively).
- **Base rate per bin:** Pile B success rate, pooled over the bin's included
  problems. [AMENDED: majority-of-4 per problem, see below]
- **Difficulty per problem:** median thinking tokens over Pile B's
  *successful* attempts. Continuous, shared units across datasets.
- **Bins:** log-spaced edges over difficulty, e.g. [0, 250), [250, 500),
  [500, 1000), [1000, 2000), [2000, inf). Floor: 25 included problems per
  bin; merge adjacent bins below the floor. Fix edges after seeing clean
  runs, before seeing any ablated results.
- **Ablated outcome per problem per (ablation level, B):** 4 fresh samples;
  problem counts as solved iff majority (>= 2 of 4, matching the inclusion
  vote).
- **Retention per bin:** ablated solve rate / Pile-B base rate, same problems.
- **Threshold:** retention >= 0.9.
- **B\* per bin per ablation level:** fit a logistic of per-problem
  solved/not against log B over the budget grid; B* is where the fitted curve
  crosses the threshold. Bootstrap over problems for CIs. If the fitted
  asymptote never crosses 0.9, the bin is censored: mark with an arrow at the
  top of the plot, do not drop.

## Ablation

- **J-lens vectors:** first check the paper's released artifacts and
  Neuronpedia for Qwen3-4B. [RESOLVED: pre-fitted lens found at
  neuronpedia/jacobian-lens qwen3-4b, validated causally.]
- **Protocol (following the paper):** at each token position, across a layer
  band, zero the residual stream's projection onto the top k = 10 most active
  J-lens vectors. Exempt tokens in the top-10 of the clean forward pass.
- **Levels:** light / medium / heavy = three layer-band widths within the
  workspace band. [AMENDED: band widths give no gradient on Qwen3-4B; levels
  differ in k instead — see calibration.]
- **Exemption variant:** CUT (approved).

## Condition grid

- Ablation levels: {light, medium, heavy}.
- Budgets: {0, 128, 512, 2048, 16384}.
- Samples: 4 per problem per cell, on the solvable set only.
- AIME: run only at B in {2048, 16384}. Expect wide intervals on 30 problems;
  report them.

## Controls (do not cut)

1. **Matched random ablation:** k random directions, norm-matched to the
   removed J-space projection, medium layer band, 2 seeds, full budget grid,
   same pipeline, same solvable set, same splits.
2. **Out-of-band ablation:** same protocol outside the workspace band, one
   budget (16384), one ablation level.
3. **Selectivity battery:** MMLU slice (~500) + SST-2 (~300) under medium
   ablation, direct answering. Should be near baseline.
4. **Fluency/verbosity logging:** tokens used and a coarse fluency check on
   ablated CoT.

## Faithfulness spot check

At 2 grid points (medium ablation x 16384 and base x 16384): truncate the
generated trace at 50% and force an answer; measure the accuracy drop.
~50 problems per point. [AMENDED from 2-3 points to 2.]

## Analysis and figures

- **Figure 1 (main):** B* vs. bin difficulty, one line per ablation level,
  random-control line overlaid, censored bins arrowed, y = x reference line
  (base model's own usage). Bootstrap CIs.
- **Figure 2:** asymptotic retention (B = 16384) vs. difficulty, per ablation
  level plus control.
- **Strip under both:** inclusion rate per bin.
- **Robustness check:** one pooled logistic, success ~ log B + log difficulty
  (+ interaction), compared against binned B*.
- **Cross-check:** difficulty proxy vs. MATH level annotations; GSM8K vs.
  MATH behavior at matched token difficulty.
- **Error taxonomy:** ~30 ablated-CoT failures per bin: within-step
  arithmetic slip, invalid step-to-step inference, lost intermediate, no
  viable approach, format failure.

## Caveats to state in the report

- METR analogy is a borrowed plot style, not a borrowed construct.
- Difficulty is model-relative; MATH-level cross-check partially covers this.
- Solvable-set survivorship at high difficulty.
- Deployment untested; models might route around the J-space; per-position
  projections cost compute.
- If token usage saturates below the cap, measured B* is bounded by
  verbosity, not ability. Report where this happens.

## Approved amendments (user, 2026-08-29)

1. MATH-500 -> 243 problems (50/level, seed 0).
2. Exemption-disabled variant cut.
3. Faithfulness kept, at 2 grid points.
4. Base rate per bin = majority-of-4 on Pile B (estimator consistency fix).
5. Added base-model budget sweep at {128, 512, 2048} (+ B=0), same truncation
   protocol, as the reference line for prediction 1.
6. Added one adaptive budget point per bin (second pass, near preliminary B*).
7. Added sensitivity checks: difficulty median-over-all-B vs over-successes;
   verbosity confound quantified at B=16384.
8. Forced answer prefix "Final answer: \boxed{" in ALL conditions (this is
   the spec's "answer prefix"; makes the think block the only reasoning
   channel).
9. Ablation levels differ in k (10/25/...) on the measured workspace band
   (22-34) if band width gives no gradient — calibration pilot decides.
   [User informed 2026-08-29, not yet explicitly confirmed; k=10 kept as one
   level in any case.]
