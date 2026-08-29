# The Token Cost of Forced Externalization

CS 2881 HW0. Measures how many external chain-of-thought tokens an ablated
Qwen3-4B needs to recover base performance, as a function of problem
difficulty. J-space ablation follows Anthropic's global-workspace paper
("Verbalizable Representations Form a Global Workspace in Language Models",
transformer-circuits.pub/2026/workspace), using the companion `jlens` package
and a pre-fitted Qwen3-4B Jacobian lens from `neuronpedia/jacobian-lens`
(wikitext-103, n=479 prompts, converged).

## Layout

- `src/extcot/` — library
  - `model.py` — Qwen3-4B loading, prompt construction (thinking / direct)
  - `ablation.py` — J-space ablation forward hooks (paper protocol: per
    position, per layer in band, zero the projection onto the top-k most
    active J-lens vectors; clean-top-10 token exemption; norm-matched random
    control)
  - `engine.py` — batched generation with thinking-budget forcing, forced
    answer prefix, per-sequence RNG, KV-budget eviction/resume
  - `grading.py` — \boxed extraction; GSM8K numeric / math-verify equivalence
  - `data.py` — GSM8K (test, n=150, seed 0), MATH-500 (n=243, 50/level,
    seed 0), AIME 2024 (HuggingFaceH4/aime_2024, all 30), MMLU (n=500, seed 0)
  - `analysis.py` — retention, logistic B* fits, bootstrap CIs
- `scripts/` — entry points (see Reproduction)
- `results/` — run JSONLs, frozen design, figures

## Reproduction

Hardware: one 24GB GPU (RTX 4090 used), ~64GB RAM. Install:

```bash
pip install torch transformers accelerate datasets math-verify scipy pandas matplotlib
pip install -e /path/to/anthropics-jacobian-lens
pip install -e .
```

Pipeline (each step is checkpointed and resumable):

```bash
# 1. Layer diagnostics -> workspace band (results/layer_diagnostics.json)
python3 scripts/diagnostics.py

# 2. Clean runs: 8 samples/problem at 16384 cap
python3 scripts/run_cell.py --tag clean --mode none --budget 16384 --samples 8 \
    --problems all --batch 20

# 3. Freeze design: inclusion, difficulty, bins (results/design.json)
python3 scripts/build_bins.py

# 4. Condition grid (see scripts/run_grid.sh for the full cell list)
# 5. Faithfulness spot check
# 6. Analysis + figures
```

TODO: finalize as experiments complete.
