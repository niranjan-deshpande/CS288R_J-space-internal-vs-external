# The Token Cost of Forced Externalization

CS 2881 HW0 — measures how many external chain-of-thought tokens a
J-space-ablated Qwen3-4B needs to recover base performance, as a function of
problem difficulty. **The report is [`report.pdf`](report.pdf)** (root of this
repo). A comprehensive experiment record — every cell run, design decisions,
deviations, and performance engineering — is
[`results/EXPERIMENT_LOG.md`](results/EXPERIMENT_LOG.md); the frozen
experimental spec is [`SPEC.md`](SPEC.md).

J-space ablation follows Anthropic's global-workspace paper
(["Verbalizable Representations Form a Global Workspace in Language Models"](https://transformer-circuits.pub/2026/workspace)),
using the companion [`jlens` package](https://github.com/anthropics/jacobian-lens)
and the pre-fitted Qwen3-4B Jacobian lens from
[`neuronpedia/jacobian-lens`](https://huggingface.co/neuronpedia/jacobian-lens)
(`qwen3-4b/jlens/Salesforce-wikitext/`; wikitext-103, n=479 prompts, converged).
Datasets: [GSM8K](https://huggingface.co/datasets/openai/gsm8k) (test, n=150),
[MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) (n=243,
~50/level), [AIME 2024](https://huggingface.co/datasets/HuggingFaceH4/aime_2024)
(n=30), plus [MMLU](https://huggingface.co/datasets/cais/mmlu) (n=500) and
SST-2 (n=300) for the selectivity battery.

## Layout

- `report.pdf` — the report.
- `src/extcot/` — library
  - `model.py` — Qwen3-4B loading, prompt construction (thinking / direct),
    fast GQA decode attention
  - `ablation.py` — J-space ablation forward hooks (paper protocol: per
    position, per layer in band, remove the residual-stream projection onto
    the span of the top-k most active J-lens vectors; clean-top-10 token
    exemption; norm-matched random-direction control)
  - `engine.py` — batched generation with thinking-budget forcing, forced
    answer prefix `Final answer: \boxed{`, per-sequence RNG (results are
    independent of batch composition), KV-budget eviction/resume
  - `grading.py` — \boxed extraction; GSM8K numeric / math-verify equivalence
  - `data.py` — dataset loading (seeds fixed)
  - `analysis.py` — majority-of-4 outcomes, floor-logistic B* fits
    (p = c + (1-c)·σ(a + b·log(B+1))), bootstrap-over-problems CIs
- `scripts/` — entry points
  - `run_cell.py` — run one (condition × budget) cell; `--staged` enables
    early-stop majority sampling
  - `run_reuse.py` — derive sub-16K budget cells from 16K transcripts by
    prefix reuse (exact under the budget-as-truncation definition)
  - `build_bins.py` — freeze inclusion, difficulty, bins (`results/design.json`)
  - `run_grid.sh` + `grid_lib.sh` — the full condition grid;
    `final_push*.sh` — the deadline-triaged sequence actually used
  - `run_faithfulness.py`, `diagnostics.py`, `slim_runs.py`
  - `make_figures.py`, `make_tables.py` — everything in the report
- `results/` — `design.json` (frozen inclusion/difficulty/bins),
  `tables.md` (all numeric tables), `calibration.md` (level calibration),
  `EXPERIMENT_LOG.md`, `runs-slim/` (see below), logs
- `figures/` — `fig_combined.{pdf,png}` (the report figure), full-size
  `fig1_bstar` / `fig2_retention`, `bstar_summary.json` (all B*, CIs,
  retention values)

## Reproducing the report's figures and tables (CPU-only, ~5 minutes)

Every record's analysis-relevant fields (problem id, sample index, solved,
token counts, condition metadata) are committed in `results/runs-slim/`
(4MB). Only the raw generated-token arrays are omitted (215MB; they are not
read by any analysis code). From the repo root:

```bash
pip install numpy scipy matplotlib datasets transformers
cp results/runs-slim/*.jsonl results/runs/
python3 scripts/make_figures.py --results-dir results --out-dir figures
python3 scripts/make_tables.py  --results-dir results
```

This regenerates `figures/fig_combined.pdf` (the report figure),
`figures/bstar_summary.json`, and `results/tables.md` — verified to produce
identical B* values to the shipped artifacts.

## Full reproduction from scratch (GPU)

Hardware used: 1× A100 80GB (a 24GB GPU works with smaller `--batch` /
`--kv-budget`). Install:

```bash
pip install torch transformers accelerate datasets math-verify scipy pandas matplotlib
pip install -e /path/to/anthropics-jacobian-lens   # github.com/anthropics/jacobian-lens
pip install -e .
```

Pipeline (each step checkpointed and resumable; seeds are
`zlib.crc32(pid|sample|tag)`, so every record is independently reproducible):

```bash
# 1. Layer diagnostics -> workspace band (results/layer_diagnostics.json)
python3 scripts/diagnostics.py

# 2. Clean runs: 8 samples/problem at the 16384 cap (~9h on A100)
python3 scripts/run_cell.py --tag clean --mode none --budget 16384 --samples 8 \
    --problems all --batch 48 --kv-budget 48

# 3. Freeze design: inclusion, difficulty, bins (results/design.json)
python3 scripts/build_bins.py

# 4. Condition grid: 16K cells fresh, sub-16K derived by prefix reuse
bash scripts/run_grid.sh          # or final_push4.sh for the triaged version

# 5. Faithfulness spot check
python3 scripts/run_faithfulness.py

# 6. Analysis, figures, tables
python3 scripts/make_figures.py && python3 scripts/make_tables.py
```

## Artifacts not stored in the repo

- **Full run transcripts** (`results/runs/*.jsonl`, 215MB): the committed
  `results/runs-slim/` mirror contains every field the analysis reads; the
  full files add only the generated token ids/text. Regenerate them exactly
  via the pipeline above (deterministic per-record seeds), or request a copy.
- **Model weights / J-lens**: downloaded automatically from Hugging Face
  (`Qwen/Qwen3-4B`, `neuronpedia/jacobian-lens`) on first run.

## Notes for graders

- `results/runs/_archived-heavy-B16384-*.jsonl` (not committed) were killed
  partial cells, excluded from all analysis (early-finisher bias); see
  `results/EXPERIMENT_LOG.md` §5 for all deadline-driven deviations from
  `SPEC.md`.
- `HANDOFF.md` is an internal working document kept for provenance.
