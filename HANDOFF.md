# Session handoff — read this first (written 2026-08-29, pre-GPU-swap)

You are resuming the CS 2881 HW0 experiment "The Token Cost of Forced
Externalization" mid-flight. The previous session (RTX 4090) built and
validated the full pipeline; the pod was swapped for an A100 before the main
runs. The user's spec is reproduced in SPEC.md; approved changes and standing
instructions are in the restored Claude memory (jlens-externalization-project).

## Immediate next steps (in order)

1. `bash /workspace/jlens-cot/setup.sh` (if the user hasn't already).
2. Re-measure batch limits on the A100 (80GB removes the 24GB dual-KV-cache
   ceiling; batch 32-64 should work everywhere; keep kv_budget_gb generous,
   e.g. 40-60 on 80GB). Quick bench: `python3 scripts/bench.py`.
3. Rerun calibration pilots at A100 batch sizes (~30 min total):
   - `python3 scripts/pilot_strength.py`  (k-sweep; the ONE row completed on
     the 4090: heavy(22-34) k=25 B=2048 -> 46/60 vs clean 52/60, rem_frac
     0.056 — the first math effect with exemption on)
   - `python3 scripts/pilot_bands2048.py` (k=10 band rows lost to OOM)
   - `python3 scripts/check_extraction.py` (recall-vs-extraction dissociation,
     answers a user concern: recall should break, extraction survive)
4. Lock the three ablation levels from pilot results (likely: light = k10,
   medium = k25, heavy = k50, all on heavy band 22:34 — but let the data
   decide; keep k=10 as one level since it's the paper's protocol). Edit the
   level definitions at the top of `scripts/run_grid.sh`. Record the
   calibration table in results/calibration.md.
5. Launch the clean run: `python3 scripts/run_cell.py --tag clean --mode none
   --budget 16384 --samples 8 --problems all --batch <A100-sized>`  (~6h on
   4090, likely ~2-3h on A100).
6. `python3 scripts/build_bins.py` — freezes design.json (inclusion,
   difficulty, bins). Consider the approved per-bin subsample cap (~45/bin)
   when finalizing.
7. `bash scripts/run_grid.sh` (checkpointed; safe to interrupt/re-run).
8. Faithfulness (`scripts/run_faithfulness.py`), adaptive second-pass budget
   points, analysis (`src/extcot/analysis.py`), figures, 2-page report.pdf.

## Validated findings so far (pilot-scale, n=20-60)

- Ablation protocol works and is selective: two-hop recall 23/30 -> 13/30,
  one-hop 20/20 -> 12/20 under jspace k=10; norm-matched random control
  intact (21/30, 20/20). Failures are fluent lost-intermediate errors.
- One-hop recall breaking is CONSISTENT with the paper (TriviaQA is on its
  impaired list); the real selectivity axis is generation-from-latent vs
  extraction/selection (hence check_extraction.py + MMLU/SST-2 in grid).
- Math is immune to k=10 ablation everywhere tested (B=0 and B=256 CoT,
  any band width; exact ties with clean). Explanation: the clean-top-10
  exemption shields numeric answer candidates at answer positions; latent
  intermediates are what ablation removes. k=25 on heavy band is the first
  strength that bites math (46/60 vs 52/60 at B=2048).
- Clean budget axis has good dynamic range on the hard pilot slice
  (MATH lvl>=4 x40 + GSM8K x20): B=256 -> 16/60, B=2048 -> 52/60.
- Workspace band for Qwen3-4B: layers ~22-34 of 36 (later than the paper's
  relative 38-92%). Layers 14-21 add nothing. Diagnostics in
  results/layer_diagnostics.json; J-lens reads two-hop intermediates
  (' Italy') at L30 where logit lens is noise.

## Hard-won gotchas (do not relearn these)

- ALWAYS `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (in
  run_grid.sh already). Two OOMs on 24GB came from fragmentation + dual
  caches; engine has kv_budget eviction/resume (waves) — run_cell handles it.
- Python `hash()` is process-salted: NEVER use it for seeds (pilots did,
  harmless there; run_cell uses zlib.crc32 — keep it that way).
- Do not chain background jobs with `pgrep -f <script>` — the waiter's own
  command line matches and deadlocks. Chain via task notifications instead.
- The forced answer prefix ("Final answer: \boxed{" after </think>) is
  ESSENTIAL: without it the model reasons in the answer channel and the
  budget axis is meaningless. For thinking=False the prefix is in the PROMPT,
  and grading prepends it back (see engine._finalize).
- transformers 5.16: DynamicCache.batch_select_indices for compaction;
  logits_to_keep=1 works; left padding + explicit position_ids everywhere.
- Pre-fitted lens: /workspace/artifacts/neuronpedia-jlens/qwen3-4b/jlens/
  Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt (layers 0-34, fp32 after
  load; fitted on Qwen/Qwen3-4B, converged, n=479). Ablation directions:
  g_v = J_l^T (gain ⊙ w_v); score/topk via z = (h @ J.T) @ W_tilde.T.
- A cell's per-(pid,sample_idx) records are resumable; JSONLs in results/runs/.
- AIME only at B in {2048, 16384}. MATH is 243 problems (50/level cap).
- User wants periodic check-ins with ETAs and judgment calls. Full history of
  judgment calls is in git log and this file.

## Cost model (4090 numbers; scale by measured A100 speedup)

Grid ≈ 45-55 GPU-hours on 4090 at ~150-590 tok/s; expect 2.5-4x on A100 80GB.
Dominant cost: the five B=16384 ablated/control cells.
