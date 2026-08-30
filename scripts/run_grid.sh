#!/bin/bash
# Master condition grid, v2 (A100, deadline-optimized — see results/calibration.md
# and scripts/run_reuse.py). Structure:
#   - Only B=16384 cells are generated fresh; every B in {128,512,2048} cell is
#     DERIVED from its 16K source by prefix reuse (exact under the spec's
#     budget-as-truncation definition; answers regenerated under the cell's
#     condition with fresh per-cell seeds).
#   - B=0 cells are fresh (different prompt path, cheap).
#   - Cells run in report-value order: medium + random-s0 first, light last,
#     so the core figures survive an early stop.
# Every cell is checkpointed+resumable; re-run this script after interruption.
set -euo pipefail
cd /workspace/jlens-cot
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN="python3 scripts/run_cell.py"
REUSE="python3 scripts/run_reuse.py"
LOG=results/grid.log
exec >> $LOG 2>&1
trap 'echo "GRID FAILED at line $LINENO (cell above this message)"; date' ERR

[ -f results/design.json ] || { echo "FATAL: results/design.json missing — run build_bins.py first"; exit 1; }

# ---- ablation levels (locked 2026-08-30, results/calibration.md) ----
LIGHT_BAND="26:30";  LIGHT_K=10
MED_BAND="22:34";    MED_K=10     # paper protocol on measured workspace band
HEAVY_BAND="22:34";  HEAVY_K=100
OOB_BAND="4:16"      # out-of-band control (13 layers, width-matched to medium)

run16 () {  # $1 tag-prefix  $2 band  $3 k  $4 mode  $5 extra
  # --staged: early-stop samples once the majority-of-4 outcome is determined
  $RUN --tag $1-B16384 --mode $4 --band $2 --k $3 --budget 16384 --samples 4 \
    --problems solvable --datasets gsm8k,math,aime --batch 128 --kv-budget 65 \
    --staged ${5:-}
}
derive_one () {  # $1 tag-prefix $2 band $3 k $4 mode $5 budget $6 datasets $7 batch $8 extra
  # staged-source derive + backfill loop: run_reuse derives what the staged
  # 16K source has; problems still undecided at this budget that lack a
  # sample get it generated fresh (backfill-*.json), then re-derive
  local tag=$1-B$5
  local pass
  for pass in 1 2 3; do
    $REUSE --source-tag $1-B16384 --tag $tag --budget $5 --mode $4 --band $2 \
      --k $3 --datasets $6 --batch $7 --kv-budget 55 --staged-source ${8:-}
    local more=0 s f
    for s in 2 3; do
      f=results/runs/backfill-$tag-s$s.json
      if [ -s "$f" ] && [ "$(cat "$f")" != "[]" ]; then
        more=1
        $RUN --tag $tag --mode $4 --band $2 --k $3 --budget $5 --samples 1 \
          --sample-offset $s --problems $f --datasets $6 --batch 64 \
          --kv-budget 55 ${8:-}
      fi
    done
    if [ "$more" -eq 0 ]; then break; fi
  done
}
derive () {  # $1 tag-prefix  $2 band  $3 k  $4 mode  $5 extra
  # B=2048 derives prefill ~2.3K tokens/row under dual caches: keep batch modest
  derive_one $1 $2 $3 $4 2048 gsm8k,math,aime 48 "${5:-}"
  derive_one $1 $2 $3 $4 512  gsm8k,math 96 "${5:-}"
  derive_one $1 $2 $3 $4 128  gsm8k,math 96 "${5:-}"
}
b0 () {  # $1 tag-prefix  $2 band  $3 k  $4 mode  $5 extra
  $RUN --tag $1-B0 --mode $4 --band $2 --k $3 --budget 0 --samples 4 \
    --problems solvable --datasets gsm8k,math --batch 128 --kv-budget 65 \
    --staged ${5:-}
}

date; echo "=== smoke: selectivity (MMLU + SST-2, direct answering) ==="
$RUN --tag select-clean --mode none --budget 0 --samples 1 --problems all --datasets mmlu,sst2 --batch 128 --kv-budget 65
$RUN --tag select-med   --mode jspace --band $MED_BAND --k $MED_K --budget 0 --samples 1 --problems all --datasets mmlu,sst2 --batch 96 --kv-budget 65

date; echo "=== B=0 row (fresh, cheap) ==="
$RUN --tag base-B0 --mode none --budget 0 --samples 4 --problems solvable --datasets gsm8k,math --batch 128 --kv-budget 65
b0 jspace-med   $MED_BAND   $MED_K   jspace
b0 jspace-heavy $HEAVY_BAND $HEAVY_K jspace
b0 jspace-light $LIGHT_BAND $LIGHT_K jspace
b0 random-s0    $MED_BAND   $MED_K   random "--rand-seed 0"
b0 random-s1    $MED_BAND   $MED_K   random "--rand-seed 1"

date; echo "=== medium level (core) ==="
run16  jspace-med $MED_BAND $MED_K jspace
derive jspace-med $MED_BAND $MED_K jspace

date; echo "=== random control seed 0 ==="
run16  random-s0 $MED_BAND $MED_K random "--rand-seed 0"
derive random-s0 $MED_BAND $MED_K random "--rand-seed 0"

date; echo "=== heavy level ==="
run16  jspace-heavy $HEAVY_BAND $HEAVY_K jspace
derive jspace-heavy $HEAVY_BAND $HEAVY_K jspace

date; echo "=== base budget sweep (FRESH: reuse from clean would couple the ==="
echo "=== reference line to the design freeze via Pile B; Pile A is spec-barred) ==="
$RUN --tag base-B2048 --mode none --budget 2048 --samples 4 --problems solvable --datasets gsm8k,math,aime --batch 128 --kv-budget 65 --staged
$RUN --tag base-B512  --mode none --budget 512  --samples 4 --problems solvable --datasets gsm8k,math --batch 128 --kv-budget 65 --staged
$RUN --tag base-B128  --mode none --budget 128  --samples 4 --problems solvable --datasets gsm8k,math --batch 128 --kv-budget 65 --staged

date; echo "=== light level ==="
run16  jspace-light $LIGHT_BAND $LIGHT_K jspace
derive jspace-light $LIGHT_BAND $LIGHT_K jspace

date; echo "=== random control seed 1 ==="
run16  random-s1 $MED_BAND $MED_K random "--rand-seed 1"
derive random-s1 $MED_BAND $MED_K random "--rand-seed 1"

date; echo "=== out-of-band control (B=16384 only) ==="
$RUN --tag oob-B16384 --mode jspace --band $OOB_BAND --k $MED_K --budget 16384 \
    --samples 4 --problems solvable --datasets gsm8k,math,aime --batch 128 --kv-budget 65 --staged

date; echo "=== grid done ==="
