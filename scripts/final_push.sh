#!/bin/bash
# Final single-pod push (deadline triage, 2026-08-31):
#   1. cheap report-critical items first: base sweep, faithfulness, adaptive
#   2. heavy restricted to bins 0-2 at B=16384 (frontier-location question;
#      bins 3-4 censored a fortiori) + derived budgets on the same bins
#   3. heavy-B2048 fresh on bins 3-4 (evidence for the a-fortiori censoring)
# Everything checkpointed; safe to re-run.
set -euo pipefail
cd /workspace/jlens-cot
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN="python3 scripts/run_cell.py"
REUSE="python3 scripts/run_reuse.py"
exec >> results/grid.log 2>&1
trap 'echo "FINAL-PUSH FAILED at line $LINENO"; date' ERR

MED_BAND="22:34"; MED_K=10
HEAVY_BAND="22:34"; HEAVY_K=100
source scripts/grid_lib.sh

date; echo "=== (F) base budget sweep (fresh, staged) ==="
$RUN --tag base-B2048 --mode none --budget 2048 --samples 4 --problems solvable --datasets gsm8k,math,aime --batch 128 --kv-budget 65 --staged
$RUN --tag base-B512  --mode none --budget 512  --samples 4 --problems solvable --datasets gsm8k,math --batch 128 --kv-budget 65 --staged
$RUN --tag base-B128  --mode none --budget 128  --samples 4 --problems solvable --datasets gsm8k,math --batch 128 --kv-budget 65 --staged

date; echo "=== (F) faithfulness (2 grid points, frac 0.5) ==="
python3 scripts/run_faithfulness.py --source clean --mode none --n 50 --sample-idx 4 --batch 32
python3 scripts/run_faithfulness.py --source jspace-med-B16384 --mode jspace --band $MED_BAND --k $MED_K --n 50 --batch 32

date; echo "=== (F) adaptive budget points (amendment 6; bins 0-2, med + random) ==="
for BIN in 0 1 2; do
  for BUD in 1024 4096; do
    $REUSE --source-tag jspace-med-B16384 --tag jspace-med-B$BUD-bin$BIN --budget $BUD --mode jspace --band $MED_BAND --k $MED_K --datasets gsm8k,math,aime --problems results/bin$BIN.json --batch 64 --kv-budget 55 --staged-source
    $REUSE --source-tag random-s0-B16384 --tag random-s0-B$BUD-bin$BIN --budget $BUD --mode random --band $MED_BAND --k $MED_K --rand-seed 0 --datasets gsm8k,math,aime --problems results/bin$BIN.json --batch 64 --kv-budget 55 --staged-source
  done
done

date; echo "=== (F) heavy 16K restricted to bins 0-2 ==="
$RUN --tag jspace-heavy-B16384 --mode jspace --band $HEAVY_BAND --k $HEAVY_K --budget 16384 --samples 4 --problems results/bins012.json --datasets gsm8k,math,aime --batch 128 --kv-budget 65 --staged

date; echo "=== (F) heavy derived budgets (bins 0-2) ==="
for BUD in 2048 512 128; do
  DS=gsm8k,math; BATCH=96; [ $BUD -eq 2048 ] && { DS=gsm8k,math,aime; BATCH=48; }
  for pass in 1 2 3; do
    $REUSE --source-tag jspace-heavy-B16384 --tag jspace-heavy-B$BUD --budget $BUD --mode jspace --band $HEAVY_BAND --k $HEAVY_K --datasets $DS --problems results/bins012.json --batch $BATCH --kv-budget 55 --staged-source
    more=0
    for s in 2 3; do
      f=results/runs/backfill-jspace-heavy-B$BUD-s$s.json
      if [ -s "$f" ] && [ "$(cat "$f")" != "[]" ]; then
        more=1
        $RUN --tag jspace-heavy-B$BUD --mode jspace --band $HEAVY_BAND --k $HEAVY_K --budget $BUD --samples 1 --sample-offset $s --problems $f --datasets $DS --batch 64 --kv-budget 55
      fi
    done
    if [ "$more" -eq 0 ]; then break; fi
  done
done

date; echo "=== (F) heavy B=2048 fresh on bins 3-4 (a-fortiori censoring evidence) ==="
$RUN --tag jspace-heavy-B2048 --mode jspace --band $HEAVY_BAND --k $HEAVY_K --budget 2048 --samples 4 --problems results/bins34.json --datasets gsm8k,math,aime --batch 64 --kv-budget 55 --staged

date; echo "=== (F) final push done ==="
