#!/bin/bash
# Deadline-boxed final sequence (target: everything incl. figures by 17:30 UTC).
# Heavy de-powered per user sign-off: 16K on 25/bin subsample of bins 0-2
# (spec's bin floor); bins 3-4 censored a fortiori + evidenced at B=2048.
set -euo pipefail
cd /workspace/jlens-cot
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN="python3 scripts/run_cell.py"
REUSE="python3 scripts/run_reuse.py"
exec >> results/grid.log 2>&1
trap 'echo "FINAL-PUSH2 FAILED at line $LINENO"; date' ERR

MED_BAND="22:34"; MED_K=10
HEAVY_BAND="22:34"; HEAVY_K=100



date; echo "=== (F2) faithfulness ==="
python3 scripts/run_faithfulness.py --source jspace-med-B16384 --mode jspace --band $MED_BAND --k $MED_K --n 50 --batch 8

date; echo "=== (F2) adaptive budget points (bins 0-2, med + random-s0) ==="
for BIN in 0 1 2; do
  for BUD in 1024 4096; do
    $REUSE --source-tag jspace-med-B16384 --tag jspace-med-B$BUD-bin$BIN --budget $BUD --mode jspace --band $MED_BAND --k $MED_K --datasets gsm8k,math,aime --problems results/bin$BIN.json --batch 64 --kv-budget 55 --staged-source
    $REUSE --source-tag random-s0-B16384 --tag random-s0-B$BUD-bin$BIN --budget $BUD --mode random --band $MED_BAND --k $MED_K --rand-seed 0 --datasets gsm8k,math,aime --problems results/bin$BIN.json --batch 64 --kv-budget 55 --staged-source
  done
done

date; echo "=== (F2) heavy 16K, bins 0-2 subsample 25/bin ==="
$RUN --tag jspace-heavy-B16384 --mode jspace --band $HEAVY_BAND --k $HEAVY_K --budget 16384 --samples 4 --problems results/bins012_sub25.json --datasets gsm8k,math,aime --batch 128 --kv-budget 65 --staged

date; echo "=== (F2) heavy derived budgets (same subsample) ==="
for BUD in 2048 512 128; do
  DS=gsm8k,math; BATCH=96; [ $BUD -eq 2048 ] && { DS=gsm8k,math,aime; BATCH=48; }
  for pass in 1 2 3; do
    $REUSE --source-tag jspace-heavy-B16384 --tag jspace-heavy-B$BUD --budget $BUD --mode jspace --band $HEAVY_BAND --k $HEAVY_K --datasets $DS --problems results/bins012_sub25.json --batch $BATCH --kv-budget 55 --staged-source
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

date; echo "=== (F2) heavy B=2048 fresh, bins 3-4 (a-fortiori evidence) ==="
$RUN --tag jspace-heavy-B2048 --mode jspace --band $HEAVY_BAND --k $HEAVY_K --budget 2048 --samples 4 --problems results/bins34.json --datasets gsm8k,math,aime --batch 64 --kv-budget 55 --staged

date; echo "=== (F2) figures + tables ==="
python3 scripts/make_figures.py
python3 scripts/make_tables.py

date; echo "=== (F2) ALL DONE ==="
