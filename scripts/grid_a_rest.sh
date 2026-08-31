#!/bin/bash
# Two-pod split, POD A (this pod): heavy + light + OOB (+ finish medium
# derives if the main grid wrapper was stopped mid-block). Run AFTER killing
# the original run_grid.sh wrapper. Pod B runs grid_b.sh (randoms + base).
set -euo pipefail
cd /workspace/jlens-cot
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN="python3 scripts/run_cell.py"
REUSE="python3 scripts/run_reuse.py"
exec >> results/grid.log 2>&1
trap 'echo "GRID-A FAILED at line $LINENO"; date' ERR
[ -f results/design.json ] || { echo "FATAL: no design.json"; exit 1; }

LIGHT_BAND="26:30";  LIGHT_K=10
MED_BAND="22:34";    MED_K=10
HEAVY_BAND="22:34";  HEAVY_K=100
OOB_BAND="4:16"

source scripts/grid_lib.sh   # run16 / derive / derive_one definitions

date; echo "=== (A) medium: ensure 16K + derives complete ==="
run16  jspace-med $MED_BAND $MED_K jspace
derive jspace-med $MED_BAND $MED_K jspace

date; echo "=== (A) heavy level ==="
run16  jspace-heavy $HEAVY_BAND $HEAVY_K jspace
derive jspace-heavy $HEAVY_BAND $HEAVY_K jspace

date; echo "=== (A) light level ==="
run16  jspace-light $LIGHT_BAND $LIGHT_K jspace
derive jspace-light $LIGHT_BAND $LIGHT_K jspace

date; echo "=== (A) out-of-band control ==="
$RUN --tag oob-B16384 --mode jspace --band $OOB_BAND --k $MED_K --budget 16384 \
    --samples 4 --problems solvable --datasets gsm8k,math,aime --batch 128 --kv-budget 65 --staged

date; echo "=== (A) done ==="
