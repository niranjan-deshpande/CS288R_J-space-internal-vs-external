#!/bin/bash
# Two-pod split, POD B: random controls (both seeds) + base budget sweep.
# Fresh pod: run `bash setup.sh` first (~15 min), then this script.
set -euo pipefail
cd /workspace/jlens-cot
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN="python3 scripts/run_cell.py"
REUSE="python3 scripts/run_reuse.py"
exec >> results/grid_b.log 2>&1
trap 'echo "GRID-B FAILED at line $LINENO"; date' ERR
[ -f results/design.json ] || { echo "FATAL: no design.json"; exit 1; }

MED_BAND="22:34"; MED_K=10

source scripts/grid_lib.sh

date; echo "=== (B) random control seed 0 ==="
run16  random-s0 $MED_BAND $MED_K random "--rand-seed 0"
derive random-s0 $MED_BAND $MED_K random "--rand-seed 0"

date; echo "=== (B) base budget sweep (fresh) ==="
$RUN --tag base-B2048 --mode none --budget 2048 --samples 4 --problems solvable --datasets gsm8k,math,aime --batch 128 --kv-budget 65 --staged
$RUN --tag base-B512  --mode none --budget 512  --samples 4 --problems solvable --datasets gsm8k,math --batch 128 --kv-budget 65 --staged
$RUN --tag base-B128  --mode none --budget 128  --samples 4 --problems solvable --datasets gsm8k,math --batch 128 --kv-budget 65 --staged

date; echo "=== (B) random control seed 1 ==="
run16  random-s1 $MED_BAND $MED_K random "--rand-seed 1"
derive random-s1 $MED_BAND $MED_K random "--rand-seed 1"

date; echo "=== (B) done ==="
