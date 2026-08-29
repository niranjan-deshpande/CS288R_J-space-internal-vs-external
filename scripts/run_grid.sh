#!/bin/bash
# Master condition grid. Levels use band/k set after calibration pilots
# (see results/calibration.md). Every cell is checkpointed+resumable, so this
# script can be re-run after interruption.
set -uo pipefail
cd /workspace/jlens-cot
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN="python3 scripts/run_cell.py"
LOG=results/grid.log
exec >> $LOG 2>&1

# ---- ablation levels (FILLED AFTER CALIBRATION) ----
LIGHT_BAND="24:32";  LIGHT_K=10
MED_BAND="22:34";    MED_K=25
HEAVY_BAND="22:34";  HEAVY_K=50
OOB_BAND="4:12"      # out-of-band control (matched width to medium)

date; echo "=== base budget sweep (solvable) ==="
$RUN --tag base-B0    --mode none --budget 0    --samples 4 --problems solvable --datasets gsm8k,math --batch 64
$RUN --tag base-B128  --mode none --budget 128  --samples 4 --problems solvable --datasets gsm8k,math --batch 48
$RUN --tag base-B512  --mode none --budget 512  --samples 4 --problems solvable --datasets gsm8k,math --batch 32
$RUN --tag base-B2048 --mode none --budget 2048 --samples 4 --problems solvable --datasets gsm8k,math,aime --batch 24

run_level () {  # $1 tag-prefix  $2 band  $3 k  $4 mode  $5 extra
  local tag=$1 band=$2 k=$3 mode=$4 extra=${5:-}
  $RUN --tag $tag-B0     --mode $mode --band $band --k $k --budget 0     --samples 4 --problems solvable --datasets gsm8k,math --batch 48 $extra
  $RUN --tag $tag-B128   --mode $mode --band $band --k $k --budget 128   --samples 4 --problems solvable --datasets gsm8k,math --batch 32 $extra
  $RUN --tag $tag-B512   --mode $mode --band $band --k $k --budget 512   --samples 4 --problems solvable --datasets gsm8k,math --batch 24 $extra
  $RUN --tag $tag-B2048  --mode $mode --band $band --k $k --budget 2048  --samples 4 --problems solvable --datasets gsm8k,math,aime --batch 16 $extra
  $RUN --tag $tag-B16384 --mode $mode --band $band --k $k --budget 16384 --samples 4 --problems solvable --datasets gsm8k,math,aime --batch 12 --kv-budget 9 $extra
}

date; echo "=== jspace levels ==="
run_level jspace-light $LIGHT_BAND $LIGHT_K jspace
run_level jspace-med   $MED_BAND   $MED_K   jspace
run_level jspace-heavy $HEAVY_BAND $HEAVY_K jspace

date; echo "=== random control (2 seeds, medium level) ==="
run_level random-s0 $MED_BAND $MED_K random "--rand-seed 0"
run_level random-s1 $MED_BAND $MED_K random "--rand-seed 1"

date; echo "=== out-of-band control (B=16384, medium strength) ==="
$RUN --tag oob-B16384 --mode jspace --band $OOB_BAND --k $MED_K --budget 16384 \
    --samples 4 --problems solvable --datasets gsm8k,math,aime --batch 12 --kv-budget 9

date; echo "=== selectivity: MMLU + SST-2, direct answering ==="
$RUN --tag select-clean --mode none --budget 0 --samples 1 --problems all --datasets mmlu,sst2 --batch 64
$RUN --tag select-med   --mode jspace --band $MED_BAND --k $MED_K --budget 0 --samples 1 --problems all --datasets mmlu,sst2 --batch 48

date; echo "=== grid done ==="
