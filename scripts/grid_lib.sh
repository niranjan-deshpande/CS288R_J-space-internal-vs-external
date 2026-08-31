# Shared cell helpers for grid scripts (sourced; expects RUN/REUSE set).
run16 () {  # $1 tag-prefix  $2 band  $3 k  $4 mode  $5 extra
  $RUN --tag $1-B16384 --mode $4 --band $2 --k $3 --budget 16384 --samples 4 \
    --problems solvable --datasets gsm8k,math,aime --batch 128 --kv-budget 65 \
    --staged ${5:-}
}
derive_one () {  # $1 tag-prefix $2 band $3 k $4 mode $5 budget $6 datasets $7 batch $8 extra
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
  derive_one $1 $2 $3 $4 2048 gsm8k,math,aime 48 "${5:-}"
  derive_one $1 $2 $3 $4 512  gsm8k,math 96 "${5:-}"
  derive_one $1 $2 $3 $4 128  gsm8k,math 96 "${5:-}"
}
