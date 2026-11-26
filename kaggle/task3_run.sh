#!/bin/bash
#kaggle task 3 on two T4s. STAGE (from the notebook):
#  1: gpu 0 = real-model checks, step costs, baseline spec eval with the original draft;
#     gpu 1 = generated prefix banks (teacher_pure, exact_spec_original_gamma4), then the off-policy teacher pool
#every script skips finished files, so rerunning a stage continues it
cd /tmp/pa4/atml-pa4
RES=/kaggle/working/results
OUT=/kaggle/working/outputs
STAGE=${STAGE:-1}
mkdir -p "$RES/task3/logs"
test -z "$(git status --porcelain)" || { echo "code tree is dirty:"; git status --porcelain; exit 1; }
COMMON="--data_root /tmp/pa4/atml-pa4-datasets --output_dir $OUT --results_dir $RES --device cuda"

run() {
  tag=$1; gpu=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 python -m "$@" $COMMON 2>&1 \
    | grep --line-buffered -v -i "warn\|cudart_stub" | tee -a "$RES/task3/logs/$tag.log" | sed -u "s/^/[$tag] /"
  rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] || echo "[$tag] FAILED: $*" | tee -a "$RES/task3/logs/failed.log"
  return $rc
}

if [ "$STAGE" = "1" ]; then
  run gpu0 0 task3.spec_eval --draft_name original &
  A=$!
  ( run gpu1 1 task3.prefix_banks --bank teacher_pure
    run gpu1 1 task3.prefix_banks --bank exact_spec_original_gamma4
    run gpu1 1 task3.teacher_pool ) &
  B=$!
  wait $A; wait $B
fi

echo "stage $STAGE finished"
ls -R "$OUT/task3" | head -40
cat "$RES/task3/logs/failed.log" 2>/dev/null
