#!/bin/bash
#kaggle task 2 on two T4s. PARTS (from the notebook) picks what runs, default B:
#  A: attention-operator benchmark on gpu 0, run alone first so nothing else shares the gpu/cpu during timing
#  B: mqar sweep: data checks + streams (cpu), then NW workers (two per gpu) over all runs x seeds;
#     every (run, seed) skips if finished and resumes from its last checkpoint
#  C: kv cache part C: quality run (180 cases x 6 caches) on gpu 0 while the mqar forward-only sweep runs on gpu 1,
#     then the packed encode/decode benchmark alone on gpu 0. finished cases / lengths are skipped on rerun
PARTS=${PARTS:-B}
cd /tmp/pa4/atml-pa4
RES=/kaggle/working/results
OUT=/kaggle/working/outputs
SPL=/kaggle/working/task2_splits
NW=${NW:-4}
mkdir -p "$RES/task2/logs" "$SPL"
test -z "$(git status --porcelain)" || { echo "code tree is dirty:"; git status --porcelain; exit 1; }

if [[ "$PARTS" == *A* ]]; then
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python -m task2.bench_attention --results_dir "$RES" --output_dir "$OUT" --device cuda 2>&1 \
    | grep --line-buffered -v -i "warn\|cudart_stub" | tee -a "$RES/task2/logs/attention.log" | sed -u "s/^/[A] /"
  [ "${PIPESTATUS[0]}" -eq 0 ] || echo "[A] FAILED attention benchmark" | tee -a "$RES/task2/logs/failed.log"
fi

if [[ "$PARTS" == *C* ]]; then
  krun() {
    tag=$1; gpu=$2; shift 2
    CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 python -m "$@" --results_dir "$RES" --output_dir "$OUT" --device cuda 2>&1 \
      | grep --line-buffered -v -i "warn\|cudart_stub" | tee -a "$RES/task2/logs/$tag.log" | sed -u "s/^/[$tag] /"
    [ "${PIPESTATUS[0]}" -eq 0 ] || echo "[$tag] FAILED: $*" | tee -a "$RES/task2/logs/failed.log"
  }
  krun kv_eval 0 task2.kv_eval --data_root /tmp/pa4/atml-pa4-datasets &
  A=$!
  krun mqar_inference 1 task2.bench_mqar_inference &
  B=$!
  wait $A; wait $B
  krun kv_bench 0 task2.kv_bench --data_root /tmp/pa4/atml-pa4-datasets
fi

if [[ "$PARTS" != *B* ]]; then
  echo "parts $PARTS finished"
  if [ -f "$RES/task2/logs/failed.log" ]; then cat "$RES/task2/logs/failed.log"; exit 1; fi
  exit 0
fi

PYTHONUNBUFFERED=1 python -m task2.mqar_data --workers "$(nproc)" --output_dir "$OUT" \
  --check_path "$SPL/mqar_data_check.json" --data_root /tmp/pa4/atml-pa4-datasets 2>&1 | tee "$RES/task2/logs/data.log" || exit 1
test -f "$SPL/mqar_data_check.json" || { echo "data check failed"; exit 1; }

#seed-major order: every mechanism's first seed finishes before any second seed starts
JOBS=$(python -c "
from common.io_utils import load_config
c= load_config('configs/task2/mqar.yaml')
for s in c['seeds']:
    for r in c['runs']:
        print(r['name'], s)
")
echo "$(echo "$JOBS" | wc -l) jobs on $NW workers"

worker() {
  w=$1
  i=0
  echo "$JOBS" | while read -r name seed; do
    if [ $((i % NW)) -eq "$w" ]; then
      CUDA_VISIBLE_DEVICES=$((w % 2)) PYTHONUNBUFFERED=1 python -m task2.train_mqar --run_name "$name" --seed "$seed" \
        --output_dir "$OUT" --results_dir "$RES" --data_root /tmp/pa4/atml-pa4-datasets --device cuda 2>&1 \
        | grep --line-buffered -v -i "warn\|cudart_stub" | tee -a "$RES/task2/logs/worker$w.log" | sed -u "s/^/[w$w] /"
      [ "${PIPESTATUS[0]}" -eq 0 ] || echo "[w$w] FAILED $name $seed" | tee -a "$RES/task2/logs/failed.log"
    fi
    i=$((i + 1))
  done
}

for w in $(seq 0 $((NW - 1))); do worker "$w" & done
wait
echo "sweep finished"
#training streams are rebuilt deterministically in minutes; drop them so the output download stays small
rm -rf "$OUT/task2/mqar_streams"
python -m kaggle.task2_summary --results_dir "$RES"
