#!/bin/bash
#kaggle task 1 p3/q3 pipeline on two T4s, called by kaggle/task1.ipynb; EVALS comes from the notebook
cd /tmp/pa4/atml-pa4
RES=/kaggle/working/results/task1
SPL=/kaggle/working/task1_splits
mkdir -p "$RES/logs" "$SPL"
rm -rf splits/task1 && ln -s "$SPL" splits/task1
COMMON="--data_root /tmp/pa4/atml-pa4-datasets --output_dir /tmp/pa4/outputs"

run() {
  tag=$1; shift
  PYTHONUNBUFFERED=1 python -m "$@" $COMMON 2>&1 | grep --line-buffered -v -i "warn" | tee -a "$RES/logs/$tag.log" | sed -u "s/^/[$tag] /"
  return ${PIPESTATUS[0]}
}

#calibration-only sensitivity, then q3 allocations (cpu ilp): all frozen before any held-out eval
CUDA_VISIBLE_DEVICES=0 run adaptive task1.sensitivity || exit 1
run adaptive task1.allocate --method q3 --bits 4 --results_dir "$RES" || exit 1
run adaptive task1.allocate --method q3 --bits 3 --results_dir "$RES" || exit 1

chain() {
  gpu=$1; s=$2; b=$3
  export CUDA_VISIBLE_DEVICES=$gpu
  run gpu$gpu task1.allocate --method p3 --sparsity $s --results_dir "$RES" &&
  run gpu$gpu task1.run_condition --method p3_nonuniform --sparsity $s --evals $EVALS --results_dir "$RES" &&
  run gpu$gpu task1.run_condition --method q3_mixed_budget --bits $b --evals $EVALS --results_dir "$RES"
}

chain 0 0.5 4 & A=$!
chain 1 0.75 3 & B=$!
wait $A; ra=$?
wait $B; rb=$?
echo "gpu0 chain exit $ra, gpu1 chain exit $rb"
ls -R "$SPL"