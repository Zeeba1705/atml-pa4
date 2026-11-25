# ATML PA4 — compression, attention/KV memory, speculative decoding

Code for EE-5102/CS-6304 Programming Assignment 4 (Fall 2026). Interpretation choices are in
[docs/decisions.md](docs/decisions.md).

## Environment

- `transformers==4.57.1`, `huggingface_hub==0.36.2` (pinned by the handout); other versions in `requirements.txt`.
- GPU runs on Kaggle (T4 ×2, FP16). Everything has a `--smoke` CPU path.

```bash
pip install -r requirements.txt
python -m pytest tests -q
```

## Data setup

```bash
git clone <course data repo> atml-pa4-datasets
cd atml-pa4-datasets
pip install -r requirements-public.txt
python scripts/setup_data.py --public          # CPU, C4 streaming is slow
python scripts/setup_data.py --prefix-banks    # GPU
python scripts/check_setup.py
cp data/local_source_revisions.json ../splits/data_lock/
```

## Task 1 — compression

Data paths in `configs/task1/base.yaml` are relative to `--data_root` (default `atml-pa4-datasets`).
Each (condition × eval) writes `results/task1/<condition>/<eval>.json` (+ `<eval>_examples.jsonl`) and is skipped
if that file exists. Evals: `wikitext2`, `c4_val`, `hellaswag`, `resources`.

```bash
# cpu smoke: tiny random qwen2, real tokenizer and data, writes to outputs/smoke/
python -m task1.run_condition --method baseline --smoke

# gpu baseline (one T4)
CUDA_VISIBLE_DEVICES=0 python -m task1.run_condition --method baseline

# pruning: P1 magnitude / P2 Wanda at s in {0.25, 0.5, 0.75}; 2:4 Wanda at 0.5
CUDA_VISIBLE_DEVICES=0 python -m task1.run_condition --method p1_magnitude --sparsity 0.5
CUDA_VISIBLE_DEVICES=0 python -m task1.run_condition --method p2_wanda --sparsity 0.5
CUDA_VISIBLE_DEVICES=0 python -m task1.run_condition --method p2_wanda_2of4 --sparsity 0.5

# quantization: Q1 uniform / Q2 AWQ-style at b in {8, 4, 3}, group 128 (fake-quant FP16, packed bytes reported)
CUDA_VISIBLE_DEVICES=0 python -m task1.run_condition --method q1_uniform --bits 4
CUDA_VISIBLE_DEVICES=0 python -m task1.run_condition --method q2_awq --bits 4

# p3/q3 (decision 1.7): gradient pass on calibration data, then freeze allocations BEFORE any held-out eval
CUDA_VISIBLE_DEVICES=0 python -m task1.sensitivity                          # -> splits/task1/sensitivity.json
CUDA_VISIBLE_DEVICES=0 python -m task1.allocate --method p3 --sparsity 0.5  # lambda sweep on c4 dev -> p3_allocation_s0.5.json
CUDA_VISIBLE_DEVICES=0 python -m task1.allocate --method p3 --sparsity 0.75
python -m task1.allocate --method q3 --bits 4                               # cpu, exact ilp under the q2-int4 budget
python -m task1.allocate --method q3 --bits 3
CUDA_VISIBLE_DEVICES=0 python -m task1.run_condition --method p3_nonuniform --sparsity 0.5
CUDA_VISIBLE_DEVICES=0 python -m task1.run_condition --method q3_mixed_budget --bits 4

# smoke for the same chain: add --smoke to every command (frozen files then go to outputs/smoke/task1/splits)
```

Colab: `colab/task1.ipynb` (edit `CONDITIONS` in cell 0, Runtime → Run all; results go to Drive).
Pruning conditions also write `compression_layers.jsonl` (per-projection output error on calibration activations,
nonzero fraction, Wanda score quantiles).

## Task 2 — attention and KV memory

### Part A — attention operators (configs/task2/attention.yaml)

```bash
# dense / tiled online-softmax / sdpa (default + each forced backend), B=2 H=4 d=64, L 256..8192, fp16 + fp32;
# one file per (dtype, L) in results/task2/attention/, plus edge_cases.json; finished files are skipped
python -m task2.bench_attention --device cuda
python -m task2.bench_attention --smoke     # cpu, L in {64, 128}
```

### Part B — MQAR (configs/task2/mqar.yaml)

```bash
# checks the frozen dev/test files against the supplied generator, builds the 6 training streams once
# (shared by all seeds and mechanisms), asserts no train/dev/test overlap -> splits/task2/mqar_data_check.json
python -m task2.mqar_data --workers 4

# one run = one mechanism x one seed (names in configs/task2/mqar.yaml `runs`), 4,500 updates, final checkpoint tested;
# dense runs also write the test-time W=64 mask control. skips finished runs, resumes from outputs/task2/mqar/<run>/last.pt
python -m task2.train_mqar --run_name dense_K16_mixed --seed 6304 --device cuda

# re-evaluate a final checkpoint on test, optionally under another mask
python -m task2.eval_mqar --checkpoint outputs/task2/mqar/dense_K16_mixed_s6304/final.pt --out_dir results/task2/mqar/dense_K16_mixed_s6304 --attention window

# cpu smoke (6 updates, 2 examples per cell, writes under outputs/smoke/)
python -m task2.mqar_data --smoke && python -m task2.train_mqar --smoke --run_name linear_K8_mixed --seed 6304
```

Kaggle: `kaggle/task2.ipynb` with `PARTS= "A"` runs the Part A benchmark alone on GPU 0; `PARTS= "B"` runs all 39 B1/B2 runs (13 mechanisms × seeds 6304/6305/6306) on both T4s via `kaggle/task2_run.sh`.
Outputs per run in `results/task2/mqar/<run>_s<seed>/`: `config.json`, `train_log.jsonl`, `<run>_history.json` (dev curves),
`test_metrics.json` (+ `test_window_mask_metrics.json` for dense), `test_predictions.jsonl.gz`.

## Task 3 — distillation and speculative decoding

TODO

## Kaggle notebooks

TODO

## Attribution

- `task1/evaluate_hellaswag.py::preprocess` and the query/choice construction are copied from
  lm-evaluation-harness 0.4.13 (`lm_eval/tasks/hellaswag/utils.py`, task `hellaswag` v1.0), MIT licence.
- `task1/awq.py::awq_scales` follows the scale formula of the llm-awq reference implementation
  (`awq/quantize/auto_scale.py`: `x.pow(ratio).clamp(min=1e-4)`, normalised by `sqrt(max*min)`), MIT licence.
