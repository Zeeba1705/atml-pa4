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
```

Colab: `colab/task1.ipynb` (edit `CONDITIONS` in cell 0, Runtime → Run all; results go to Drive).
Pruning conditions also write `compression_layers.jsonl` (per-projection output error on calibration activations,
nonzero fraction, Wanda score quantiles).

## Task 2 — attention and KV memory

TODO

## Task 3 — distillation and speculative decoding

TODO

## Kaggle notebooks

TODO

## Attribution

- `task1/evaluate_hellaswag.py::preprocess` and the query/choice construction are copied from
  lm-evaluation-harness 0.4.13 (`lm_eval/tasks/hellaswag/utils.py`, task `hellaswag` v1.0), MIT licence.
- `task1/awq.py::awq_scales` follows the scale formula of the llm-awq reference implementation
  (`awq/quantize/auto_scale.py`: `x.pow(ratio).clamp(min=1e-4)`, normalised by `sqrt(max*min)`), MIT licence.
