# PA4 results guide (numbers, locations, definitions, caveats — no interpretation)

Lookup sheet for writing the report. It lists where every result lives, what each metric means, the headline
numbers, the honest labels that must accompany them, and which differences are statistically resolved. It
deliberately contains **no analysis or explanation of why** — that is the report's job.

All numbers were copied from the committed CSVs on 2026-11-27. If a CSV is regenerated, it wins over this file.

Branches: Task 1 = `task1-compress`, Task 2 = `task2-attention`, Task 3 = `task3-specdec`. Each task's tables are
regenerated on CPU with `python -m task{1,2,3}.make_tables`, and decisions are in `docs/decisions.md` on each branch.

Bootstrap convention, used by all tasks (`common/bootstrap.py`): paired, 2,000 resamples, seed 6304, 95% percentile
interval. "Resolved" below means the 95% CI excludes 0. These CIs are **not** corrected for multiple comparisons,
except where a Bonferroni column is given.

---

## Task 1 — compression of Qwen2.5-1.5B (base), FP16, no training

### Where
| What | File (branch `task1-compress`) |
|---|---|
| Per-condition raw | `results/task1/<condition>/{wikitext2,c4_val,hellaswag,resources,condition}.json`, `*_examples.jsonl`, `compression_layers.jsonl` |
| Main table (one row per condition) | `results/task1/tables/main.csv`; short version `results/task1/summary.csv` |
| Per-projection error / nonzeros / bits / AWQ α | `results/task1/tables/layer_errors.csv` |
| Sensitivity + final sparsity/bit maps | `results/task1/tables/sensitivity_and_allocation_maps.csv`, `allocation_meta.csv` |
| P3 λ sweep on C4 dev (selection evidence) | `results/task1/tables/p3_dev_sweep.csv` |
| Paired bootstraps | `results/task1/tables/paired_bootstrap.csv` |
| Qualitative candidates (fixed rule) | `results/task1/tables/qualitative_candidates.csv` |
| Figures | `results/task1/figures/{quality_vs_sparsity,quality_vs_bytes,calibration_error_vs_ppl,sensitivity_and_sparsity_maps,q3_bit_maps}.png` |
| Frozen allocations (pre-held-out) | `splits/task1/{sensitivity,p3_allocation_s0.5,p3_allocation_s0.75,q3_allocation_b4,q3_allocation_b3}.json`, `splits/task1/p3_dev/` |

### Definitions
- **PPL**: exp(Σ NLL over `score_target_offsets` / N). Each window is independent, and N = 50,000 per suite in every
  condition (asserted). WikiText-2: 437 windows. C4-val: 177 windows.
- **HellaSwag**: zero-shot, summed log-likelihood per ending. `acc` = argmax raw LL. `acc_norm` = argmax LL /
  **characters** of the ending (lm-eval v1.0). The full 2,000 examples are used for the 10 conditions in decision 1.1;
  all others use the first 512. `hs512_*` in `main.csv` is the first-512 subset for **every** condition, so all 18
  rows are comparable on it.
- **Nonzero fraction**: nonzeros / eligible weights (q, k, v, o, gate, up, down; 1,310,195,712 weights). For
  quantized conditions it counts weights that round to exactly 0. It is not a sparsity target.
- **Packed bytes** (`packed_total_bytes`): serialized size of the b-bit codes, the FP16 scale per 128-group, packed
  b-bit zero-points and padding, plus any unfolded AWQ FP16 scales, plus all non-eligible params in FP16, with tied
  weights counted once.
- **16-bit sparse estimate**: 1 mask bit per eligible weight + 16 bits per nonzero, byte-aligned, plus non-eligible
  FP16. Dense FP16 is 3,087,428,608 bytes.
- **Layer-output error**: per projection ‖WX − ŴX‖² / ‖WX‖² on the 128 calibration sequences, collected sequentially
  (decision 1.5). `layer_rel_err_mean` is the unweighted mean over 196 projections.
- **Latency**: batch 1, 2 warm-ups + 5 reps, median (IQR in the CSV). Prefill at 512 and 2,048 tokens; 64-token decode
  after a 512-token prompt (decision 1.9).

### Headline numbers (`main.csv`)
| condition | WT2 PPL | C4 PPL | HS n | HS acc_norm | HS512 acc_norm | nonzero frac | packed bytes | prefill512 ms | decode ms/tok |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 16.107 | 14.962 | 2000 | 0.694 | 0.582 | 1.000 | (dense 3,087,428,608) | 101.3 | 30.4 |
| p1_magnitude_s0.25 | 21.33 | 17.99 | 512 | 0.588 | 0.588 | 0.75 | – | 99.0 | 31.2 |
| p1_magnitude_s0.5 | 127.80 | 102.46 | 2000 | 0.3985 | 0.414 | 0.50 | – | 99.0 | 32.6 |
| p1_magnitude_s0.75 | 1,579,373 | 1,221,827 | 512 | 0.285 | 0.285 | 0.25 | – | 99.0 | 30.1 |
| p2_wanda_s0.25 | 16.54 | 15.39 | 512 | 0.596 | 0.596 | 0.75 | – | 102.8 | 30.8 |
| p2_wanda_s0.5 | 25.41 | 21.11 | 2000 | 0.5775 | 0.533 | 0.50 | – | 98.4 | 30.6 |
| p2_wanda_s0.75 | 2811.0 | 1772.9 | 512 | 0.293 | 0.293 | 0.25 | – | 100.8 | 31.2 |
| p2_wanda_2of4_s0.5 | 66.79 | 50.90 | 512 | 0.436 | 0.436 | 0.50 | – | 98.6 | 31.9 |
| p3_nonuniform_s0.5 | 23.63 | 20.19 | 2000 | 0.584 | 0.527 | 0.50 | – | 124.6* | 39.4* |
| p3_nonuniform_s0.75 | 3172.6 | 1113.9 | 512 | 0.258 | 0.258 | 0.25 | – | 112.1* | 40.0* |
| q1_uniform_b8 | 16.085 | 14.968 | 512 | 0.582 | 0.582 | 0.991 | 1,807,940,608 | 99.4 | 30.1 |
| q1_uniform_b4 | 18.90 | 16.73 | 2000 | 0.670 | 0.578 | 0.850 | 1,147,724,800 | 99.9 | 31.2 |
| q1_uniform_b3 | 40.53 | 30.57 | 2000 | 0.5615 | 0.508 | 0.689 | 982,670,848 | 99.2 | 30.7 |
| q2_awq_b8 | 16.119 | 14.970 | 512 | 0.582 | 0.582 | 0.990 | 1,808,026,624 | 105.3 | 31.2 |
| q2_awq_b4 | 17.63 | 15.96 | 2000 | 0.6785 | 0.586 | 0.842 | 1,147,810,816 | 102.0 | 32.8 |
| q2_awq_b3 | 25.22 | 20.98 | 2000 | 0.6165 | 0.539 | 0.675 | 982,756,864 | 100.4 | 31.0 |
| q3_mixed_budget_b4 | 17.61 | 16.00 | 2000 | 0.6715 | 0.578 | 0.822 | 1,147,810,816 | 130.0* | 39.5* |
| q3_mixed_budget_b3 | 23.97 | 20.28 | 2000 | 0.612 | 0.535 | 0.661 | 982,756,864 | 121.5* | 40.7* |

\* Different software environment; see the caveats.

Other facts:
- Q2 packed bytes minus Q1 packed bytes = 86,016, which is the unfolded o_proj FP16 scales (decision 1.4).
- Q3 packed bytes equal the Q2 budget exactly at both b4 and b3.
- Q3 bit counts: b4 budget {3: 18, 4: 142, 8: 36}; b3 budget {2: 10, 3: 117, 4: 67, 8: 2} (196 projections).
- Q3 ILP objective (summed Fisher sensitivity), from `allocation_meta.csv`: b4 0.0507 vs 0.0704 uniform; b3 0.2018 vs
  0.2619 uniform.
- P3 chosen λ = 0.15 at both 50% and 75%. Dev PPL for λ ∈ {0, 0.05, 0.1, 0.15, 0.2}:
  - 50%: 17.09 / 16.61 / 16.36 / 16.27 / 16.31
  - 75%: 1461 / 1128 / 815 / 803 / 1187
- P3 zero count equals P2 exactly: 655,097,856 at 50% (`matched_zero_check` in `condition.json`).

### Paired bootstrap (`paired_bootstrap.csv`; diff = first − second)
| comparison | WT2 PPL diff [CI] | C4 PPL diff [CI] | HS acc_norm diff [CI] (n) |
|---|---|---|---|
| P3 − P2, 50% | −1.78 [−1.94, −1.61] | −0.92 [−1.09, −0.78] | +0.0065 [−0.004, 0.016] (2000) |
| P3 − P2, 75% | +361.6 [187.1, 586.3] | −659.0 [−814.3, −515.3] | −0.035 [−0.067, −0.002] (512) |
| Q3 − Q2, b4 | −0.02 [−0.12, 0.08] | +0.04 [−0.03, 0.11] | −0.007 [−0.016, 0.002] (2000) |
| Q3 − Q2, b3 | −1.25 [−1.56, −0.96] | −0.69 [−1.00, −0.46] | −0.0045 [−0.018, 0.0085] (2000) |
| P2 − P1, 50% (control) | −102.4 [−113.0, −93.8] | −81.3 [−90.6, −73.0] | +0.179 [0.159, 0.2005] (2000) |
| Q2 − Q1, b4 (control) | −1.26 [−1.51, −1.06] | −0.77 [−0.90, −0.64] | +0.0085 [−0.0035, 0.0195] (2000) |
| Q2 − Q1, b3 (control) | −15.3 [−16.8, −14.0] | −9.6 [−10.9, −8.5] | +0.055 [0.0385, 0.071] (2000) |
| 2:4 − unstructured, 50% | +41.4 [37.6, 45.6] | +29.8 [26.5, 33.5] | −0.098 [−0.131, −0.065] (512) |

Raw `acc` rows are in the same file. Q3 − Q2 b4 raw acc is −0.007 [−0.0135, −0.001], whose CI excludes 0, while
acc_norm's does not.

PPL bootstrap: resample windows with replacement, pool the NLL sums and token counts for both conditions on the
same resample, then take exp. The CI is on the PPL difference.

### Caveats and labels to carry into the report
- **Quantized = fake-quantized FP16** (dequantized weights in memory). Packed bytes are a serialized count, not a
  low-bit kernel, so quantized latency is FP16 latency.
- **Pruned = zeros in dense FP16.** There is no sparse kernel and no 2:4 tensor-core path (handout p.4), so pruned
  latency is dense latency. The 16-bit sparse figure is an estimate.
- **Software environments differ.** Baseline, P1, P2, 2:4, Q1 and Q2 ran with torch 2.14.1; P3 and Q3 ran in the
  2026-11-25 Kaggle session with torch 2.11.0 (commit `0194164…-dirty`, decision 1.12). All ran on a Tesla T4.
  - The P3/Q3 latency (*) differences against the other rows therefore mix method and environment.
  - The P3−P2 and Q3−Q2 quality comparisons also cross this environment boundary. No same-environment baseline rerun
    exists to size the numeric effect.
- Commit hashes of the pre-rewrite runs are mapped in decision 1.13.
- WikiText-2 is scored per HF row as a separate document (decision 1.10), so its PPL is not comparable to published
  concatenated-stream numbers.
- Q2 is an "AWQ-style scale search" (decision 1.11), not the llm-awq engine. There is no auto-clip (decision 1.3), and
  o_proj scales are not folded under GQA (decision 1.4).
- The P3/Q3 sensitivity measures and allocation rules were your choices (decision 1.7). Allocations were frozen
  (sha256 recorded) before any held-out eval; λ was chosen on the 32 C4 dev sequences only.
- The 75% rows have PPL in the thousands to millions. The PPL CIs there are wide and very sensitive to a few windows.
- The qualitative case-finder rule was fixed in code before any results were looked at:
  - HellaSwag: baseline right and condition wrong (acc_norm), ranked by the drop in normalized gold margin.
  - WikiText-2: windows with ≥ 32 scored tokens, ranked by the increase in per-token NLL.
  - Top 5 of each, per full-2,000 condition.

---

## Task 2 — attention and KV memory

### Where (branch `task2-attention`)
| What | File |
|---|---|
| Part A raw | `results/task2/attention/{float16,float32}_L*.json`, `edge_cases.json` |
| Part A tables | `results/task2/tables/attention.csv`, `attention_sdpa_backends.csv`, `attention_tile_sizes.csv`, `attention_edge_cases.csv` |
| Part B raw | `results/task2/mqar/<run>_s<seed>/` (test_metrics, test_window_mask_metrics for dense, history, predictions) |
| Part B tables | `mqar_test_per_seed.csv`, `mqar_test_seed_mean.csv`, `mqar_distance_bins.csv`, `mqar_dev_curves.csv`, `mqar_theoretical_costs.csv`, `mqar_inference_sweep.csv`, `mqar_b3_vs_linear_bootstrap.csv` |
| Part C raw | `results/task2/kv/cases/<id>.json` (180), `kv/summary.json`, `kv_bench/len*_b*.json` |
| Part C tables | `kv_main.csv`, `kv_breakdowns.csv`, `kv_bootstrap.csv`, `kv_lengths.csv`, `kv_c1_checks.csv`, `kv_bench.csv`, `kv_fixed_case_generations.csv` |
| Figures | `results/task2/figures/{attention_latency,attention_memory,mqar_learning_curves,mqar_distance_by_load,mqar_inference_sweep,kv_accuracy_by_position,kv_selection_masks}.png` |
| Summary | `results/task2/summary.csv` |

### Part A — operators (B=2, H=4, d=64, Tesla T4, torch 2.11)
- **Errors**: vs the dense FP32 reference, max abs. FP16 output errors are about 1e-3 for dense, tiled and SDPA at
  every L (e.g. L=8192: dense 1.5e-3, tiled 1.0e-3, SDPA 1.0e-3); dq up to 2.7e-3. FP32 errors are about 1e-6.
- **Status**: every forward ran. The only non-OK entry is dense FP32 forward+backward at L=8192 (**OOM**). Its
  forward ran (168 ms).
- **Latency at L=8192, forward / forward+backward (ms)**:
  - FP16: dense 168.7 / 543.3, tiled 934.3 / 3378.6, SDPA 4.47 / 24.7.
  - FP32: dense 168.1 / OOM, tiled 842.7 / 3143.5, SDPA 24.6 / 102.8.
- **Peak allocated at L=8192, FP16, forward (MiB)**: dense 6333, tiled 141.5, SDPA 132.2. Forward+backward: dense
  14,557, tiled 3343, SDPA 196.8.
- **SDPA backends**: the default dispatch is EFFICIENT_ATTENTION at every (dtype, L). Forced FLASH_ATTENTION and
  CUDNN_ATTENTION are **unsupported** on the T4 (compute capability 7.5); EFFICIENT and MATH run.
- **Edge cases** (`attention_edge_cases.csv`):
  - Causality: the max change before the perturbed position is 0.0 for all three implementations.
  - All-masked row: dense outputs a zero row and SDPA's output is finite, with example values all 0.0.
  - Extreme scores (q×300): every output is finite. FP16 dense/tiled max error vs FP64 is 0.42, SDPA's is 0.0019.
- **Tile sizes** 32/64/128/256 give identical max error (`attention_tile_sizes.csv`).
- Labels: "tiled" is a Python reference with an autograd backward that stores tiles. It is **not FlashAttention**.

### Part B — MQAR (test set; 1,024 examples per (K, regime) cell; seeds 6304/6305/6306)
Chance: 1/64 = 0.0156 (label space) and 1/K (0.25, 0.125, 0.0625).

Test accuracy, mean over 3 seeds (`mqar_test_seed_mean.csv`, "trained mask" rows), on the trained cell and the K16 cells:

| run | trained-cell acc | K16 near | K16 mixed | K16 far |
|---|---|---|---|---|
| dense_K4_mixed | 1.000 | 0.338 | 0.335 | 0.334 |
| dense_K8_mixed | 1.000 | 0.520 | 0.521 | 0.520 |
| dense_K16_mixed | 1.000 | 1.000 | 1.000 | 1.000 |
| dense_K16_near | 1.000 | 1.000 | 0.9995 | 0.9992 |
| window_K4_near | 1.000 (K4 near) | 0.353 | 0.075 | 0.0155 |
| window_K8_near | 1.000 (K8 near) | 0.552 | 0.114 | 0.0153 |
| window_K16_near | 1.000 | 1.000 | 0.224 | 0.015 |
| window_K16_mixed | 0.042 | 0.069 | 0.042 | 0.014 |
| dilated_K4_mixed | 0.4165 (K4 mixed) | 0.115 | 0.0996 | 0.096 |
| dilated_K16_mixed | 0.061 | 0.066 | 0.061 | 0.064 |
| linear_K4_mixed | 0.993 (K4 mixed) | 0.282 | 0.281 | 0.283 |
| linear_K8_mixed | 0.717 (K8 mixed) | 0.335 | 0.336 | 0.336 |
| linear_K16_mixed | 0.542 | 0.541 | 0.542 | 0.544 |
| b3_gated_linear_K16_mixed | 0.065 | 0.065 | 0.065 | 0.065 |

- Per-seed values (`mqar_test_per_seed.csv`) are important for linear runs. linear_K16_mixed test K16_mixed per seed
  is about 0.47 / 1.00 / 0.16 (s6304/6305/6306). For B3 it is about 0.065 on every seed.
- **Test-time W=64 mask on dense-trained models** (`condition` = "dense-trained, W=64 mask at test only"):
  - dense_K16_mixed: K16 near 1.000, mixed 0.224, far 0.0153.
  - window_K16_near trained with the mask: 1.000 / 0.224 / 0.015.
  - Per-distance-bin rows are in `mqar_distance_bins.csv`.
- **Runs that did not learn on any seed** (reported, decision 2.20): window_K16_mixed, dilated_K16_mixed, and B3.
  dilated_K4_mixed s6306 reached 0.66 dev on its trained cell; the other two seeds stayed near 0.3.
- **Theoretical cost at T=256** (`mqar_theoretical_costs.csv`, decision 2.18):
  - Attended pairs (total over 4 heads): dense 131,584; window 57,472; dilated 45,376.
  - FP32 state bytes: dense 524,288; linear and B3 33,792.
  - Parameters: 437,824 for all B1/B2 models; 439,888 for B3 (+0.47%).
- **Forward-only sweep** (batch 2, FP32, untrained; `mqar_inference_sweep.csv`). Values at T=8192:

  | kind | median ms | peak alloc MiB | notes |
  |---|---|---|---|
  | dense (SDPA causal) | 40.6 | 110.8 | |
  | window | 94.7 | 1606.8 | dense boolean mask materialized: 256 MiB |
  | local_dilated | 95.4 | 2118.8 | dense boolean mask materialized: 256 MiB |
  | linear_parallel | 143.6 | 5262.8 | score tensor 2048 MiB |
  | linear_recurrent | 2960.5 | 1134.8 | |

  Window and dilated use SDPA with a dense boolean mask, which is a correctness implementation, not a sparse kernel.
- **B3 vs linear_K16_mixed** (`mqar_b3_vs_linear_bootstrap.csv`, paired over test examples, per seed and seed-mean):
  - K16_mixed diff: s6304 −0.403 [−0.410, −0.396]; s6305 −0.934 [−0.938, −0.931]; s6306 −0.095 [−0.101, −0.090].
  - Seed-mean diff: −0.477 [−0.480, −0.474].
  - The seed-mean CI resamples examples only; seed-to-seed variation is shown by the per-seed rows.
- **B3 prediction** (recorded before launch, decision 2.23, commit f515e9f): "clearly better; more consistent; mostly
  near queries".
- MQAR training ms/update was measured with 2 jobs sharing each T4, so it is not an isolated speed (decision 2.20).

### Part C — KV cache (Qwen2.5-1.5B-Instruct, FP16, greedy, 180 cases)
Grid: nominal length {2048, 4096} × evidence position {start, middle, end} × distractors {0, 2, 8} × 10 cases.

Actual context tokens (`kv_lengths.csv`): 2,605–2,649 (mean 2,627.7) and 5,193–5,243 (mean 5,216.9). Suffix: 33–34
tokens.

Budget = M_4(L), the serialized INT4 channel-K / token-V bytes. Mean budget is 20.70 MiB and 40.77 MiB, against
71.85 MiB and 142.65 MiB for FP16.

| method | accuracy | ref-answer NLL (mean) | post-prefill KL | evidence retained | K rel err | V rel err | fraction of FP16 |
|---|---|---|---|---|---|---|---|
| fp16 | 1.000 | 0.005 | 0 | – | – | – | 1.000 |
| evict_sink_recent | 0.333 | 9.163 | 2.098 | 0.333 | – | – | 0.2867 |
| evict_question | 0.983 | 0.038 | 0.00062 | 0.936 | – | – | 0.2869 |
| int4_token | 0.000 | 22.06 | 5.591 | – | 0.148 | 0.099 | 0.2869 |
| int4_channel_k | 1.000 | 0.008 | 0.0024 | – | 0.0068 | 0.0987 | 0.2869 |
| hybrid_int2 | 0.994 | 0.020 | 0.0039 | 0.880 | 0.031 | 0.464 | 0.2786 |

- Every method is within budget on all 180 cases and all logits are finite.
- `accuracy_given_fp16_correct` equals `accuracy`, because FP16 is correct on all 180 cases.
- **Breakdowns** (`kv_breakdowns.csv`; per position, 60 cases each):
  - evict_sink_recent: end 1.000, middle 0.000, start 0.000.
  - evict_question: end 1.000, middle 0.950, start 1.000.
  - hybrid_int2: end 1.000, middle 0.983, start 1.000.
  - By distractors 0/2/8, evict_question: 1.000 / 0.983 / 0.967. Lengths and the full 3-way cells are in the same file.
- **Paired bootstrap over cases** (`kv_bootstrap.csv`, accuracy, all 180):
  - evict_question − evict_sink_recent: +0.650 [0.578, 0.717].
  - int4_channel_k − int4_token: +1.000 [1.0, 1.0].
  - hybrid − evict_question: +0.011 [0.0, 0.028].
  - hybrid − int4_channel_k: −0.006 [−0.017, 0.0].
  - evict_question − fp16: −0.017 [−0.039, 0.0].

  The CIs with an endpoint at exactly 0 come from a handful of discordant cases (1–3), so they are not resolved.
  The NLL and KL rows give continuous comparisons.
- **C1 checks** (`kv_c1_checks.csv`, 6 cases, tolerance 2e-2): all passed.
  - Full vs split-prefill: max prob diff ≤ 1.6e-4.
  - 100% retention reproduces FP16 tokens exactly (diff 0.0).
  - Shared eviction vs masked full forward: max prob diff ≤ 1.6e-3.
- **kv_bench** (`kv_bench.csv`, real packing in PyTorch, 10 reps, medians, ms):

  | ctx tokens | batch | FP16 decode step | INT4 channel-K pack | unpack+dequant | unpack + step |
  |---|---|---|---|---|---|
  | 2,633 | 1 | 37.5 | 24.0 | 13.9 | 55.6 |
  | 2,633 | 8 | 138.8 | 139.6 | 86.2 | 226.0 |
  | 5,217 | 1 | 56.3 | 38.7 | 22.9 | 78.5 |
  | 5,217 | 8 | 293.5 | 280.6 | 172.2 | 455.1 |

  - Serialized bytes equal the formula exactly (asserted).
  - Serialized size at L=5,217, B=8: FP16 1,141.2 MiB, INT4 326.2 MiB (0.2858), INT2 184.4 MiB (0.1616).
- **Selection masks and generations for the 6 fixed cases** (`*-2-00`): `kv_fixed_case_generations.csv` and
  `figures/kv_selection_masks.png`. The masks plot the fraction of (layer, KV head) keeping each position, with the
  gold evidence span shaded.
- **Labels**:
  - Quality runs use **fake-quantized** caches (dequantized FP16 in the cache); bytes are the packed-format count.
  - kv_bench uses Python uint8 nibble/crumb packing, **not a fused low-bit attention kernel**. SDPA reads a
    dequantized copy.
  - The question-aware ranking is "our SnapKV-style scoring", **not full SnapKV**.
  - Gold evidence positions are used for evaluation only, never by a policy.
  - Retention granularity: shared for sink+recent; per (layer, KV head) for question-aware and hybrid
    (decision 2.21, which also gives the index accounting).

---

## Task 3 — distillation + speculative decoding

### Where (branch `task3-specdec`)
| What | File |
|---|---|
| Raw | `results/task3/spec/`, `align/`, `standalone/`, `distill/`, `merge/`, `markov_chi2.json`, `logs_stage*` |
| Tables | `results/task3/tables/*.csv` (17), `results/task3/summary.csv` |
| Figures | `results/task3/figures/{alignment_fkl_by_position,distill_kl_curves,speedup_measured_vs_predicted,standalone_accuracy}.png` |

### Definitions
- **accepted/examined** = accepted draft tokens / draft tokens the target examined. **accepted/proposed** also
  counts proposals cut off after a rejection.
- **Model terms**:
  - c = draft step cost / target step cost, measured at a context of 256.
  - v_γ = verify-pass cost / target step.
  - E(α) = (1 − α^{γ+1}) / (1 − α).
  - Ŝ = E / (γc + v_γ).
  - Break-even α solves E = γc + v_γ.
- **Measured S**: geometric mean of the per-prompt tok/s ratio (spec / target-only), paired and adjacent. The median
  ratio is also given.
- **Alignment** (decision 3.17): per bank record, the next-token distribution at FP32, T=1, over all 151,936 rows.
  - fkl = KL(target ‖ draft); rkl = KL(draft ‖ target); tv = total variation.
  - greedy_agree = whether the argmax matches; 1−tv = expected acceptance.

### Headline numbers
**Speculative decoding** (`spec_main.csv`, "all" split, 120 prompts × 2 reps, 64-token EOS-aware cap):

| draft | γ | acc/examined | acc/proposed | geomean tok/s ratio [95% CI] | median ratio |
|---|---|---|---|---|---|
| original | 2 | 0.872 | 0.813 | 0.864 [0.837, 0.891] | 0.921 |
| original | 4 | 0.868 | 0.705 | 0.757 [0.720, 0.795] | 0.836 |
| off_policy s6304 | 2 / 4 | 0.869 / 0.876 | | 0.870 / 0.790 | |
| off_policy s6305 | 2 / 4 | 0.865 / 0.881 | | 0.873 / 0.797 | |
| on_policy s6304 | 2 / 4 | 0.876 / 0.879 | | 0.869 / 0.779 | |
| on_policy s6305 | 2 / 4 | 0.858 / 0.866 | | 0.856 / 0.771 | |
| unmerged off_policy s6304 | 2 / 4 | | | 0.660 / 0.566 | |

Every speculative configuration has a tok/s ratio CI entirely below 1 (`spec_bootstrap.csv`).

Draft − original, paired, γ=4 (`spec_bootstrap.csv`):
- log tok/s ratio: off s6304 +0.043 [0.013, 0.076]; off s6305 +0.051 [0.021, 0.085]; on s6304 +0.029 [0.003, 0.056];
  on s6305 +0.018 [−0.012, 0.049].
- At γ=2, none of these is resolved.
- accepted/examined: on s6305 at γ=2 −0.012 [−0.023, −0.0005].

**Speedup model** (`speedup_model.csv`): c = 0.769.
- γ=2: v = 0.914, break-even α = 0.805.
- γ=4: v = 0.902, break-even α = 0.885.
- Original draft: Ŝ (measured α) 1.073 (γ=2) and 0.966 (γ=4); measured S geomean 0.864 and 0.757.

**Per-round decomposition** (`spec_decomposition.csv`), original γ=2, ms/token, from a separate synchronized pass:

| | draft | verify | accept | cache | prefill | end-to-end |
|---|---|---|---|---|---|---|
| speculative | 26.7 | 15.7 | 0.25 | 0.07 | 2.19 | 44.9 |
| target-only | – | 41.0 | – | – | 1.20 | 42.6 |

**Distillation** (`distillation_costs.csv`):
- 157 updates each; skipped updates 0 / 1 / 1 / 1; 30,000 math + 10,000 short scored tokens.
- Mean KL over the first 10 → last 10 updates:

  | run | first 10 | last 10 |
  |---|---|---|
  | off s6304 | 0.131 | 0.097 |
  | off s6305 | 0.133 | 0.099 |
  | on s6304 | 0.192 | 0.191 |
  | on s6305 | 0.261 | 0.192 |

- Upfront cost: off-policy about 4,000 s (teacher pool 3,791 s); on-policy about 2,400 s (rollouts about 2,200 s).

**Merge check** (`merge_check.csv`): merged vs unmerged max |logit| difference is 0.17–0.23 (FP16).

**Alignment** (`alignment.csv`; vs original: `alignment_vs_original_bootstrap.csv`). On the exact-spec bank, all
positions, fkl / tv:
- original 0.157 / 0.107
- off s6304 0.131 / 0.102
- off s6305 0.129 / 0.100
- on s6304 0.133 / 0.105

Student − original fkl, all splits:
- off s6304 −0.032 [−0.055, −0.006]
- off s6305 −0.040 [−0.060, −0.019]
- on s6304 −0.037 [−0.056, −0.019]
- on s6305 −0.036 [−0.057, −0.018]

TV for on s6305: −0.003 [−0.013, 0.006], not resolved.

**Bank consistency** (teacher_pure vs exact_spec; `bank_consistency_teacher_vs_exact_spec.csv`):
- All-split differences are not resolved for any draft or metric.
- Math split, original fkl: +0.031 [0.0045, 0.063] uncorrected; Bonferroni-3 [−0.0007, 0.071].

**Standalone quality** (`standalone.csv`, pooled over 2 reps; math cap 512, short cap 32):

| model | math acc | math truncated | familiar strict / lenient | shifted strict / lenient |
|---|---|---|---|---|
| target | 0.583 | 0.067 | 0.350 / 0.933 | 0.700 / 0.917 |
| original | 0.392 | 0.142 | 0.317 / 0.817 | 0.883 / 0.950 |
| off s6304 | 0.292 | 0.117 | 0.333 / 0.983 | 0.933 / 0.967 |
| off s6305 | 0.242 | 0.125 | 0.333 / 0.950 | 0.933 / 0.983 |
| on s6304 | 0.308 | 0.200 | 0.333 / 0.983 | 0.783 / 0.917 |
| on s6305 | 0.175 | 0.408 | 0.333 / 1.000 | 0.817 / 0.967 |

Paired bootstrap vs original (`standalone_bootstrap.csv`), math accuracy:
- off s6304 −0.100 [−0.192, −0.017]
- off s6305 −0.150 [−0.233, −0.067]
- on s6304 −0.083 [−0.167, 0.0]
- on s6305 −0.217 [−0.325, −0.117]
- target +0.192 [0.100, 0.292]

The familiar-split strict accuracy is bounded by format compliance (decision 3.16); `compliant` is in the same table.

**Deployment break-even** (`deployment_break_even.csv`):
- Against target-only: "never" for every draft and γ, because speculative is slower.
- Against the original draft at γ=4: 26,691 to 86,834 requests. At γ=2: "never".

### Caveats and labels
- The exact-speculative prefix bank is **generated by us** at γ=4; the data repo does not supply it (decision 3.1).
- The 0.5B student is merged into FP16 before timing. The unmerged-adapter timing is reported separately.
- The speedup model uses step costs at context 256. Measured prompts vary in length.
- No early stopping or checkpoint selection: every run uses its fixed 40,000-token quota.
- Held-out prompts never enter training or the teacher pool (asserted in `task3/load.py::train_prompts`). There is no checkpoint choice.
- Markov χ² (`markov_chi2.json`, 100,000 samples, 26 dof):
  - Correct sampler: p = 0.38–0.41 at γ = 1–4, not rejected.
  - "Residual = p" bug: χ² 16,668–24,196, p = 0, rejected.
