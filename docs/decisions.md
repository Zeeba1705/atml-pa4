# Decisions and deviations

Every interpretation choice or deviation from the handout (PA4.pdf), with the page it concerns.
Rule used throughout: the handout is the primary authority; where it is silent we follow the reference method of its
required readings; choices the handout leaves to the student are marked **(student choice)**.

## General

| ID | Decision | Handout basis |
|---|---|---|
| G1 | Qwen2.5-Instruct `generation_config.json` sets `repetition_penalty=1.1, top_k=20, top_p=0.8, temperature=0.7`. `setup_data.py --prefix-banks` overrides temperature/top_p/top_k but not the penalty, so the supplied `teacher` and `original_draft` banks are sampled with repetition penalty 1.1. We keep the supplied banks as the common prefix sets and additionally generate a `teacher_pure` bank (target-only, penalty 1.0, same prompts / seed formula / position rule); the target-vs-speculative consistency check compares `exact_spec_original_gamma4` against `teacher_pure`. All our own generation calls set `repetition_penalty=1.0, top_k=0, top_p=1.0` explicitly. | p.13 "(i) and (iii) … sample the same target-prefix law" |
| G2 | No "course environment lock" exists in the data repo; we pin our own versions in `requirements.txt` and record torch/CUDA at runtime. | p.2 "use the course environment lock for all other dependencies" |
| G3 | No early stopping and no best-checkpoint selection anywhere (overrides CODE_STYLE2 §9). Runs save final + resume checkpoints only. | p.7 "Preserve learning curves and the final checkpoint"; p.2 "Do not use held-out test outcomes to choose … distillation checkpoints" |
| G4 | `setup_data.py --prefix-banks` passes `generator=` to `model.generate`, which the pinned transformers 4.57.1 rejects (`ValueError: model_kwargs are not used: ['generator']`), so the supplied script cannot run as released. We run it unmodified via `kaggle/supplied_banks.py`, which drops that kwarg and calls `torch.manual_seed(generator.initial_seed())` instead; 4.57.1 samples with `torch.multinomial` on the default RNG, so this reproduces the intended per-prompt seeds `6304+10007j+offset`. Prompts, position rule and output file are unchanged. Tested in `tests/test_supplied_banks.py`. Generated 2026-11-23 on Colab (Tesla T4) via `colab/setup_banks.ipynb` with the committed lock: 120 prompts × {teacher, original_draft}, `prefix_banks_locked.jsonl.gz` = 145,049 bytes, `check_setup.py` OK. | p.13 "Use the supplied banks"; p.2 pinned transformers |

## Task 1

| ID | Decision | Handout basis |
|---|---|---|
| 1.1 | Full 2,000 HellaSwag for: baseline; P1/P2/P3 at 50%; Q1/Q2 at INT4 and INT3; Q3 at INT4 and INT3 budgets (10). All others, incl. 2:4, use the first 512. | p.3 "the three 50%-sparsity pruning methods, and all three INT4 and INT3 quantization variants" |
| 1.2 | FP16 scale per 128-group; integer zero-point z clamped to [0, 2^b−1] and stored packed at b bits; `torch.round` (half-to-even). Range widened to include 0 (w_min = min(w,0), w_max = max(w,0), GPTQ convention) so z always fits in [0, 2^b−1]; padding columns are zero, which therefore never changes a group's range. The scale is rounded to FP16 *before* codes are computed, so dequantization uses exactly the stored scale. Codes and zero-points are separate b-bit streams; LSB-first bitstream per tensor, row-major, byte-padded at the tensor end. All eligible d_in (1536, 8960) are multiples of 128, so group padding is zero in practice. | p.2 quantizer formula ("z is an integer zero-point"); p.4 step 5 |
| 1.3 | Min/max range, no clipping search, identical for Q1/Q2/Q3. AWQ's auto-clip step is therefore not reproduced. | p.2 "s ≈ (wmax−wmin)/(2^b−1)"; p.5 "clipping … must agree across Q1/Q2/Q3" |
| 1.4 | AWQ folds: input_layernorm → q/k/v, post_attention_layernorm → gate/up, up → down (v_proj bias and RMSNorm weights rescaled, function-preserving). o_proj input scales are not foldable into v_proj under GQA (12 query / 2 KV heads), so they are stored as FP16 and their bytes counted. | p.4 step 6 "Fold … when algebraically valid; otherwise serialize the compensating FP16 channel scales and include their bytes" |
| 1.5 | Calibration activations collected sequentially (block ℓ sees outputs of already-compressed blocks < ℓ), as in the Wanda and AWQ reference code. Layer-output error = per-projection ‖WX−ŴX‖²/‖WX‖² on those inputs. | handout silent; required readings p.3 |
| 1.6 | P3 rescaling: s_u = clip(raw_u + t, 0.10, 0.85) with the shift t found by bisection so the size-weighted mean equals the target (keeps the sensitivity ranking). Per-unit zero counts: floor, then largest-remainder +1s so the total exactly equals P2's zero count (asserted). | p.4 step 4 "rescale/round allocations to meet the target sparsity exactly" |
| 1.7 | P3/Q3 sensitivity measure, allocation rule and unit **(student choice)** — pending. | p.4 steps 4, 7 |
| 1.8 | HellaSwag preprocessing and acc_norm follow lm-eval `hellaswag` v1.0 (attributed in README), quirks included: `preprocess` strips before removing `[step]`-style tags, so an ending that starts with a tag keeps a leading space; the continuation scored is `" " + choice` and acc_norm divides by `len(choice)` (characters, space excluded). | p.3 "acc_norm convention used by lm-evaluation-harness" |
| 1.9 | Latency: batch 1, prompt = first 512/2,048 tokens of concatenated C4-val windows, 2 warm-ups + 5 reps, `cuda.synchronize` around each timed call. Prefill = one forward building the KV cache with the LM head on the last position only (`logits_to_keep=1`, as in generation). Decode = 64 greedy tokens ignoring EOS after an untimed 512-token prefill. PPL windows: batch 1, no padding. HellaSwag: the 4 endings of an example are scored in one right-padded batch (tested equal to unpadded scoring). | p.4 shared resources |
| 1.10 | `setup_data.py` treats each WikiText-2 HF row as a document, so PPL is not comparable to published concatenated-stream numbers. Used as materialized. | p.3 "Tokenize each document separately" |

## Task 2

| ID | Decision | Handout basis |
|---|---|---|
| 2.1 | Tiled attention default tiles 128×128; verified across {32, 64, 128, 256}. Backward via autograd (limitation reported). | p.8 A2 |
| 2.2 | SDPA called with `is_causal=True`; dispatched backend recorded from profiler kernel names plus forced-backend trials. | p.7, p.9 A3 |
| 2.3 | MQAR head is 64-way over value IDs 128–191 (label = id − 128). | p.7 "The label-space chance baseline is 1/64" |
| 2.4 | One training stream for all seeds and methods at a given (K, regime): index = update × 32 + b. The seed sets initialization only. | p.7 "Keep the supplied disjoint training, development, and final-test streams" |
| 2.5 | Checkpoint evals at 500…4,500 on the development set; test once on the final checkpoint. Test-time W=64 mask applied to all four dense models. | p.7 table; p.9 B1 |
| 2.6 | KV pipeline: prefill pre-question context → compress → prefill question suffix over the compressed cache → greedy decode. Question-aware scoring uses a separate full-FP16 suffix pass on an uncompressed copy (counted as scoring time). | p.8 "the point where the pre-question context cache is compressed … question suffix … shared decode-time component"; p.10 "post-prefill logit KL" |
| 2.7 | Question suffix = from the token where `QUESTION_BEGIN` starts through the end of the generation prompt. Sinks = first 8 tokens (system prompt); recent = last 64 tokens before the marker. Budget uses L = pre-question length. | p.8, p.9 C2 |
| 2.8 | After eviction, cached keys keep their post-RoPE rotation; new tokens get explicit true absolute `position_ids` and physical `cache_position`. Verified against a full forward with a 4D mask hiding evicted positions. | p.9 C1 |
| 2.9 | Retention selected independently per layer × KV head (same count per head); score = summed FP32 attention mass from the 6 grouped query heads and all suffix queries; no max-pool (simplified SnapKV-style). int16 index per retained position per (layer, head), counted. **(student choice)** | p.9 C2 "Specify whether retention is shared across heads/layers or selected independently" |
| 2.10 | Appended tokens (question suffix, generated) stay FP16 until they complete a 64-token group, which is then quantized (KIVI-style), for C3 layouts and the hybrid's low-bit part. | p.8 "including newly generated tokens until they form a complete group" |
| 2.11 | Quality runs store dequantized FP16 values in the cache (fake-quant, labelled); real packed encode/decode is benchmarked separately in C5. | p.10 C5 |
| 2.12 | Greedy, max 16 new tokens, stop at EOS; prediction = first `\d{5}`; reference NLL teacher-forced over the 5 code tokens. | p.8 "fixed, deterministic answer extractor" |
| 2.13 | No development prompts: all retention parameters are fixed by the handout, nothing is tuned. | p.8 |
| 2.14 | B3 intervention **(student choice)** — pending, chosen from B1/B2 + development set. | p.9 B3 |

## Task 3

| ID | Decision | Handout basis |
|---|---|---|
| 3.1 | The exact-speculative bank is not supplied (data README); we generate it with our sampler at γ=4, same prompts, RNG string `PA4-banks-v1:6304:{pid}:exact_spec_original_gamma4`, generation seed `6304+10007j+456`. | p.13 "Use the supplied banks … (iii)" vs data README |
| 3.2 | Prompt = raw question / instruction as the single user message with the default chat template (as in `setup_data.py`), everywhere. Math answer: `\boxed{}` if present, else last number (commas removed); short-form: stripped exact match; unextractable/truncated flagged. | p.12 "common non-thinking assistant prompt format" |
| 3.3 | Seeded permutation per category (seed = training seed), cycled after exhaustion; presentations alternate math/short, and once one quota is met only the other continues. Off- and on-policy share the schedule at the same seed. | p.12 "fixed, seeded round-robin permutation within the two quota categories; reuse prompts after exhausting a category" |
| 3.4 | Off-policy pool: one frozen teacher response per training prompt (800), cycled. | p.12 "frozen teacher-response collection … cycle deterministically" |
| 3.5 | Exactly 256 scored tokens per update; a straddling response is split (remainder re-forwarded with the updated adapter in the next update). | p.13 "one update per 256 teacher-scored response tokens (short final update permitted)" |
| 3.6 | Per-round synchronized timers per phase (draft / verify / accept-residual / cache); sync overhead measured and reported. | p.14 step 6 "Time each speculative round with synchronized timers" |
| 3.7 | Standalone student quality with T=1, top-p=1 sampling, 2 reps. | p.13 "Evaluate temperature-1, top-p = 1 sampling" |
| 3.8 | Student own-rollout banks on the 120 held-out prompts, same position rule; teacher + all 5 drafts evaluated on all 4 student banks. | p.13 "Also required" |
| 3.9 | Stop on EOS 151645 or 151643; EOS counts as an output token in acceptance stats. KL over all 151,936 logit rows. LoRA init seeded by the training seed. Throughput diagnostic uses the first 24 GSM8K test prompts by order. | p.12–13 |
| 3.10 | The data's `family` field (0/1/2) is the task type, shared by familiar and shifted; template transfer is recorded in `split`. Breakdowns use `split`. | p.12 |
