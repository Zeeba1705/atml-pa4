# CLAUDE4.md — ATML PA4 (compression, attention/KV memory, speculative decoding), Tasks 1–3

Ask me which task we're on if it isn't clear from my message. Task 1 work lives on branch `task1-compress`,
Task 2 on `task2-attention`, Task 3 on `task3-specdec`.

Course assignment, individual, **due 30 Nov 2026**. Public GitHub repo. I am responsible for and must understand every
line, so explain non-trivial changes before making them. Code style: follow CODE_STYLE2.md for all new code (its
section 12 "additions" — commit hash/config/seed/GPU in results, `--smoke`, `--resume`, unit tests — apply to PA4 too).

## Hard rules (never break these)
- NEVER write, edit, or draft anything in `report/` or report prose/analysis. The PDF report must be my own words
  (handout AI policy, p.1). You may produce plots, tables (CSV/markdown of numbers), and scripts that generate them.
- The course data repo `atml-pa4-datasets/` is a fixed asset: clone it next to the code (gitignored), never edit its
  files, never regenerate the frozen MQAR test/dev or KV files. Copy `data/local_source_revisions.json` (created by
  `setup_data.py`) into `splits/data_lock/` and commit it — I must report those revisions.
  Never commit weights, datasets, prefix banks, compressed models, cached activations, or `outputs/`.
- Pinned: `transformers==4.57.1`, `huggingface_hub==0.36.2`. Seed 6304 (Task 2B also 6305, 6306; Task 3 training
  6304, 6305). Read constants from `configs/`; don't hardcode duplicates or silently "improve" hyperparameters.
- Do not change models, splits, budgets, prompts, decoding settings or scoring rules between matched comparisons.
- No leakage, no selection on held-out results:
  - Task 1: calibration = the 128 C4 sequences only; P3/Q3 allocation hyperparameters chosen on the 32-sequence C4
    dev manifest only, frozen to JSON before any WikiText-2 / C4-val / HellaSwag result exists for that method.
  - Task 2B: B3 intervention chosen from B1/B2 results + MQAR **development** set; never from test.
  - Task 2C: gold `evidence_char_start/end` and the gold code are evaluation-only — never passed to a retention policy.
  - Task 3: GSM8K test + short-form test never enter training, teacher pools or checkpoint choice. No early stopping
    or "best checkpoint" — every run uses its fixed token quota.
  - Add cheap asserts for each of these.
- Report every specified condition, including failures, OOMs, and runs that don't learn. Never drop examples after
  looking at outcomes.
- Labels must be honest: fake-quantized FP16 ≠ low-bit kernel; zeros ≠ sparse speedup; Python tiled attention ≠
  FlashAttention; our SnapKV-style scoring ≠ full SnapKV; our AWQ scale search ≠ the AWQ inference engine.
  Keep "theoretical/packed bytes" and "measured allocated GPU memory / measured latency" as separate columns.
- If you reuse external code (AWQ reference, VQA-style normalizers, lm-eval acc_norm logic, KIVI/SnapKV snippets),
  tell me and add it to the README "Attribution" section.
- Don't run `git push --force`, `git reset --hard`, or delete branches without asking.

## Compute setup
- No local GPU. I write/test locally with you, then run on **Kaggle** via thin notebooks in `kaggle/`.
- Kaggle GPU = **T4 ×2** (16 GB each, **no bf16** — use FP16), ~30 GPU-h/week, sessions die at 12 h. Internet must be
  on for HF downloads. CPU sessions are free: data prep (`setup_data.py --public`), all analysis/bootstrap/tables on CPU.
- T4 facts to respect: no FlashAttention-2; SDPA will usually dispatch to mem-efficient or math backend — record what
  it actually picks. No 2:4 sparse tensor-core claims (handout p.4).
- Everything must be checkable on CPU first:
  - unit tests in `tests/` on small hand-made tensors (`python -m pytest tests -q`)
  - every script takes `--smoke` (few steps/examples, FP32 on CPU, may use a tiny random-init Qwen2 config with the
    real tokenizer) so the pipeline runs end to end without a GPU.
- Every GPU script takes `--device` so I can run two jobs at once, one per T4 (`CUDA_VISIBLE_DEVICES=0` / `=1`).
- Default `--output_dir` to `/kaggle/working/outputs` if `/kaggle` exists, else `outputs`. ~20 GB disk: never save
  full compressed copies of Qwen per condition — save masks/allocations/AWQ scales (small) and rebuild in seconds.
- Progress must survive a dead session: each (condition × eval) writes its own result file, skips finished items on
  rerun, and syncs finished files to a private HF Hub repo (`--sync_repo`, token `HF_TOKEN` from Kaggle Secrets;
  background upload, never crash on upload failure). Training jobs (2B, Task 3) checkpoint model/adapter, optimizer,
  scheduler, GradScaler, all RNG states, and data/prompt-schedule position; `--resume` is the default.
- One `kaggle/queue.py --task N` runs whatever isn't done yet, prints a done/partial/not-started table first.
- `kaggle/*.ipynb` are thin launchers only (clone/pull, pip install, run setup, call scripts, copy outputs).
- After a few steps, every long job prints measured speed and estimated total time.

## Before writing code: explore
Handout: `handout/PA4.pdf` (gitignored). Task 1 pp.2–5, Task 2 pp.5–11, Task 3 pp.11–15, report/deliverables pp.15–16.
Read `atml-pa4-datasets/README.md`, `docs/DATA_FORMATS.md`, `docs/REPRODUCIBILITY.md` and `scripts/mqar_generator.py`
before touching data. Check library APIs in the pinned versions before relying on them (DynamicCache internals and
`crop()` in 4.57.1, SDPA backend context managers, Qwen2 rotary/position handling, PEFT merge).
Pin the rest of `requirements.txt` once smoke tests pass; don't upgrade afterwards. (The handout mentions a "course
environment lock" — it's not in the data repo; record what we pin.)

Known data-repo facts (verified):
- `setup_data.py --public` writes `data/public_locked/*` (C4 calib 128×512, C4 dev 32×512, WikiText-2 / C4-val windows
  with `score_target_offsets` and `scored_tokens` summing to 50,000 each, HellaSwag 2,000, GSM8K train 500 / test 60).
- `--prefix-banks` (GPU) writes teacher + original_draft banks only, positions in bins [1,16],[17,32],[33,64],[65,192],
  `response_prefix_token_ids = seq[:pos-1]`. **The exact-speculative bank is NOT supplied** (the handout p.13 says
  supplied; the repo README says generate it with our own sampler at γ=4) — we generate it, using the same prompt
  order, chat template, sampling settings and position-selection rule; record this in docs/decisions.md.
- MQAR: keys at even positions 0..2K-2, value right after; `distances = query_pos - key_pos`; test 1,024 per (K, regime).
- KV cases: 180 prompts, question after the marker `QUESTION_BEGIN` in `user_text`.
- Short-form: train families 0–2 (300); test 30 `familiar` + 30 `shifted` templates.

Layout:
```
configs/{task1,task2,task3}/*.yaml
common/{seed,io,timing,bootstrap,sync,packing}.py
task1/{load,evaluate_ppl,evaluate_hellaswag,calibrate,prune,quantize,awq,sensitivity,allocate,resources,run_condition,make_tables}.py
task2/attention_ops.py  task2/bench_attention.py
task2/mqar_model.py  task2/train_mqar.py  task2/eval_mqar.py  task2/bench_mqar_inference.py
task2/kv_cache.py  task2/kv_policies.py  task2/kv_quant.py  task2/kv_eval.py  task2/kv_bench.py  task2/make_tables.py
task3/{spec_sampler,toy_markov,load,teacher_pool,distill,merge,prefix_banks,align_metrics,spec_eval,standalone_eval,cost_model,make_tables}.py
splits/{data_lock,task1,task2,task3}/   results/{task1,task2,task3}/   tests/   kaggle/   docs/decisions.md
```

## Task 1 spec — compression of Qwen/Qwen2.5-1.5B base (handout pp.2–5)
FP16, no training. Each condition starts from a fresh copy of unchanged weights. Eligible = q,k,v,o,gate,up,down in
every block; embeddings, norms, biases, lm_head untouched; tied params counted once in storage.

Evaluation (shared code, used by every condition):
- PPL on WikiText-2 test windows and C4-val windows exactly as materialized: each window independent, score only
  `score_target_offsets`, report N, mean NLL, PPL. Assert N = 50,000 per suite and identical across conditions.
  Save per-window NLL sums + counts (for window bootstrap).
- HellaSwag: zero-shot, sum log-likelihood of each ending given the context; acc and acc_norm (length-normalized by
  **characters** of the ending, lm-eval convention; not tokens). Record the exact prompt/preprocessing (lm-eval's
  hellaswag preprocess) and save per-example 4 candidate scores. Full 2,000 for: baseline, P1/P2 at 50%, Q1/Q2/Q3 at
  INT4 and INT3 (handout p.3 wording — confirm list with me). All other conditions: both PPL suites + first 512 IDs.
- Resources per condition: effective nonzeros, nominal bits/eligible weight, packed bytes incl. scales/zeros/padding,
  16-bit sparse estimate (1 mask bit/eligible weight + 16 bits/nonzero, byte-aligned) next to dense FP16, compression/
  calibration time, peak allocated memory, prefill latency at 512 and 2,048, 64-token decode latency, ≥5 warmed reps.

Steps:
0. Tests first (`tests/test_task1.py`): eligible-matrix enumeration + tied-weight counting on a tiny Qwen2 config;
   per-row pruning hits exact count; Wanda score = |W|·‖X_j‖₂ on hand-made tensors; quantize→dequantize round trip on
   a hand group (incl. padding); packed-byte formula vs hand count; AWQ rescale-before-rounding preserves outputs;
   allocation rescaling hits target zero count exactly with 0.10 ≤ s ≤ 0.85; 2:4 mask keeps exactly 2 of every 4;
   PPL scorer matches a hand-computed NLL; acc_norm vs hand example. Show me the tests, then implement.
1. Baseline + measurement protocol.
2. P1 magnitude, per-output-row, s ∈ {0.25, 0.50, 0.75}. Report achieved nonzero fraction and layer-output error on
   calibration activations.
3. P2 Wanda, same sparsities/conventions, same 128 calibration seqs. Save per-projection score distributions.
4. P3 nonuniform Wanda at 50% and 75%, matched by total zeroed eligible weights; Wanda ranking fixed. **I choose the
   sensitivity measure and allocation rule** — you may lay out options with the paper they come from (OWL outlier
   ratio, ShortGPT block influence, Taylor, isolated-removal Δloss on dev) and their cost; don't pick for me.
5. Q1 uniform asymmetric group-wise weight-only PTQ, b ∈ {8,4,3}, group 128 along input columns. Fix and document
   rounding, scale/zero-point dtype and storage, padding and packing once; identical for Q1/Q2/Q3.
6. Q2 AWQ-style: per-input-channel scales from calibration activations, small recorded grid (e.g. α ∈ {0,0.05,…,1}),
   fold inverse scales where algebraically valid (input_layernorm → q/k/v jointly, post_attention_layernorm →
   gate/up jointly, up→down). **v→o under GQA (12 q heads / 2 kv heads) needs care — ask me before folding it**;
   otherwise store FP16 compensating scales and count their bytes. Assert pre-rounding outputs unchanged.
7. Q3 mixed precision, bits ∈ {2,3,4,8}, same quantizer as Q2, two allocations with packed bytes ≤ Q2-INT4 and
   ≤ Q2-INT3 budgets (incl. metadata and all uncompressed params). I choose sensitivity + rule (offer options).
8. 2:4 vs unstructured at 50% Wanda.
9. Tables/plots: quality vs sparsity and vs packed bytes; sensitivity map per adaptive method; final sparsity/bit maps;
   calibration-error vs held-out quality; paired 95% bootstrap P3−P2 and Q3−Q2 (HellaSwag: examples; PPL: resample
   windows, pool NLL, then exp). Fixed-rule qualitative case finder (I choose and interpret).

## Task 2 spec — attention and KV memory (handout pp.5–11)
### Part A — operators (`task2/attention_ops.py`, `bench_attention.py`)
Dense causal (explicit scores), tiled online-softmax (exact, no full L×L), SDPA. B=2, H=4, d=64,
L ∈ {256,…,8192}, FP16 and FP32. Errors (output + grads) vs dense FP32 where feasible; future-perturbation causality
test; all-masked / extreme-score edge cases; 10 warmed reps; forward vs forward+backward; peak allocated and reserved;
record the dispatched SDPA backend (try forcing each, log unsupported). Skip infeasible dense shapes, log as such.

### Part B — MQAR (`task2/mqar_model.py`, `train_mqar.py`)
2 pre-norm blocks, d=128, 4 heads × 32, 4× MLP, learned token emb + sinusoidal pos, vocab 256; only the attention
operator varies. AdamW LR 8e-4 wd 0.01, batch 32, 4,500 updates, 200 warmup then linear to 15%, clip 1.0, evals at
500/1000/1500/2000/3000/4500 (on dev; final checkpoint used for test — no best-checkpoint choice). Seeds 6304/5/6.
Loss = CE at query positions only against value IDs (not next-token LM loss). Training stream = `make_example('train',…)`.
Report accuracy by regime and distance bins [0,32),[32,64),[64,128),[128,256), plus 1/64 and 1/K baselines.
Runs: B1 dense K=4,8,16 mixed + dense K=16 near; window W=64 K=4,8,16 near + K=16 mixed; dense model with W=64 mask at
test only. B2 ELU+1 linear (causal recurrence) K=4,8,16 mixed; local/dilated (2 heads last 64, 2 heads offsets
{0,4,…,252}) K=4,16. B3 one intervention at K=16 that I pick after B1/B2, within 5% params, prediction written down
first. Theoretical pair counts/state bytes at T=256 + forward-only sweep T ∈ {256,1024,4096,8192}; state when a dense
mask is materialized. Tests: masks (window, dilated offsets, causality), linear recurrence == parallel causal form,
query-position loss masking, receptive-field limit 2(W−1)=126.

### Part C — KV cache (`task2/kv_*.py`)
`Qwen/Qwen2.5-1.5B-Instruct` FP16, greedy, SDPA, chat template on `user_text`. Record actual token lengths.
- C1: full-forward logits vs split-prefill/decode; 100% retention reproduces normal generation; finite logits; full-
  cache accuracy per cell. Evicted caches must keep original absolute RoPE positions for retained keys; new tokens get
  their true absolute positions; physical cache length handled consistently (explain your approach to me first).
- Budget per case = actual serialized bytes of uniform INT4 channel-wise-K / token-wise-V at the pre-question cache:
  M_b(L) = 2·nℓ·Hkv·dh·[⌊L/64⌋(64b/8 + 4) + 2(L mod 64)], group 64, FP16 scale + FP16 offset. Question suffix and
  generated tokens are a shared FP16 component reported separately.
- C2 eviction (FP16): sink 8 + recent 64 in both; fill rest with most recent (locality) or question-aware ranking from
  masked FP32 attention of question-suffix queries vs cached rotary keys (no eager path). State shared vs per-head/
  per-layer selection and count indices.
- C3 INT4: token-wise K&V vs channel-wise-K/token-wise-V; same rounding/metadata, same FP16 trailing rule; quantize
  prompt cache and later generated tokens. K and V reconstruction error, teacher-forced logit KL, accuracy.
- C4 hybrid: sink+recent+question-aware FP16, rest channel-K/token-V INT2 in complete groups, residual FP16; largest
  FP16 count fitting the budget incl. indices/metadata/padding.
- C5: exact match, reference-answer NLL, post-prefill KL vs FP16, evidence-token retention, realized bytes and
  fraction of FP16, scoring/prefill/decode time, peak memory; paired bootstrap ≥2,000 over prompts; conditional-on-
  reference-correct accuracy too. Separate encode/decode benchmark at batch 1 and 8, both lengths, ≥10 reps.
Tests: byte formula vs hand counts; group packing round trip; channel vs token grouping shapes; trailing-r FP16 rule;
eviction keeps sink/recent and never exceeds budget; retained RoPE positions unchanged.

## Task 3 spec — distillation + speculative decoding (handout pp.11–15)
Target/teacher `Qwen/Qwen2.5-1.5B-Instruct`, draft/student `Qwen/Qwen2.5-0.5B-Instruct`, both frozen FP16; LoRA r8 α16
dropout 0 on q,k,v,o of the student. Assert tokenizer + chat-template agreement and equal vocab size (KL over the full
common vocab). One non-thinking chat format, fixed EOS handling everywhere.

0. Tests first (`tests/test_task3.py`): accept prob min(1,p/q); residual normalization; bonus token on full accept;
   cache rollback/advance (`crop`) gives logits equal to a fresh forward on the returned sequence; EOS mid-round;
   γ > remaining tokens. 3-symbol Markov toy: exact joint over length-3 sequences, 100,000 seeded samples, Pearson χ²
   doesn't reject correct sampler at p<0.01 and DOES reject the "residual replaced by p" bug. Show me, then implement.
1. Sampler on real models + split-prefill equivalence check before any timing.
2. Baseline: target-only vs spec with original draft, γ ∈ {2,4}: accepted/examined, accepted/proposed, mean accepted
   prefix, target passes/token, draft passes, memory, EOS/truncation, paired time & tok/s; warm-up, prefill and steady
   decode separated; isolated draft/target step costs at several prompt lengths.
3. Off-policy: frozen teacher-response pool (T=1, top-p 1, caps 192 math / 32 short), cycle to quota.
4. On-policy: fresh current-student rollouts, teacher scores them, no gradient through generation.
   Both: 40,000 teacher-scored tokens (30k math + 10k short), microbatch 1, update every 256 scored tokens,
   AdamW 3e-4 wd 0.01, FP16 autocast + GradScaler, unscale before clip 1.0, KL normalized by supervised tokens per
   update, seeds 6304/6305, independent from the original student. Log per presentation: source ID, rollout seed,
   response IDs, termination reason, scored tokens; skipped updates; all cost components separately.
5. Alignment on common prefixes: forward KL, reverse KL, TV, greedy agreement, 1−TV for original/off/on drafts on
   teacher, original-draft, our exact-spec bank, and each trained student's frozen own-rollout bank (both students on
   each other's). By math / familiar / shifted, positions [1,16],[17,32],[33,64] and 65–192; seed dispersion;
   teacher-vs-spec bank consistency check.
6. Merge each adapter into a fresh FP16 backbone, report merged-vs-unmerged max logit error. Spec eval on 120 prompts,
   γ ∈ {2,4}, 2 paired reps, 64-token EOS-aware cap; counterbalanced order; adjacent target-only timing; per-round
   decomposition (draft / verify / accept-residual / cache) + prefill + host; c, v_γ, Ŝ_γ, measured S_γ, break-even α.
   24-prompt fixed-64 no-EOS throughput diagnostic. One unmerged adapter timing.
7. Standalone student quality (512 math / 32 short caps): answer extraction, format compliance, truncation flags;
   by workload and output-length bins; amortized break-even incl. distillation + merge cost. Paired bootstrap 95% CIs
   over prompts; seeds reported separately.

## Open questions — ask me, don't guess (record answers in docs/decisions.md with handout page)
- T1: HellaSwag full-2,000 condition list; zero-point storage (FP16 vs packed b-bit vs 8-bit); AWQ v→o fold under GQA;
  P3/Q3 allocation unit (block vs projection); sensitivity measures and allocation rules (mine to choose).
- T2: tile sizes; how cache "physical length" and RoPE positions are handled after eviction; retention shared across
  heads/layers or independent; exact question-suffix span for scoring; B3 intervention (mine).
- T3: how math/short presentations interleave under the round-robin + token quotas; teacher pool size (1 or several
  samples per prompt); whether a response straddling the 256-token update boundary is split or the update runs long;
  per-round synchronized timing vs unsynchronized end-to-end timing (probably both, separate passes).

## Results and logging
- `results/<task>/<run_name>/`: `config.json`, per-update `train_log.jsonl` where training exists,
  `<run_name>_history.json` (as per CODE_STYLE2.md), metrics JSON, per-example outputs (jsonl/parquet).
- Every results file records: git commit, seed, config, GPU name, dtype, attention backend, model revision (from the
  data lock), torch/transformers versions, wall-clock, peak allocated (and reserved where relevant) memory.
- Timing helper in `common/timing.py`: warm-up, `torch.cuda.synchronize()` around timed regions, fixed shapes, reports
  unit + reps + median/IQR. Use it everywhere.
- `make_tables.py` per task → `results/<task>/summary.csv` + `figures/` (I copy what I need into the report). Large grids
  go to machine-readable supplementary files.
- Every interpretation choice or deviation goes in `docs/decisions.md` with the handout page.

## README
Top-level README.md, one section per task: environment, data setup (`setup_data.py --public`, `--prefix-banks`),
exact commands for every experiment (smoke + full), which Kaggle notebook runs what, expected outputs, "Attribution".
Flag names use underscores, e.g. `python -m task1.run_condition --config configs/task1/base.yaml --method wanda --sparsity 0.5`.

## Workflow
- One coherent change per commit (tests / one method / eval / tables). No giant commits.
- Deadline is tight (30 Nov). Order to fit GPU quota and leave report time:
  1. Day 1 CPU: skeleton, start `setup_data.py --public` (C4 streaming is slow), all Step-0 tests for 3 tasks,
     Task 3 Markov χ² sampler test, MQAR model + training `--smoke`.
     GPU: `setup_data.py --prefix-banks`.
  2. GPU: Task 1 baseline + P1/P2/Q1/Q2 (one T4) ∥ Task 2B training sweep (other T4, many small runs per session).
  3. GPU: Task 2A benchmarks; Task 2C C1 then C2–C5 ∥ Task 3 baseline spec eval + spec bank + distillation runs.
  4. Pause for my P3/Q3 and B3 decisions (made on calibration/dev data only), then run them.
  5. GPU: Task 3 merged-draft spec eval + standalone quality. CPU: bootstraps, tables, figures, README.
- Before any full-run command, show me the command, the GPU it goes on, expected runtime, and expected outputs.
  After I paste a Kaggle error, fix the root cause and add a smoke check that would have caught it.
- When unsure about a spec detail, ask me instead of guessing; quote the handout line you're unsure about.
