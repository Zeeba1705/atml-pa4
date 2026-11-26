#task 2 part C1-C5 quality run (handout pp.8-10): fp16 reference + 5 compressed caches at the case-specific int4 budget
#on all 180 prompts. per case: prefill the pre-question context once, question-aware scores once (suffix queries only),
#then per method: compress a copy -> prefill the question suffix -> teacher-forced code nll/kl -> greedy decode.
#c1 validity checks on the first case of each (length, position). one result file per case, finished cases are skipped
import os
import re
import time
import gzip
import json
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM

from common.seed import set_seed
from common.timing import sync
from common.io_utils import load_config, write_json, resolve_data_root, default_output_dir, run_info, model_revision
from task2.kv_quant import mb_bytes, fp16_bytes
from task2.kv_cache import encode_case, prefill, forward, clone_cache, gather_cache, greedy_decode
from task2.kv_policies import METHODS, Policy, question_scores, shape_of, evict_counts, sink_recent_index

CHECK_TOL= 2e-2   #max |prob diff| for fp16 path equivalences (same as task 3)


def load_model(cfg, device, smoke):
    if smoke:
        #tiny random qwen2, head_dim 64 so token-wise groups of 64 channels exist
        torch.manual_seed(1)
        s= cfg["smoke_model"]
        config= Qwen2Config(vocab_size=151936, hidden_size=s["hidden_size"], intermediate_size=s["intermediate_size"],
                            num_hidden_layers=s["num_hidden_layers"], num_attention_heads=s["num_attention_heads"],
                            num_key_value_heads=s["num_key_value_heads"], tie_word_embeddings=True, attn_implementation="sdpa")
        return Qwen2ForCausalLM(config).float().to(device).eval()

    model= AutoModelForCausalLM.from_pretrained(cfg["models"]["target"], revision=model_revision(cfg, "target"),
                                                torch_dtype=torch.float16, attn_implementation=cfg["attn_implementation"])
    model= model.to(device).eval()
    assert shape_of(model) == (cfg["num_layers"], cfg["num_kv_heads"], cfg["head_dim"])

    return model


def read_cases(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f]


def timed(device, fn):
    sync(device)
    t= time.perf_counter()
    out= fn()
    sync(device)

    return out, time.perf_counter() - t


def rel_err(cache, ref, L):
    #relative frobenius error of the compressed context k and v vs fp16, pooled over layers
    out= {}

    for name in ["keys", "values"]:
        num= sum(float((getattr(a, name)[:, :, :L].float() - getattr(b, name)[:, :, :L].float()).pow(2).sum()) for a, b in zip(cache.layers, ref.layers))
        den= sum(float(getattr(b, name)[:, :, :L].float().pow(2).sum()) for b in ref.layers)
        out[name]= (num / den) ** 0.5

    return out


def max_prob_diff(a, b):
    return float((torch.softmax(a.float(), -1) - torch.softmax(b.float(), -1)).abs().max())


def c1_checks(model, enc, ref_post, ref_tokens, cfg, eos_ids, max_new):
    #full forward vs split prefill; 100% retention == ordinary generation; shared eviction == full forward with masked
    #positions (true rope positions after eviction)
    ids, L= enc["ids"], enc["L"]
    suffix= ids[L:]
    nl, hkv, dh= shape_of(model)
    device= model.device

    with torch.no_grad():
        full= model(input_ids=torch.tensor([ids], device=device), logits_to_keep=1).logits[0, -1].float()

    cache, _= prefill(model, ids[:L])
    c100= gather_cache(cache, torch.arange(L, device=device)[None, None].expand(nl, hkv, -1))
    post100= forward(model, c100, suffix, L)[-1]
    tokens100= greedy_decode(model, c100, post100, len(ids), max_new, eos_ids)

    n= evict_counts(L, mb_bytes(L, cfg["quant"]["budget_bits"], nl, hkv, dh), nl, hkv, dh,
                    cfg["eviction"]["sink_tokens"], cfg["eviction"]["recent_tokens"])["evict_sink_recent"]
    idx= sink_recent_index(L, n, cfg["eviction"]["sink_tokens"], nl, hkv, device)
    post_ev= forward(model, gather_cache(cache, idx), suffix, L)[-1]

    N= len(ids)
    mask= torch.ones(N, N, dtype=torch.bool, device=device).tril()
    evicted= torch.ones(L, dtype=torch.bool, device=device)
    evicted[idx[0, 0]]= False
    mask[L:, :L]&= ~evicted

    with torch.no_grad():
        full_ev= model(input_ids=torch.tensor([ids], device=device), attention_mask=mask[None, None], logits_to_keep=1).logits[0, -1].float()

    out= {
        "full_vs_split_prefill": max_prob_diff(full, ref_post),
        "retain100_vs_fp16_post": max_prob_diff(post100, ref_post),
        "retain100_tokens_equal": tokens100 == ref_tokens,
        "eviction_vs_masked_full_forward": max_prob_diff(post_ev, full_ev),
        "finite": bool(torch.isfinite(full).all() and torch.isfinite(post_ev).all()),
        "tolerance": CHECK_TOL
    }
    ok= out["finite"] and out["retain100_tokens_equal"] and max(out[k] for k in ["full_vs_split_prefill", "retain100_vs_fp16_post", "eviction_vs_masked_full_forward"]) <= CHECK_TOL
    out["passed"]= ok

    return out


def run_case(model, tok, case, cfg, device, max_new, eos_ids, regex, keep_masks):
    enc= encode_case(tok, case, cfg["question_marker"])
    ids, L= enc["ids"], enc["L"]
    suffix= ids[L:]
    nl, hkv, dh= shape_of(model)
    budget= mb_bytes(L, cfg["quant"]["budget_bits"], nl, hkv, dh)
    code= tok.encode(case["code"], add_special_tokens=False)
    assert len(code) == 5, code

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    (ctx, _), prefill_s= timed(device, lambda: prefill(model, ids[:L]))
    prefill_peak= torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    scores, scoring_s= timed(device, lambda: question_scores(model, clone_cache(ctx), suffix, L))

    row= {"id": case["id"], "target_tokens": case["target_tokens"], "position": case["position"],
          "distractors": case["distractors"], "code": case["code"], "context_tokens": L, "suffix_tokens": len(suffix),
          "total_prompt_tokens": len(ids), "budget_bytes": budget, "fp16_context_bytes": fp16_bytes(L, nl, hkv, dh),
          "evidence_tokens": len(enc["evidence"]), "context_prefill_seconds": prefill_s, "context_prefill_peak_bytes": prefill_peak,
          "question_scoring_seconds": scoring_s, "methods": {}}
    ref= None

    for method in METHODS:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        policy= Policy(method, L, budget, model, cfg)
        cache, compress_s= timed(device, lambda: policy.compress(clone_cache(ctx), scores))
        post, suffix_s= timed(device, lambda: forward(model, cache, suffix, L)[-1])
        policy.after_append(cache)

        #teacher forcing over the reference code (the cache is cropped back afterwards, nothing quantized meanwhile)
        before= cache.get_seq_length()
        tf= forward(model, cache, code[:4], len(ids), keep=4)
        cache.crop(before)
        logp= torch.log_softmax(torch.cat([post[None], tf]), -1)

        tokens, decode_s= timed(device, lambda: greedy_decode(model, cache, post, len(ids), max_new, eos_ids, policy.after_append))
        text= tok.decode(tokens, skip_special_tokens=True)
        m= re.search(regex, text)

        if method == "fp16":
            ref= {"logp": logp, "tokens": tokens, "post": post}

        p_ref= ref["logp"].exp()
        kl= (p_ref * (ref["logp"] - logp)).sum(-1)
        r= {
            "prediction": m.group(0) if m else None, "correct": bool(m and m.group(0) == case["code"]), "response": text,
            "n_generated": len(tokens), "stopped_eos": bool(tokens and tokens[-1] in eos_ids),
            "ref_answer_nll": float(-logp[torch.arange(5), torch.tensor(code)].sum()),
            "post_prefill_kl": float(kl[0]), "teacher_forced_kl_mean": float(kl.mean()),
            "finite": bool(torch.isfinite(post).all() and torch.isfinite(tf).all()),
            "compress_seconds": compress_s, "suffix_prefill_seconds": suffix_s, "decode_seconds": decode_s,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "bytes_at_budget_point": policy.info["bytes"], "fraction_of_fp16": policy.info["bytes"] / fp16_bytes(L, nl, hkv, dh),
            "within_budget": method == "fp16" or policy.info["bytes"] <= budget,
            "live_cache_bytes_end": policy.live_bytes(cache),
            "shared_fp16_bytes_suffix_and_generated": fp16_bytes(len(suffix) + max(0, len(tokens) - 1), nl, hkv, dh),
            "evidence_retained": policy.evidence_retained(enc["evidence"]),
            **{k: v for k, v in policy.info.items() if k != "bytes"}
        }
        assert r["within_budget"], (case["id"], method, policy.info["bytes"], budget)
        assert r["finite"], (case["id"], method)

        if method.startswith("int4") or method == "hybrid_int2":
            r["reconstruction_rel_err"]= rel_err(cache, ctx, L)

        if keep_masks and method in ["evict_question", "hybrid_int2"]:
            #per context position: how many of the nl x hkv (layer, head) slots keep it in fp16
            if method == "evict_question":
                r["fp16_keep_count"]= torch.zeros(L, dtype=torch.long).scatter_add_(0, policy.index.flatten().cpu(), torch.ones(policy.index.numel(), dtype=torch.long)).tolist()
            else:
                r["fp16_keep_count"]= policy.fp16_mask.sum((0, 1)).cpu().tolist()
            r["evidence_token_positions"]= enc["evidence"]

        row["methods"][method]= r
        del cache

    return row, enc, ref


def main(args):
    cfg= load_config(args.config)
    set_seed(cfg["seed"])
    device= torch.device(args.device)
    print("Using device:", device)

    out_dir= os.path.join(args.output_dir, "smoke") if args.smoke else args.output_dir
    res_root= os.path.join(out_dir, "results") if args.smoke else args.results_dir
    res_dir= os.path.join(res_root, "task2", "kv")
    os.makedirs(os.path.join(res_dir, "cases"), exist_ok=True)

    data_root= resolve_data_root(cfg["data_repo"], args.data_root)
    cases= read_cases(os.path.join(data_root, cfg["data"]))
    assert len(cases) == cfg["num_cases"]

    if args.smoke:
        cases= [cases[0], next(c for c in cases if c["target_tokens"] == 4096)]

    max_new= 4 if args.smoke else cfg["max_new_tokens"]
    eos_ids= cfg["eos_token_ids"]
    tok= AutoTokenizer.from_pretrained(cfg["models"]["target"], revision=model_revision(cfg, "target"))
    model= load_model(cfg, device, args.smoke)
    check_ids= {next(c["id"] for c in cases if (c["target_tokens"], c["position"]) == cell) for cell in {(c["target_tokens"], c["position"]) for c in cases}}
    #a small fixed set of aligned generations with selection masks: the first case of each (length, position) at 2 distractors
    mask_ids= {c["id"] for c in cases if c["id"].endswith("-00") and c["distractors"] == 2} | ({cases[0]["id"]} if args.smoke else set())
    todo= [c for c in cases if not os.path.exists(os.path.join(res_dir, "cases", f"{c['id']}.json"))]
    print(f"{len(cases) - len(todo)} cases done, {len(todo)} to run")
    start= time.perf_counter()

    for k, case in enumerate(todo):
        row, enc, ref= run_case(model, tok, case, cfg, device, max_new, eos_ids, cfg["answer_regex"], case["id"] in mask_ids)

        if case["id"] in check_ids:
            row["c1_checks"]= c1_checks(model, enc, ref["post"], ref["tokens"], cfg, eos_ids, max_new)
            print(f"{case['id']} c1 checks: {row['c1_checks']}")
            assert row["c1_checks"]["passed"] or args.smoke, "c1 cache validity check failed, stopping before compressed results"

        write_json(os.path.join(res_dir, "cases", f"{case['id']}.json"), row)
        elapsed= time.perf_counter() - start
        print(f"[{k + 1}/{len(todo)}] {case['id']} L={row['context_tokens']} " + " ".join(f"{m}={int(v['correct'])}" for m, v in row["methods"].items())
              + f" | {elapsed:.0f}s, eta {elapsed / (k + 1) * (len(todo) - k - 1) / 60:.1f} min")

    rows= [json.load(open(os.path.join(res_dir, "cases", f))) for f in sorted(os.listdir(os.path.join(res_dir, "cases")))]
    info= run_info(cfg, device, next(model.parameters()).dtype, model_key="target")
    info.update({"smoke": args.smoke, "n_cases": len(rows), "methods": METHODS, "max_new_tokens": max_new,
                 "c1_checks_passed": all(r["c1_checks"]["passed"] for r in rows if "c1_checks" in r),
                 "accuracy": {m: sum(r["methods"][m]["correct"] for r in rows) / len(rows) for m in METHODS},
                 "labels": "quantized caches are fake-quantized (dequantized fp16 values in the cache); bytes are the packed-format count, see kv_bench for real packing"})
    write_json(os.path.join(res_dir, "summary.json"), info)
    print("accuracy:", info["accuracy"])


if __name__ == "__main__":
    parser= argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/task2/kv.yaml"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=default_output_dir()
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--smoke",
        action="store_true"
    )

    args= parser.parse_args()
    main(args)
