#task 3 steps 1-2 and 6 (handout pp.13-14): real-model sampler checks, isolated step costs, paired target-only vs
#exact speculative sampling on the 120 held-out prompts (64-token eos-aware cap), per-phase decomposition pass,
#fixed-64-token throughput diagnostic. one results dir per draft (original or a merged checkpoint)
import os
import time
import argparse

import yaml
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

from common.seed import set_seed
from common.timing import time_fn, sync
from common.io_utils import load_config, write_json, write_jsonl, resolve_data_root, default_output_dir, run_info
from task3.load import load_tokenizer, load_model, prompt_ids, check_agreement, eval_prompts
from task3.spec_sampler import HFRunner, speculative_generate, target_generate

CHECK_TOL= 2e-2   #max abs probability difference allowed between cached and fresh fp16 forwards


def load_cfgs(args):
    cfg= load_config(args.base_config)

    with open(args.config) as f:
        cfg.update(yaml.safe_load(f))

    return cfg


def run_seed(j, gamma, rep, method, kind):
    #ours: distinct, reproducible stream per (prompt, gamma, rep, method, pass)
    return 6304 + 10007 * j + 1000 * gamma + 100 * rep + {"target": 0, "spec": 1}[method] + {"main": 0, "decomp": 10, "diag": 20}[kind]


def fresh_probs(model, ids, device):
    with torch.no_grad():
        logits= model(torch.tensor([ids], device=device)).logits[0]

    return torch.softmax(logits.float(), dim=-1)


def real_model_checks(target, draft, tok, prompts, device, eos_ids):
    #split prefill == full forward; after speculative generation each cache + pending tokens == fresh forward
    out= {"split_prefill_max_abs_prob_diff": {}, "post_generation_max_abs_prob_diff": {}}

    for p in prompts:
        ids= prompt_ids(tok, p["text"])

        for name, model in [("target", target), ("draft", draft)]:
            r= HFRunner(model, device)
            r.feed(ids[:-4], keep=1)
            split= r.feed(ids[-4:])
            full= fresh_probs(model, ids, device)[-4:]
            out["split_prefill_max_abs_prob_diff"][f"{p['id']}/{name}"]= (split - full).abs().max().item()

        g= torch.Generator(device=device).manual_seed(0)
        t_run, d_run= HFRunner(target, device), HFRunner(draft, device)
        gen_out, stats= speculative_generate(t_run, d_run, ids, 4, 32, eos_ids, g)
        seq= ids + gen_out

        for name, run, model in [("target", t_run, target), ("draft", d_run, draft)]:
            cached= run.feed(seq[run.length():])[-1]
            out["post_generation_max_abs_prob_diff"][f"{p['id']}/{name}"]= (cached - fresh_probs(model, seq, device)[-1]).abs().max().item()

    worst= max(max(v.values()) for v in out.values())
    out["worst"]= worst
    out["tolerance"]= CHECK_TOL
    assert worst < CHECK_TOL, f"cache/prefill mismatch {worst}"

    return out


def step_costs(target, draft, tok, prompts, device, lengths, gammas, warmup, reps):
    #isolated decode-step cost at fixed context lengths: one draft step, one target step, one target verify of g+1
    seq= []

    for p in prompts:
        seq+= prompt_ids(tok, p["text"])

    while len(seq) < max(lengths) + max(gammas) + 1:
        seq+= seq

    out= {}

    for L in lengths:
        ctx= seq[:L]
        row= {}

        for name, model, widths in [("draft", draft, [1]), ("target", target, [1] + [g + 1 for g in gammas])]:
            r= HFRunner(model, device)
            r.feed(ctx[:-1], keep=1)

            for w in widths:
                toks= seq[L - 1:L - 1 + w]

                def step(_):
                    r.feed(toks, keep=w)
                    r.crop(L - 1)

                row[f"{name}_feed{w}"]= time_fn(step, device, warmup, reps)

        t1= row["target_feed1"]["median"]
        row["c"]= row["draft_feed1"]["median"] / t1
        row["v"]= {str(g): row[f"target_feed{g + 1}"]["median"] / t1 for g in gammas}
        out[str(L)]= row
        print(f"step costs L={L}: target {t1:.2f} ms, draft {row['draft_feed1']['median']:.2f} ms, c={row['c']:.3f}, v={row['v']}")

    return out


def generate_once(method, target, draft, ids, gamma, max_new, eos_ids, seed, device, timed):
    #fresh caches per generation; timed=True adds synchronized per-phase timers (decomposition pass)
    t_run= HFRunner(target, device)
    gen= torch.Generator(device=device).manual_seed(seed)
    timer= {"device": device, "phases": {}} if timed else None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    sync(device)
    start= time.perf_counter()

    if method == "spec":
        out, stats= speculative_generate(t_run, HFRunner(draft, device), ids, gamma, max_new, eos_ids, gen, timer=timer)
    else:
        out, stats= target_generate(t_run, ids, max_new, eos_ids, gen, timer=timer)

    sync(device)
    e2e= (time.perf_counter() - start) * 1000.0

    row= {k: v for k, v in stats.items() if k != "rounds"}
    row.update({
        "method": method,
        "e2e_ms": e2e,
        "tokens_per_s": 1000.0 * len(out) / e2e,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "output_ids": out,
        "seed": seed
    })

    if method == "spec":
        row["accepted_per_round"]= [r["accepted"] for r in stats["rounds"]]
        row["proposed_per_round"]= [r["proposed"] for r in stats["rounds"]]

    if timed:
        row["phases_ms"]= {k: 1000.0 * v for k, v in timer["phases"].items()}
        row["host_other_ms"]= e2e - sum(row["phases_ms"].values())

    return row


def run_pass(kind, prompts, cfg, target, draft, tok, device, eos_ids, gamma, rep, max_new, timed):
    rows= []

    for j, p in enumerate(prompts):
        ids= prompt_ids(tok, p["text"])
        #counterbalanced: alternate which method goes first
        order= ["target", "spec"] if (j + rep) % 2 == 0 else ["spec", "target"]

        for k, method in enumerate(order):
            row= generate_once(method, target, draft, ids, gamma, max_new, eos_ids, run_seed(j, gamma, rep, method, kind), device, timed)
            row.update({"prompt_id": p["id"], "split": p["split"], "domain": p["domain"], "gamma": gamma, "rep": rep,
                        "pass": kind, "order_position": k, "prompt_tokens": len(ids)})
            rows.append(row)

    return rows


def summarize(rows):
    spec= [r for r in rows if r["method"] == "spec"]
    tgt= {r["prompt_id"]: r for r in rows if r["method"] == "target"}
    ratio= sorted(r["tokens_per_s"] / tgt[r["prompt_id"]]["tokens_per_s"] for r in spec)

    return {
        "accepted_over_examined": sum(r["accepted"] for r in spec) / max(1, sum(r["examined"] for r in spec)),
        "accepted_over_proposed": sum(r["accepted"] for r in spec) / max(1, sum(r["proposed"] for r in spec)),
        "target_passes_per_token_spec": sum(r["target_passes"] for r in spec) / sum(r["n_out"] for r in spec),
        "median_throughput_ratio": ratio[len(ratio) // 2],
        "eos_fraction_spec": sum(r["stop"] == "eos" for r in spec) / len(spec)
    }


def load_merged(path, device):
    #a merged fp16 draft checkpoint saved by task3.merge
    return AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, attn_implementation="sdpa").to(device).eval()


def main(args):
    cfg= load_cfgs(args)
    set_seed(cfg["seed"])
    device= torch.device(args.device)
    print("Using device:", device)

    data_root= resolve_data_root(cfg["data_repo"], args.data_root)
    prompts= eval_prompts(cfg, data_root)
    max_new= cfg["speed_max_new_tokens"]
    reps= cfg["paired_reps"]
    lengths= [64, 256, 512, 1024]
    warmup, timing_reps= 3, 10

    if args.smoke:
        prompts= [prompts[0], prompts[60], prompts[90]]
        max_new, reps, lengths, warmup, timing_reps= 8, 1, [16, 32], 1, 2

    out_dir= os.path.join(args.output_dir, "smoke", "results", "task3", "spec_eval", args.draft_name) if args.smoke else os.path.join(args.results_dir, "task3", "spec_eval", args.draft_name)
    os.makedirs(out_dir, exist_ok=True)

    t0= time.perf_counter()
    tok= load_tokenizer(cfg, "target")
    target= load_model(cfg, "target", device, args.smoke)
    if args.adapter_path:
        #unmerged peft wrapper: only to quantify adapter overhead, not the primary speed comparison (p.14)
        draft= PeftModel.from_pretrained(load_model(cfg, "draft", device, args.smoke), args.adapter_path).eval()
    elif args.draft_path:
        draft= load_merged(args.draft_path, device)
    else:
        draft= load_model(cfg, "draft", device, args.smoke)
    load_seconds= time.perf_counter() - t0
    eos_ids= cfg["eos_token_ids"]
    dtype= next(target.parameters()).dtype

    info= run_info(cfg, device, dtype, model_key="target")
    info.update({"draft_name": args.draft_name, "draft_path": args.draft_path, "adapter_path": args.adapter_path, "smoke": args.smoke, "max_new_tokens": max_new,
                 "temperature": cfg["temperature"], "top_p": cfg["top_p"], "eos_ids": eos_ids})

    path= os.path.join(out_dir, "checks.json")

    if not os.path.exists(path):
        #before any timing (p.13 step 1)
        agree= check_agreement(tok, load_tokenizer(cfg, "draft"), target, draft, prompts)
        checks= real_model_checks(target, draft, tok, prompts[:3] if not args.smoke else prompts, device, eos_ids)
        write_json(path, {**info, "agreement": agree, **checks, "model_load_seconds": load_seconds})
        print(f"checks ok, worst prob diff {checks['worst']:.2e}")

    #warm setup: first generations after load, untimed for the results, cost recorded
    sync(device)
    w= time.perf_counter()

    for p in prompts[:2]:
        generate_once("spec", target, draft, prompt_ids(tok, p["text"]), 4, 16, eos_ids, 0, device, False)

    warm_seconds= time.perf_counter() - w

    path= os.path.join(out_dir, "step_costs.json")

    if not os.path.exists(path):
        costs= step_costs(target, draft, tok, prompts, device, lengths, cfg["gammas"], warmup, timing_reps)
        write_json(path, {**info, "lengths": lengths, "costs": costs, "warmup_reps": warmup, "timing_reps": timing_reps,
                          "model_load_seconds": load_seconds, "warm_setup_seconds": warm_seconds})

    jobs= [("main", g, r, max_new, False, prompts, eos_ids) for g in cfg["gammas"] for r in range(reps)]
    jobs+= [("decomp", g, 0, max_new, True, prompts, eos_ids) for g in cfg["gammas"]]
    diag= [p for p in prompts if p["domain"] == "math"][:cfg["throughput_diag"]["n_math_prompts"]]
    jobs+= [("diag", g, r, cfg["throughput_diag"]["fixed_new_tokens"] if not args.smoke else max_new, False, diag, []) for g in cfg["gammas"] for r in range(reps)]

    jobs= [j for j in jobs if j[0] in args.passes.split(",")]

    for kind, gamma, rep, n_new, timed, ps, eos in jobs:
        path= os.path.join(out_dir, f"{kind}_g{gamma}_r{rep}.jsonl")

        if os.path.exists(path):
            print(f"{path} exists, skipping")
            continue

        start= time.perf_counter()
        rows= run_pass(kind, ps, cfg, target, draft, tok, device, eos, gamma, rep, n_new, timed)
        write_jsonl(path, rows)
        s= summarize(rows)
        print(
            f"{args.draft_name} {kind} gamma={gamma} rep={rep}: "
            f"accepted/examined={s['accepted_over_examined']:.3f}, "
            f"target passes/token={s['target_passes_per_token_spec']:.3f}, "
            f"median tok/s ratio={s['median_throughput_ratio']:.3f}, "
            f"{time.perf_counter() - start:.0f}s"
        )

    write_json(os.path.join(out_dir, "run_info.json"), {**info, "model_load_seconds": load_seconds, "warm_setup_seconds": warm_seconds})


if __name__ == "__main__":
    parser= argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/task3/spec_eval.yaml"
    )
    parser.add_argument(
        "--base_config",
        type=str,
        default="configs/task3/base.yaml"
    )
    parser.add_argument(
        "--draft_name",
        type=str,
        default="original"
    )
    parser.add_argument(
        "--draft_path",
        type=str,
        default=None
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None
    )
    parser.add_argument(
        "--passes",
        type=str,
        default="main,decomp,diag"
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
