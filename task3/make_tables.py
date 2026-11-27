#task 3 tables and figures (handout pp.13-15, required evidence). cpu only, reads results/task3, writes
#results/task3/tables/*.csv, results/task3/summary.csv and results/task3/figures/*.png. numbers only, no interpretation.
#bootstrap: paired over prompts (per-prompt means over reps / positions), 2,000 resamples, seed 6304
import os
import csv
import json
import gzip
import argparse
from collections import defaultdict

import numpy as np

from common.bootstrap import paired_bootstrap

DRAFTS= ["original", "off_policy_s6304", "off_policy_s6305", "on_policy_s6304", "on_policy_s6305"]
SPLITS= ["all", "math", "familiar", "shifted"]
POS_GROUPS= {"all": (1, 192), "1-16": (1, 16), "17-32": (17, 32), "33-64": (33, 64), "65-192": (65, 192), "short_1-64": (1, 64)}
METRICS= ["fkl", "rkl", "tv", "greedy_agree"]
#reference palette slots in fixed order (dataviz skill): original blue, off-policy orange, on-policy aqua, target yellow
COLORS= {"original": "#2a78d6", "off_policy": "#eb6834", "on_policy": "#1baf7a", "target": "#eda100"}
LEN_BINS= [(1, 16), (17, 32), (33, 63), (64, 64)]
STANDALONE_MATH_BINS= [(1, 128), (129, 256), (257, 511), (512, 512)]


def method_of(name):
    return "original" if name == "original" else name.rsplit("_s", 1)[0]


def style(name):
    m= method_of(name)
    return {"color": COLORS.get(m, COLORS["target"]), "linestyle": "--" if name.endswith("6305") else "-"}


def read_jsonl(path):
    op= gzip.open if path.endswith(".gz") else open

    with op(path, "rt") as f:
        return [json.loads(line) for line in f]


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys= list(dict.fromkeys(k for r in rows for k in r))

    with open(path, "w", newline="") as f:
        w= csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {path} ({len(rows)} rows)")


def rnd(x, k=4):
    return round(float(x), k) if x is not None else None


def in_split(r, split):
    return split == "all" or r["split"] == split


def per_prompt(rows, key):
    d= defaultdict(list)

    for r in rows:
        d[r.get("prompt_id", r.get("id"))].append(r[key])   #standalone rows use "id"

    return {k: float(np.mean(v)) for k, v in d.items()}


#---------------------------------------------------------------- alignment (step 5)
def load_align(root):
    out= {}

    for f in sorted(os.listdir(os.path.join(root, "align"))):
        if f.endswith(".jsonl.gz"):
            out[f[:-len(".jsonl.gz")]]= read_jsonl(os.path.join(root, "align", f))

    return out


def alignment_tables(align):
    rows= []

    for bank, recs in align.items():
        for draft in DRAFTS:
            for split in SPLITS:
                for g, (lo, hi) in POS_GROUPS.items():
                    rs= [r for r in recs if r["draft"] == draft and in_split(r, split) and lo <= r["position"] <= hi]

                    if not rs:
                        continue

                    row= {"bank": bank, "draft": draft, "method": method_of(draft), "split": split, "positions": g, "n_records": len(rs),
                          "n_prompts": len({r["prompt_id"] for r in rs})}
                    row.update({m: rnd(np.mean([r[m] for r in rs])) for m in METRICS})
                    row["expected_accept_1_minus_tv"]= rnd(1 - row["tv"])
                    rows.append(row)

    #method level: mean over the two training seeds and their half-range (seed dispersion)
    methods= []
    key= lambda r: (r["bank"], r["split"], r["positions"])
    groups= defaultdict(dict)

    for r in rows:
        groups[key(r)][r["draft"]]= r

    for (bank, split, g), by in groups.items():
        for m in ["off_policy", "on_policy"]:
            seeds= [by[d] for d in by if method_of(d) == m]

            if len(seeds) == 2:
                methods.append({"bank": bank, "split": split, "positions": g, "method": m,
                                **{f"{k}_mean": rnd(np.mean([s[k] for s in seeds])) for k in METRICS},
                                **{f"{k}_seed_half_range": rnd(abs(seeds[0][k] - seeds[1][k]) / 2) for k in METRICS},
                                "original_fkl": by["original"]["fkl"], "original_tv": by["original"]["tv"]})

    return rows, methods


def alignment_vs_original(align):
    #paired over prompts: student minus original on the same records
    rows= []

    for bank, recs in align.items():
        for split in SPLITS:
            base= {m: per_prompt([r for r in recs if r["draft"] == "original" and in_split(r, split)], m) for m in ["fkl", "tv"]}

            for draft in DRAFTS[1:]:
                for m in ["fkl", "tv"]:
                    other= per_prompt([r for r in recs if r["draft"] == draft and in_split(r, split)], m)
                    ids= sorted(set(base[m]) & set(other))
                    b= paired_bootstrap([other[i] - base[m][i] for i in ids])
                    rows.append({"bank": bank, "split": split, "draft": draft, "metric": m, **{k: rnd(v) for k, v in b.items()}})

    return rows


def consistency(align, root):
    #target-generated vs exact-speculative bank: same prefix law expected up to sampling noise (p.14)
    a, b= align.get("teacher_pure"), align.get("exact_spec_original_gamma4")
    rows= []

    if a is None or b is None:
        return rows

    for draft in DRAFTS:
        for m in METRICS:
            for split in SPLITS:
                pa= per_prompt([r for r in a if r["draft"] == draft and in_split(r, split)], m)
                pb= per_prompt([r for r in b if r["draft"] == draft and in_split(r, split)], m)
                ids= sorted(set(pa) & set(pb))
                diffs= [pa[i] - pb[i] for i in ids]
                main= paired_bootstrap(diffs)
                corr= paired_bootstrap(diffs, level=1 - 0.05 / 3)
                rows.append({"draft": draft, "metric": m, "split": split, "n_prompts": len(ids),
                             "teacher_pure": rnd(np.mean([pa[i] for i in ids])), "exact_spec": rnd(np.mean([pb[i] for i in ids])),
                             "diff": rnd(main["mean_diff"]), "ci95_low": rnd(main["ci_low"]), "ci95_high": rnd(main["ci_high"]),
                             "p_two_sided": rnd(main["p_two_sided"]),
                             "ci_bonferroni3_low": rnd(corr["ci_low"]), "ci_bonferroni3_high": rnd(corr["ci_high"])})

    for bank in ["teacher_pure", "exact_spec_original_gamma4"]:
        info= json.load(open(os.path.join(root, "banks", f"{bank}_info.json")))
        rows.append({"draft": "bank_stats", "metric": bank, "split": "all", "n_prompts": info["n_prompts"],
                     "teacher_pure": rnd(info["mean_response_length"], 2), "exact_spec": rnd(info["eos_fraction"]),
                     "diff": None, "p_two_sided": None})

    return rows


def disagreements(align, tok):
    #fixed rule: on the teacher_pure bank, per split, the 3 records with the largest original-draft tv among those where
    #the original's greedy token differs from the target's; every draft's values on the same record
    recs= align.get("teacher_pure", [])
    by= defaultdict(dict)

    for r in recs:
        by[(r["prompt_id"], r["position"])][r["draft"]]= r

    out= []

    for split in ["math", "familiar", "shifted"]:
        cands= [v for v in by.values() if "original" in v and v["original"]["split"] == split and not v["original"]["greedy_agree"]]
        cands.sort(key=lambda v: -v["original"]["tv"])

        for v in cands[:3]:
            o= v["original"]
            row= {"split": split, "prompt_id": o["prompt_id"], "position": o["position"], "target_top": o["target_top"],
                  "target_top_text": tok.decode([o["target_top"]]) if tok else None}

            for d in DRAFTS:
                if d in v:
                    row[f"{d}_tv"]= rnd(v[d]["tv"])
                    row[f"{d}_top"]= v[d]["draft_top"]
                    row[f"{d}_top_text"]= tok.decode([v[d]["draft_top"]]) if tok else None
            out.append(row)

    return out


#---------------------------------------------------------------- speculative decoding (steps 2, 6, 7)
def load_pass(root, draft, kind):
    d= os.path.join(root, "spec_eval", draft)
    rows= []

    if not os.path.isdir(d):
        return rows

    for f in sorted(os.listdir(d)):
        if f.startswith(kind + "_g") and f.endswith(".jsonl"):
            rows+= read_jsonl(os.path.join(d, f))

    return rows


def pair_rows(rows, gamma):
    #(spec, target) pairs for the same prompt, gamma and rep
    tgt= {(r["prompt_id"], r["rep"]): r for r in rows if r["method"] == "target" and r["gamma"] == gamma}

    return [(r, tgt[(r["prompt_id"], r["rep"])]) for r in rows if r["method"] == "spec" and r["gamma"] == gamma and (r["prompt_id"], r["rep"]) in tgt]


def spec_stats(pairs):
    s= [p[0] for p in pairs]
    t= [p[1] for p in pairs]
    ratios= np.array([a["tokens_per_s"] / b["tokens_per_s"] for a, b in pairs])
    rounds= sum(len(r.get("accepted_per_round", [])) for r in s)

    return {
        "n_runs": len(s), "n_prompts": len({r["prompt_id"] for r in s}),
        "accepted_over_examined": rnd(sum(r["accepted"] for r in s) / max(1, sum(r["examined"] for r in s))),
        "accepted_over_proposed": rnd(sum(r["accepted"] for r in s) / max(1, sum(r["proposed"] for r in s))),
        "mean_accepted_prefix_per_round": rnd(sum(sum(r.get("accepted_per_round", [])) for r in s) / max(1, rounds)),
        "target_passes_per_token_spec": rnd(sum(r["target_passes"] for r in s) / sum(r["n_out"] for r in s)),
        "target_passes_per_token_target_only": rnd(sum(r["target_passes"] for r in t) / sum(r["n_out"] for r in t)),
        "draft_passes_per_token": rnd(sum(r["draft_passes"] for r in s) / sum(r["n_out"] for r in s)),
        "mean_output_tokens_spec": rnd(np.mean([r["n_out"] for r in s]), 2),
        "eos_fraction_spec": rnd(np.mean([r["stop"] == "eos" for r in s])),
        "eos_fraction_target": rnd(np.mean([r["stop"] == "eos" for r in t])),
        "truncated_fraction_spec": rnd(np.mean([r["stop"] == "length" for r in s])),
        "mean_latency_ms_spec": rnd(np.mean([r["e2e_ms"] for r in s]), 1),
        "mean_latency_ms_target": rnd(np.mean([r["e2e_ms"] for r in t]), 1),
        "median_tok_s_spec": rnd(np.median([r["tokens_per_s"] for r in s]), 2),
        "median_tok_s_target": rnd(np.median([r["tokens_per_s"] for r in t]), 2),
        "median_throughput_ratio": rnd(np.median(ratios)),
        "geomean_throughput_ratio": rnd(np.exp(np.mean(np.log(ratios)))),
        "peak_allocated_mib_spec": rnd(max(r["peak_allocated_bytes"] for r in s) / 2 ** 20, 1),
        "peak_allocated_mib_target": rnd(max(r["peak_allocated_bytes"] for r in t) / 2 ** 20, 1)
    }


def log_ratio_by_prompt(pairs):
    d= defaultdict(list)

    for a, b in pairs:
        d[a["prompt_id"]].append(np.log(a["tokens_per_s"] / b["tokens_per_s"]))

    return {k: float(np.mean(v)) for k, v in d.items()}


def alpha_by_prompt(pairs):
    acc, ex= defaultdict(int), defaultdict(int)

    for a, _ in pairs:
        acc[a["prompt_id"]]+= a["accepted"]
        ex[a["prompt_id"]]+= a["examined"]

    return {k: acc[k] / max(1, ex[k]) for k in acc}


def spec_tables(root, drafts):
    main, bins, ci, decomp= [], [], [], []

    for draft in drafts:
        rows= load_pass(root, draft, "main")

        for gamma in [2, 4]:
            pairs= pair_rows(rows, gamma)

            if not pairs:
                continue

            for split in SPLITS:
                ps= [p for p in pairs if in_split(p[0], split)]
                main.append({"draft": draft, "gamma": gamma, "split": split, **spec_stats(ps)})

            for lo, hi in LEN_BINS:
                ps= [p for p in pairs if lo <= p[0]["n_out"] <= hi]
                if ps:
                    bins.append({"draft": draft, "gamma": gamma, "output_len_bin": f"{lo}-{hi}", **spec_stats(ps)})

            #paired bootstrap over prompts: log throughput ratio (spec vs adjacent target-only) -> geometric-mean speedup
            lr= log_ratio_by_prompt(pairs)
            b= paired_bootstrap(list(lr.values()))
            ci.append({"draft": draft, "gamma": gamma, "comparison": "spec_vs_target_only", "statistic": "geomean_tok_s_ratio",
                       "estimate": rnd(np.exp(b["mean_diff"])), "ci_low": rnd(np.exp(b["ci_low"])), "ci_high": rnd(np.exp(b["ci_high"])),
                       "n_prompts": b["n_units"]})

            if draft != "original":
                base= pair_rows(load_pass(root, "original", "main"), gamma)

                for stat, fn in [("log_tok_s_ratio", log_ratio_by_prompt), ("accepted_over_examined", alpha_by_prompt)]:
                    x, y= fn(pairs), fn(base)
                    ids= sorted(set(x) & set(y))
                    b= paired_bootstrap([x[i] - y[i] for i in ids])
                    ci.append({"draft": draft, "gamma": gamma, "comparison": "draft_minus_original", "statistic": stat,
                               "estimate": rnd(b["mean_diff"]), "ci_low": rnd(b["ci_low"]), "ci_high": rnd(b["ci_high"]), "n_prompts": len(ids)})

        drows= load_pass(root, draft, "decomp")

        for gamma in [2, 4]:
            for method in ["spec", "target"]:
                rs= [r for r in drows if r["gamma"] == gamma and r["method"] == method]

                if not rs:
                    continue

                n= sum(r["n_out"] for r in rs)
                phases= sorted({k for r in rs for k in r.get("phases_ms", {})})
                row= {"draft": draft, "gamma": gamma, "method": method, "n_runs": len(rs), "output_tokens": n}
                row.update({f"{p}_ms_per_token": rnd(sum(r["phases_ms"].get(p, 0) for r in rs) / n) for p in phases})
                row["host_other_ms_per_token"]= rnd(sum(r.get("host_other_ms", 0) for r in rs) / n)
                row["e2e_ms_per_token"]= rnd(sum(r["e2e_ms"] for r in rs) / n)
                decomp.append(row)

    return main, bins, ci, decomp


def expected_tokens(alpha, gamma):
    #expected tokens per round (accepted prefix + residual/bonus) for i.i.d. per-token acceptance alpha
    return gamma + 1 if alpha >= 1 else (1 - alpha ** (gamma + 1)) / (1 - alpha)


def break_even_alpha(gamma, cost):
    #smallest alpha with expected_tokens(alpha) = cost (round cost in target-step units); None if gamma + 1 <= cost
    if gamma + 1 <= cost:
        return None

    lo, hi= 0.0, 1.0

    for _ in range(60):
        mid= (lo + hi) / 2
        lo, hi= (mid, hi) if expected_tokens(mid, gamma) < cost else (lo, mid)

    return (lo + hi) / 2


def speedup_model(root, main_rows, align, context="256"):
    #c and v_gamma from the isolated step costs of the original draft run (same backbone shape for every draft),
    #alpha from (a) measured accepted/examined and (b) mean 1 - tv on the exact-speculative bank, positions 1-64
    costs= json.load(open(os.path.join(root, "spec_eval", "original", "step_costs.json")))["costs"][context]
    c= costs["c"]
    rows= []

    for r in main_rows:
        if r["split"] != "all" or r["draft"] not in DRAFTS:
            continue

        g= r["gamma"]
        v= costs["v"][str(g)]
        cost= g * c + v
        bank= [x for x in align.get("exact_spec_original_gamma4", []) if x["draft"] == r["draft"] and x["position"] <= 64]
        a_bank= 1 - float(np.mean([x["tv"] for x in bank])) if bank else None
        a_meas= r["accepted_over_examined"]
        rows.append({"draft": r["draft"], "gamma": g, "context_tokens_for_costs": int(context), "c": rnd(c), "v_gamma": rnd(v),
                     "round_cost_target_steps": rnd(cost), "alpha_measured": a_meas, "alpha_bank_1_minus_tv": rnd(a_bank),
                     "S_hat_from_measured_alpha": rnd(expected_tokens(a_meas, g) / cost),
                     "S_hat_from_bank_alpha": rnd(expected_tokens(a_bank, g) / cost) if a_bank is not None else None,
                     "S_measured_geomean": r["geomean_throughput_ratio"], "S_measured_median": r["median_throughput_ratio"],
                     "break_even_alpha": rnd(break_even_alpha(g, cost)) if break_even_alpha(g, cost) is not None else "none (gamma+1 <= round cost)"})

    return rows


#---------------------------------------------------------------- standalone (step 7) and costs
def standalone_tables(root):
    models= [m for m in ["target", "original"] + DRAFTS[1:] if os.path.isdir(os.path.join(root, "standalone", m))]
    data= {m: [r for f in sorted(os.listdir(os.path.join(root, "standalone", m))) if f.endswith(".jsonl")
               for r in read_jsonl(os.path.join(root, "standalone", m, f))] for m in models}
    rows, bins, ci= [], [], []

    for m in models:
        for rep in ["pooled", 0, 1]:
            for split in ["math", "familiar", "shifted"]:
                rs= [r for r in data[m] if r["split"] == split and (rep == "pooled" or r["rep"] == rep)]
                row= {"model": m, "rep": rep, "split": split, "n": len(rs), "accuracy_strict": rnd(np.mean([r["correct"] for r in rs])),
                      "truncated": rnd(np.mean([r["truncated"] for r in rs])), "mean_tokens": rnd(np.mean([r["n_out"] for r in rs]), 1)}

                if split == "math":
                    row["extractable"]= rnd(np.mean([r["extractable"] for r in rs]))
                    row["correct_and_not_truncated"]= rnd(np.mean([r["correct"] and not r["truncated"] for r in rs]))
                else:
                    row["compliant"]= rnd(np.mean([r["compliant"] for r in rs]))
                    row["accuracy_lenient"]= rnd(np.mean([r["correct_lenient"] for r in rs]))
                rows.append(row)

        for lo, hi in STANDALONE_MATH_BINS:
            rs= [r for r in data[m] if r["split"] == "math" and lo <= r["n_out"] <= hi]
            bins.append({"model": m, "math_output_len_bin": f"{lo}-{hi}", "n": len(rs),
                         "accuracy": rnd(np.mean([r["correct"] for r in rs])) if rs else None})

    def prompt_mean(m, split, key):
        return per_prompt([r for r in data[m] if r["split"] == split], key) if m in data else {}

    for m in [x for x in models if x != "original"]:
        for split, key in [("math", "correct"), ("familiar", "correct_lenient"), ("shifted", "correct_lenient"), ("familiar", "correct"), ("shifted", "correct")]:
            x, y= prompt_mean(m, split, key), prompt_mean("original", split, key)
            ids= sorted(set(x) & set(y))
            b= paired_bootstrap([x[i] - y[i] for i in ids])
            ci.append({"model": m, "minus": "original", "split": split, "metric": key, **{k: rnd(v) for k, v in b.items()}})

    return rows, bins, ci


def distill_tables(root, main_rows):
    rows, deploy= [], []
    pool= json.load(open(os.path.join(root, "teacher_pool_info.json")))

    for name in DRAFTS[1:]:
        d= os.path.join(root, "distill", name)
        s= json.load(open(os.path.join(d, "summary.json")))
        log= read_jsonl(os.path.join(d, "train_log.jsonl.gz"))
        kl= [r["kl_per_token"] for r in log if r.get("kl_per_token") is not None]
        mg= json.load(open(os.path.join(root, "merge", f"{name}.json")))
        t= s["times_seconds"]
        pool_s= pool["teacher_generation_seconds"] if s["method"] == "off_policy" else 0.0
        upfront= s["wall_clock_seconds"] + pool_s + mg["merge_seconds"]
        rows.append({"run": name, "method": s["method"], "seed": s["seed"], "updates": s["n_updates"], "skipped_updates": s["skipped_updates"],
                     "presentations": s["n_presentations"], "truncated_presentations": s["truncated_presentations"],
                     "scored_math": s["scored_tokens"]["math"], "scored_short": s["scored_tokens"]["short"],
                     "kl_first10_updates": rnd(np.mean(kl[:10])), "kl_last10_updates": rnd(np.mean(kl[-10:])),
                     "teacher_pool_generation_s": rnd(pool_s, 1), **{f"{k}_s": rnd(v, 1) for k, v in t.items()},
                     "training_wall_clock_s": rnd(s["wall_clock_seconds"], 1), "merge_s": rnd(mg["merge_seconds"], 3),
                     "merge_max_abs_logit_error": rnd(mg["max_abs_logit_error"]), "upfront_total_s": rnd(upfront, 1),
                     "student_params": s["student_params"], "adapter_params": s["adapter_params"],
                     "peak_allocated_mib": rnd(s["peak_allocated_bytes"] / 2 ** 20, 1)})

        for r in main_rows:
            if r["draft"] == name and r["split"] == "all":
                saving_ms= r["mean_latency_ms_target"] - r["mean_latency_ms_spec"]
                base= next(x for x in main_rows if x["draft"] == "original" and x["split"] == "all" and x["gamma"] == r["gamma"])
                vs_orig= base["mean_latency_ms_spec"] - r["mean_latency_ms_spec"]
                deploy.append({"draft": name, "gamma": r["gamma"], "upfront_s": rnd(upfront, 1),
                               "saving_ms_per_request_vs_target_only": rnd(saving_ms, 1),
                               "requests_to_break_even_vs_target_only": int(np.ceil(upfront * 1000 / saving_ms)) if saving_ms > 0 else "never (speculative slower)",
                               "saving_ms_per_request_vs_original_draft": rnd(vs_orig, 1),
                               "requests_to_break_even_vs_original_draft": int(np.ceil(upfront * 1000 / vs_orig)) if vs_orig > 0 else "never (not faster than original draft)",
                               "note": "64-token eos-aware requests, mean latency over 120 prompts x 2 reps; off-policy upfront includes the full teacher pool"})

    return rows, deploy


#---------------------------------------------------------------- figures
def figures(root, align_rows, main_rows, model_rows, sa_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out= os.path.join(root, "figures")
    os.makedirs(out, exist_ok=True)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e5e5e3",
                         "grid.linewidth": 0.6, "axes.edgecolor": "#52514e", "font.size": 9, "lines.linewidth": 2})

    #fig 1: forward kl by response-position bin on each bank (all splits)
    banks= sorted({r["bank"] for r in align_rows})
    bins= ["1-16", "17-32", "33-64", "65-192"]
    fig, axes= plt.subplots(2, (len(banks) + 1) // 2, figsize=(3.2 * ((len(banks) + 1) // 2), 5.6), sharey=True)

    for ax, bank in zip(axes.flat, banks):
        for d in DRAFTS:
            ys= [next((r["fkl"] for r in align_rows if r["bank"] == bank and r["draft"] == d and r["split"] == "all" and r["positions"] == b), np.nan) for b in bins]
            ax.plot(bins, ys, marker="o", markersize=4, label=d, **style(d))
        ax.set_title(bank, fontsize=8)
    for ax in axes.flat[len(banks):]:
        ax.axis("off")
    axes.flat[0].set_ylabel("forward KL(target || draft)")
    axes.flat[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "alignment_fkl_by_position.png"), dpi=200)
    plt.close(fig)

    #fig 2: distillation kl per token vs teacher-scored tokens
    fig, ax= plt.subplots(figsize=(5, 3.2))

    for name in DRAFTS[1:]:
        log= read_jsonl(os.path.join(root, "distill", name, "train_log.jsonl.gz"))
        x= np.cumsum([r["supervised_tokens"] for r in log])
        y= np.array([r["kl_per_token"] for r in log])
        k= 10
        ys= np.convolve(y, np.ones(k) / k, mode="valid")
        ax.plot(x[k - 1:], ys, label=name, **style(name))
    ax.set_xlabel("teacher-scored tokens")
    ax.set_ylabel("KL per token (10-update mean)")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "distill_kl_curves.png"), dpi=200)
    plt.close(fig)

    #fig 3: measured vs predicted speedup
    fig, ax= plt.subplots(figsize=(4.2, 4))

    for r in model_rows:
        mk= "o" if r["gamma"] == 2 else "s"
        col= style(r["draft"])["color"]
        hollow= r["draft"].endswith("6305")
        ax.scatter(r["S_hat_from_measured_alpha"], r["S_measured_geomean"], marker=mk, s=48, facecolors="#fcfcfb" if hollow else col,
                   edgecolors=col, linewidths=1.5, label=f"{r['draft']} γ={r['gamma']}")
    lim= [0.7, 1.15]
    ax.plot(lim, lim, color="#52514e", linewidth=1, linestyle=":")
    ax.axhline(1.0, color="#52514e", linewidth=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("predicted Ŝ (measured α, isolated c and v_γ)")
    ax.set_ylabel("measured S (geomean tok/s ratio)")
    ax.legend(fontsize=6, frameon=False, ncol=1)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "speedup_measured_vs_predicted.png"), dpi=200)
    plt.close(fig)

    #fig 4: standalone strict accuracy by split (pooled reps)
    models= list(dict.fromkeys(r["model"] for r in sa_rows))
    splits= ["math", "familiar", "shifted"]
    fig, ax= plt.subplots(figsize=(6, 3.2))
    w= 0.8 / len(models)

    for i, m in enumerate(models):
        ys= [next(r["accuracy_strict"] for r in sa_rows if r["model"] == m and r["rep"] == "pooled" and r["split"] == s) for s in splits]
        st= style(m) if m != "target" else {"color": COLORS["target"]}
        ax.bar(np.arange(3) + (i - (len(models) - 1) / 2) * w, ys, width=w * 0.9, color=st["color"],
               hatch="//" if m.endswith("6305") else None, edgecolor="#fcfcfb", label=m)
    ax.set_xticks(range(3), splits)
    ax.set_ylabel("strict accuracy (2 reps pooled)")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "standalone_accuracy.png"), dpi=200)
    plt.close(fig)
    print(f"figures -> {out}")


def main(args):
    root= os.path.join(args.results_dir, "task3")
    tables= os.path.join(root, "tables")

    try:
        from transformers import AutoTokenizer
        tok= AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", local_files_only=True)
    except Exception:
        tok= None

    align= load_align(root)
    align_rows, align_methods= alignment_tables(align)
    write_csv(os.path.join(tables, "alignment.csv"), align_rows)
    write_csv(os.path.join(tables, "alignment_methods_seed_dispersion.csv"), align_methods)
    write_csv(os.path.join(tables, "alignment_vs_original_bootstrap.csv"), alignment_vs_original(align))
    write_csv(os.path.join(tables, "bank_consistency_teacher_vs_exact_spec.csv"), consistency(align, root))
    write_csv(os.path.join(tables, "prefix_disagreements.csv"), disagreements(align, tok))

    drafts= [d for d in DRAFTS + ["unmerged_off_policy_s6304"] if os.path.isdir(os.path.join(root, "spec_eval", d))]
    main_rows, bin_rows, ci_rows, decomp_rows= spec_tables(root, drafts)
    write_csv(os.path.join(tables, "spec_main.csv"), main_rows)
    write_csv(os.path.join(tables, "spec_by_output_length.csv"), bin_rows)
    write_csv(os.path.join(tables, "spec_bootstrap.csv"), ci_rows)
    write_csv(os.path.join(tables, "spec_decomposition.csv"), decomp_rows)
    model_rows= speedup_model(root, main_rows, align)
    write_csv(os.path.join(tables, "speedup_model.csv"), model_rows)

    sa_rows, sa_bins, sa_ci= standalone_tables(root)
    write_csv(os.path.join(tables, "standalone.csv"), sa_rows)
    write_csv(os.path.join(tables, "standalone_math_by_length.csv"), sa_bins)
    write_csv(os.path.join(tables, "standalone_bootstrap.csv"), sa_ci)

    d_rows, deploy= distill_tables(root, main_rows)
    write_csv(os.path.join(tables, "distillation_costs.csv"), d_rows)
    write_csv(os.path.join(tables, "deployment_break_even.csv"), deploy)

    merge= [{"run": f[:-5], **{k: v for k, v in json.load(open(os.path.join(root, "merge", f))).items() if k in ("max_abs_logit_error", "merge_seconds", "merged_params")}}
            for f in sorted(os.listdir(os.path.join(root, "merge")))]
    write_csv(os.path.join(tables, "merge_check.csv"), merge)

    summary= [{"draft": r["draft"], "gamma": r["gamma"], "alpha_measured": r["alpha_measured"], "alpha_bank": r["alpha_bank_1_minus_tv"],
               "S_hat": r["S_hat_from_measured_alpha"], "S_measured": r["S_measured_geomean"], "break_even_alpha": r["break_even_alpha"]}
              for r in model_rows]
    write_csv(os.path.join(root, "summary.csv"), summary)

    if not args.no_figures:
        figures(root, align_rows, main_rows, model_rows, sa_rows)


if __name__ == "__main__":
    parser= argparse.ArgumentParser()

    parser.add_argument(
        "--results_dir",
        type=str,
        default="results"
    )
    parser.add_argument(
        "--no_figures",
        action="store_true"
    )

    args= parser.parse_args()
    main(args)
