#task 1 tables and figures (handout pp.2-5, required evidence). cpu only, reads results/task1 and splits/task1, writes
#results/task1/tables/*.csv, results/task1/summary.csv and results/task1/figures/*.png. numbers only, no interpretation.
#bootstrap (2,000 resamples, seed 6304, paired): hellaswag over examples; ppl over windows (resample windows, pool
#nll sums and token counts, then exp), handout p.5.
import os
import csv
import json
import gzip
import argparse
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.bootstrap import paired_bootstrap, bootstrap_stat

PROJS= ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
N_SCORED= 50000
N_DIAG= 512
#paired comparisons: (a, b) -> a minus b
PAIRS= [
    ("p3_nonuniform_s0.5", "p2_wanda_s0.5"),
    ("p3_nonuniform_s0.75", "p2_wanda_s0.75"),
    ("q3_mixed_budget_b4", "q2_awq_b4"),
    ("q3_mixed_budget_b3", "q2_awq_b3"),
    ("p2_wanda_s0.5", "p1_magnitude_s0.5"),
    ("q2_awq_b4", "q1_uniform_b4"),
    ("q2_awq_b3", "q1_uniform_b3"),
    ("p2_wanda_2of4_s0.5", "p2_wanda_s0.5"),
]
#reference palette slots in fixed order (dataviz skill)
COLORS= {"baseline": "#222222", "p1": "#2a78d6", "p2": "#eb6834", "p3": "#1baf7a", "p2_2of4": "#eda100",
         "q1": "#2a78d6", "q2": "#eb6834", "q3": "#1baf7a"}
MARKERS= {"p1": "o", "p2": "s", "p3": "^", "p2_2of4": "D", "q1": "o", "q2": "s", "q3": "^"}


def read_json(path):
    with open(path) as f:
        return json.load(f)


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


def save_fig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#e6e6e6", linewidth=0.6)
    ax.set_axisbelow(True)


def family(name):
    if name == "baseline":
        return "baseline", None, None

    if name.startswith("p2_wanda_2of4"):
        return "p2_2of4", float(name.rsplit("_s", 1)[1]), None

    fam= name.split("_")[0]

    if fam in ["p1", "p2", "p3"]:
        return fam, float(name.rsplit("_s", 1)[1]), None

    return fam, None, int(name.rsplit("_b", 1)[1])


#---------------------------------------------------------------- loading
def load_conditions(root):
    conds= {}

    for name in sorted(os.listdir(root)):
        d= os.path.join(root, name)

        if not os.path.exists(os.path.join(d, "condition.json")):
            continue

        c= {"name": name, "dir": d}

        for f in ["condition", "resources", "wikitext2", "c4_val", "hellaswag"]:
            c[f]= read_json(os.path.join(d, f + ".json"))

        c["layers"]= read_jsonl(os.path.join(d, "compression_layers.jsonl")) if os.path.exists(
            os.path.join(d, "compression_layers.jsonl")) else []
        c["wt2_ex"]= {r["window"]: r for r in read_jsonl(os.path.join(d, "wikitext2_examples.jsonl"))}
        c["c4_ex"]= {r["window"]: r for r in read_jsonl(os.path.join(d, "c4_val_examples.jsonl"))}
        c["hs_ex"]= {r["order"]: r for r in read_jsonl(os.path.join(d, "hellaswag_examples.jsonl"))}
        conds[name]= c

    #identical scoring across conditions (handout p.3)
    ref= conds["baseline"]

    for c in conds.values():
        for suite in ["wikitext2", "c4_val"]:
            assert c[suite]["n_scored"] == N_SCORED, f"{c['name']} {suite} scored {c[suite]['n_scored']}"

        for key in ["wt2_ex", "c4_ex"]:
            assert {w: r["n_scored"] for w, r in c[key].items()} == {w: r["n_scored"] for w, r in ref[key].items()}, \
                f"{c['name']} {key} windows differ from baseline"

    return conds


#---------------------------------------------------------------- main table
def hs_acc(ex, orders):
    acc= np.mean([ex[o]["pred"] == ex[o]["gold"] for o in orders])
    acc_norm= np.mean([ex[o]["pred_norm"] == ex[o]["gold"] for o in orders])

    return rnd(acc), rnd(acc_norm)


def main_table(conds):
    rows= []

    for name, c in conds.items():
        fam, sp, bits= family(name)
        st= dict(c["condition"].get("storage", {}))
        st.update(c["condition"].get("storage_extra", {}))
        lat= c["resources"]["latency"]
        errs= [r["rel_output_error"] for r in c["layers"]]
        diag= [o for o in range(N_DIAG)]
        acc512, accn512= hs_acc(c["hs_ex"], diag)

        packed= st.get("packed_total_bytes")

        rows.append({
            "condition": name, "family": fam, "target_sparsity": sp, "bits": bits,
            "wt2_ppl": rnd(c["wikitext2"]["ppl"], 3), "wt2_mean_nll": rnd(c["wikitext2"]["mean_nll"]),
            "wt2_n_scored": c["wikitext2"]["n_scored"], "wt2_n_windows": c["wikitext2"]["n_windows"],
            "c4_ppl": rnd(c["c4_val"]["ppl"], 3), "c4_mean_nll": rnd(c["c4_val"]["mean_nll"]),
            "c4_n_scored": c["c4_val"]["n_scored"], "c4_n_windows": c["c4_val"]["n_windows"],
            "hs_n": c["hellaswag"]["n_examples"], "hs_acc": rnd(c["hellaswag"]["acc"]),
            "hs_acc_norm": rnd(c["hellaswag"]["acc_norm"]), "hs512_acc": acc512, "hs512_acc_norm": accn512,
            "eligible_weights": st["eligible_weights"], "eligible_nonzeros": st["eligible_nonzeros"],
            "eligible_nonzero_fraction": rnd(st["eligible_nonzero_fraction"], 6),
            "nominal_bits_per_eligible_weight": st.get("nominal_bits_per_eligible_weight", 16),
            "effective_bits_per_eligible_weight": rnd(st.get("effective_bits_per_eligible_weight"), 4),
            "packed_total_bytes": packed, "packed_eligible_bytes": st.get("packed_eligible_bytes"),
            "unfolded_scale_bytes": st.get("unfolded_scale_bytes"),
            "dense_fp16_bytes": st["dense_fp16_bytes"], "sparse16_estimate_bytes": st["sparse16_estimate_bytes"],
            "packed_over_dense_fp16": rnd(packed / st["dense_fp16_bytes"]) if packed else None,
            "sparse16_over_dense_fp16": rnd(st["sparse16_estimate_bytes"] / st["dense_fp16_bytes"]),
            "calibration_s": rnd(c["condition"].get("calibration_seconds"), 2),
            "compression_s": rnd(c["condition"].get("compression_seconds"), 2),
            "error_pass_s": rnd(c["condition"].get("error_pass_seconds"), 2),
            "build_peak_alloc_MiB": rnd(c["condition"]["build_peak_allocated_bytes"] / 2**20, 1) if c["condition"].get(
                "build_peak_allocated_bytes") else None,
            "prefill512_ms": rnd(lat["prefill_512"]["median"], 2), "prefill512_iqr": rnd(lat["prefill_512"]["iqr"], 2),
            "prefill2048_ms": rnd(lat["prefill_2048"]["median"], 2), "prefill2048_iqr": rnd(lat["prefill_2048"]["iqr"], 2),
            "decode64_ms": rnd(lat["decode_64_after_512"]["median"], 1),
            "decode64_iqr": rnd(lat["decode_64_after_512"]["iqr"], 1),
            "decode_ms_per_token": rnd(lat["decode_64_after_512"]["ms_per_token"], 3),
            "latency_reps": lat["prefill_512"]["reps"],
            "latency_peak_alloc_MiB": rnd(c["resources"]["peak_allocated_bytes"] / 2**20, 1),
            "eval_peak_alloc_MiB": rnd(max(c[s]["peak_allocated_bytes"] for s in ["wikitext2", "c4_val", "hellaswag"]) / 2**20, 1),
            "layer_rel_err_mean": rnd(np.mean(errs), 6) if errs else None,
            "layer_rel_err_median": rnd(np.median(errs), 6) if errs else None,
            "layer_rel_err_max": rnd(np.max(errs), 6) if errs else None,
            "storage_label": "fake-quant fp16 in memory; packed bytes = serialized count" if bits else (
                "zeros in dense fp16; sparse16 = estimate, no sparse kernel" if sp else "dense fp16"),
            "git_commit": c["condition"].get("git_commit") or c["wikitext2"].get("git_commit")
        })

    return rows


def layer_table(conds):
    rows= []

    for name, c in conds.items():
        for r in c["layers"]:
            rows.append({"condition": name, "layer": r["layer"], "proj": r["proj"],
                         "rel_output_error": rnd(r["rel_output_error"], 8),
                         "nonzero_fraction": rnd(r.get("nonzero_fraction"), 6), "bits": r.get("bits"),
                         "awq_alpha": r.get("awq_alpha"), "packed_bytes": r.get("packed_bytes"),
                         "pruned_score_fraction": rnd(r.get("pruned_score_fraction"), 6),
                         "score_mean": rnd(r.get("score_mean"), 6),
                         "score_q50": rnd((r.get("score_quantiles") or {}).get("0.5"), 6),
                         "score_q99": rnd((r.get("score_quantiles") or {}).get("0.99"), 6),
                         "score_max": rnd((r.get("score_quantiles") or {}).get("1.0"), 6)})

    return rows


def map_tables(splits):
    sens= read_json(os.path.join(splits, "sensitivity.json"))
    allocs= {k: read_json(os.path.join(splits, f"{k}.json")) for k in
             ["p3_allocation_s0.5", "p3_allocation_s0.75", "q3_allocation_b4", "q3_allocation_b3"]}
    by_name= {k: {u["name"]: u for u in a["units"]} for k, a in allocs.items()}
    rows= []

    for u in sens["units"]:
        row= {"name": u["name"], "layer": u["layer"], "proj": u["proj"], "params": int(np.prod(u["shape"])),
              "taylor_mean_all": u["taylor_mean_all"]}

        for s, v in u["taylor_pruned_mean"].items():
            row[f"taylor_pruned_mean_s{s}"]= v

        for b, v in u["fisher_quant"].items():
            row[f"fisher_quant_b{b}"]= v

        for s in ["0.5", "0.75"]:
            a= by_name[f"p3_allocation_s{s}"][u["name"]]
            row[f"p3_s{s}_sparsity"]= rnd(a["sparsity"], 6)
            row[f"p3_s{s}_achieved"]= rnd(a["achieved"], 6)

        for b in ["4", "3"]:
            row[f"q3_b{b}_bits"]= by_name[f"q3_allocation_b{b}"][u["name"]]["bits"]

        rows.append(row)

    meta= []

    for k, a in allocs.items():
        meta.append({"allocation": k, "sensitivity": a["sensitivity"], "rule": a["rule"], "unit": a["unit"],
                     "chosen_lambda": a.get("chosen_lambda"), "selection": a.get("selection"),
                     "total_zeros": a.get("total_zeros"), "budget_from": a.get("budget_from"),
                     "budget_total_bytes": a.get("budget_total_bytes"), "packed_total_bytes": a.get("packed_total_bytes"),
                     "bit_counts": a.get("bit_counts"), "objective": a.get("objective"),
                     "objective_uniform": a.get("objective_uniform"), "sensitivity_sha256": a["sensitivity_sha256"]})

    sweep= []

    for s in ["0.5", "0.75"]:
        for r in allocs[f"p3_allocation_s{s}"]["dev_sweep"]:
            sweep.append({"target_sparsity": float(s), **r,
                          "chosen": r["lambda"] == allocs[f"p3_allocation_s{s}"]["chosen_lambda"]})

    return rows, meta, sweep, sens


#---------------------------------------------------------------- paired bootstrap
def ppl_diff_stat(units):
    na= sum(u[0] for u in units)
    nb= sum(u[1] for u in units)
    n= sum(u[2] for u in units)

    return float(np.exp(na / n) - np.exp(nb / n))


def bootstrap_table(conds):
    rows= []

    for a, b in PAIRS:
        ca, cb= conds[a], conds[b]

        for suite, key in [("wikitext2", "wt2_ex"), ("c4_val", "c4_ex")]:
            ws= sorted(ca[key])
            units= [(ca[key][w]["nll_sum"], cb[key][w]["nll_sum"], ca[key][w]["n_scored"]) for w in ws]
            bs= bootstrap_stat(units, ppl_diff_stat)
            rows.append({"comparison": f"{a} - {b}", "metric": f"{suite}_ppl", "unit": "window", "n_units": bs["n_units"],
                         "value_a": rnd(ca[suite]["ppl"], 3), "value_b": rnd(cb[suite]["ppl"], 3),
                         "diff": rnd(bs["estimate"], 4), "ci_low": rnd(bs["ci_low"], 4), "ci_high": rnd(bs["ci_high"], 4),
                         "resamples": bs["resamples"]})

        common= sorted(set(ca["hs_ex"]) & set(cb["hs_ex"]))

        for metric, pk in [("hs_acc", "pred"), ("hs_acc_norm", "pred_norm")]:
            diffs= [float(ca["hs_ex"][o][pk] == ca["hs_ex"][o]["gold"]) - float(cb["hs_ex"][o][pk] == cb["hs_ex"][o]["gold"])
                    for o in common]
            bs= paired_bootstrap(diffs)
            rows.append({"comparison": f"{a} - {b}", "metric": metric, "unit": "example", "n_units": bs["n_units"],
                         "value_a": rnd(np.mean([ca["hs_ex"][o][pk] == ca["hs_ex"][o]["gold"] for o in common])),
                         "value_b": rnd(np.mean([cb["hs_ex"][o][pk] == cb["hs_ex"][o]["gold"] for o in common])),
                         "diff": rnd(bs["mean_diff"]), "ci_low": rnd(bs["ci_low"]), "ci_high": rnd(bs["ci_high"]),
                         "p_two_sided": rnd(bs["p_two_sided"]), "resamples": bs["resamples"]})

    return rows


#---------------------------------------------------------------- fixed-rule qualitative case finder
def norm_margin(ex):
    #char-normalised log-likelihood of gold minus best other ending (lm-eval acc_norm convention)
    s= [ll / ch for ll, ch in zip(ex["lls"], ex["choice_chars"])]
    g= ex["gold"]

    return s[g] - max(v for i, v in enumerate(s) if i != g)


def case_finder(conds, data_dir, per_cond=5, min_tokens=32):
    #rule (fixed before looking): hellaswag = examples baseline gets right (acc_norm) and the condition gets wrong,
    #ranked by the largest drop in normalised gold margin; ppl = windows with >= 32 scored tokens, ranked by the
    #largest increase in per-token nll vs baseline. top 5 per full-2,000 condition. no outcome-based filtering beyond this.
    hs_rows= read_jsonl(os.path.join(data_dir, "hellaswag_val_locked.jsonl.gz"))
    win_rows= read_jsonl(os.path.join(data_dir, "wikitext2_test_windows_locked.jsonl.gz"))
    base= conds["baseline"]
    tok= None

    try:
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
        #cached snapshot has tokenizer files but no config.json, so load from the folder (no download)
        tok= AutoTokenizer.from_pretrained(snapshot_download(
            "Qwen/Qwen2.5-1.5B",
            revision=base["wikitext2"]["model_revision"],
            local_files_only=True
        ))
    except Exception as e:
        print(f"tokenizer not available locally ({e}); window text left empty")

    out= []
    full= [n for n, c in conds.items() if n != "baseline" and c["hellaswag"]["n_examples"] == 2000]

    for name in full:
        c= conds[name]
        cand= [o for o in c["hs_ex"] if base["hs_ex"][o]["pred_norm"] == base["hs_ex"][o]["gold"]
               and c["hs_ex"][o]["pred_norm"] != c["hs_ex"][o]["gold"]]
        cand.sort(key=lambda o: norm_margin(c["hs_ex"][o]) - norm_margin(base["hs_ex"][o]))

        for o in cand[:per_cond]:
            ex= hs_rows[o]
            assert int(ex["order"]) == o and int(ex["source_row_index"]) == c["hs_ex"][o]["source_row_index"]
            e= ex["example"]
            out.append({"condition": name, "kind": "hellaswag", "id": o, "source_row_index": ex["source_row_index"],
                        "baseline_margin": rnd(norm_margin(base["hs_ex"][o])), "condition_margin": rnd(norm_margin(c["hs_ex"][o])),
                        "context": f"{e['activity_label']}: {e['ctx']}", "gold_ending": e["endings"][c["hs_ex"][o]["gold"]],
                        "condition_choice": e["endings"][c["hs_ex"][o]["pred_norm"]]})

        ws= [w for w in c["wt2_ex"] if c["wt2_ex"][w]["n_scored"] >= min_tokens]
        ws.sort(key=lambda w: -(c["wt2_ex"][w]["nll_sum"] - base["wt2_ex"][w]["nll_sum"]) / c["wt2_ex"][w]["n_scored"])

        for w in ws[:per_cond]:
            n= c["wt2_ex"][w]["n_scored"]
            ids= win_rows[w]["token_ids"]
            out.append({"condition": name, "kind": "wikitext2_window", "id": w, "source_row_index": win_rows[w]["source_row"],
                        "baseline_margin": rnd(base["wt2_ex"][w]["nll_sum"] / n), "condition_margin": rnd(c["wt2_ex"][w]["nll_sum"] / n),
                        "context": tok.decode(ids)[:400] if tok else "", "gold_ending": None, "condition_choice": None})

    return out


def cross_config_scores(conds, candidates):
    #same input across every configuration: for each hellaswag id / window picked by the fixed rule above, the four
    #char-normalised candidate scores (or per-token nll) under every condition that scored it
    rows= []
    hs_ids= sorted({c["id"] for c in candidates if c["kind"] == "hellaswag"})
    win_ids= sorted({c["id"] for c in candidates if c["kind"] == "wikitext2_window"})

    for o in hs_ids:
        for name, c in conds.items():
            if o not in c["hs_ex"]:
                continue

            ex= c["hs_ex"][o]
            s= [ll / ch for ll, ch in zip(ex["lls"], ex["choice_chars"])]
            row= {"kind": "hellaswag", "id": o, "condition": name, "gold": ex["gold"], "pred": ex["pred"],
                  "pred_norm": ex["pred_norm"], "norm_margin": rnd(norm_margin(ex))}

            for i in range(4):
                row[f"norm_score_{i}"]= rnd(s[i])
                row[f"raw_ll_{i}"]= rnd(ex["lls"][i])

            rows.append(row)

    for w in win_ids:
        for name, c in conds.items():
            r= c["wt2_ex"][w]
            rows.append({"kind": "wikitext2_window", "id": w, "condition": name, "n_scored": r["n_scored"],
                         "nll_per_token": rnd(r["nll_sum"] / r["n_scored"])})

    return rows


def baseline_deltas(conds):
    #paired change of every condition from the unchanged model (handout p.5 "paired changes from the unchanged model")
    rows= []
    base= conds["baseline"]

    for name, c in conds.items():
        if name == "baseline":
            continue

        row= {"condition": name}

        for suite, key in [("wikitext2", "wt2_ex"), ("c4_val", "c4_ex")]:
            ws= sorted(c[key])
            units= [(c[key][w]["nll_sum"], base[key][w]["nll_sum"], c[key][w]["n_scored"]) for w in ws]
            bs= bootstrap_stat(units, ppl_diff_stat)
            row[f"{suite}_ppl"]= rnd(c[suite]["ppl"], 3)
            row[f"{suite}_ppl_baseline"]= rnd(base[suite]["ppl"], 3)
            row[f"{suite}_ppl_diff"]= rnd(bs["estimate"], 3)
            row[f"{suite}_ppl_ci_low"]= rnd(bs["ci_low"], 3)
            row[f"{suite}_ppl_ci_high"]= rnd(bs["ci_high"], 3)
            row[f"{suite}_ppl_rel_change_pct"]= rnd(100 * (c[suite]["ppl"] / base[suite]["ppl"] - 1), 2)

        orders= sorted(c["hs_ex"])

        for metric, pk in [("hs_acc", "pred"), ("hs_acc_norm", "pred_norm")]:
            diffs= [float(c["hs_ex"][o][pk] == c["hs_ex"][o]["gold"]) - float(base["hs_ex"][o][pk] == base["hs_ex"][o]["gold"])
                    for o in orders]
            bs= paired_bootstrap(diffs)
            row[f"{metric}_n"]= len(orders)
            row[f"{metric}_diff_pp"]= rnd(100 * bs["mean_diff"], 2)
            row[f"{metric}_ci_low_pp"]= rnd(100 * bs["ci_low"], 2)
            row[f"{metric}_ci_high_pp"]= rnd(100 * bs["ci_high"], 2)

        rows.append(row)

    return rows


def third(layer, n_layers=28):
    return "early (0-9)" if layer < 10 else ("middle (10-18)" if layer < 19 else "late (19-27)")


def component_summary(conds, map_rows):
    #per projection type and per layer third: calibration error, nonzeros, wanda pruned-score share (pruning/quant
    #conditions), and the p3/q3 sensitivity measures and final allocations (parameter-weighted where it is a share)
    rows= []
    params= {(r["layer"], r["proj"]): r["params"] for r in map_rows}

    for name, c in conds.items():
        if not c["layers"]:
            continue

        for gname, gkey in [("proj", lambda r: r["proj"]), ("layer_third", lambda r: third(r["layer"]))]:
            groups= defaultdict(list)

            for r in c["layers"]:
                groups[gkey(r)].append(r)

            for g, rs in groups.items():
                w= np.array([params[(r["layer"], r["proj"])] for r in rs], dtype=float)
                psf= [r["pruned_score_fraction"] for r in rs if r.get("pruned_score_fraction") is not None]
                rows.append({"source": name, "group_by": gname, "group": g, "n_units": len(rs),
                             "rel_output_error_mean": rnd(np.mean([r["rel_output_error"] for r in rs]), 6),
                             "rel_output_error_max": rnd(np.max([r["rel_output_error"] for r in rs]), 6),
                             "nonzero_fraction_weighted": rnd(np.average([r.get("nonzero_fraction", np.nan) for r in rs], weights=w), 4)
                             if all(r.get("nonzero_fraction") is not None for r in rs) else None,
                             "bits_weighted": rnd(np.average([r["bits"] for r in rs], weights=w), 3) if all(r.get("bits") for r in rs) else None,
                             "pruned_score_fraction_mean": rnd(np.mean(psf), 6) if psf else None})

    keys= ["taylor_pruned_mean_s0.5", "taylor_pruned_mean_s0.75", "fisher_quant_b4", "fisher_quant_b3", "fisher_quant_b2",
           "fisher_quant_b8", "p3_s0.5_sparsity", "p3_s0.75_sparsity", "q3_b4_bits", "q3_b3_bits"]

    for gname, gkey in [("proj", lambda r: r["proj"]), ("layer_third", lambda r: third(r["layer"]))]:
        groups= defaultdict(list)

        for r in map_rows:
            groups[gkey(r)].append(r)

        for g, rs in groups.items():
            w= np.array([r["params"] for r in rs], dtype=float)
            row= {"source": "sensitivity_and_allocation", "group_by": gname, "group": g, "n_units": len(rs)}

            for k in keys:
                vals= [r[k] for r in rs]
                #sensitivities are ~1e-7, so keep 4 significant figures instead of fixed decimals
                v= np.average(vals, weights=w) if k.startswith(("p3", "q3")) else np.mean(vals)
                row[k + ("_weighted" if k.startswith(("p3", "q3")) else "_mean")]= float(f"{v:.4g}")

            for k in ["fisher_quant_b4", "taylor_pruned_mean_s0.5"]:
                row[f"{k}_share_of_total"]= rnd(sum(r[k] for r in rs) / sum(r[k] for r in map_rows), 4)

            rows.append(row)

    return rows


def storage_breakdown(rows, map_rows):
    #packed bytes split into b-bit codes, quantizer metadata (fp16 scales + packed zero-points + padding),
    #unfolded awq scales and the non-eligible fp16 params
    out= []
    eligible= sum(r["params"] for r in map_rows)

    for r in rows:
        if not r["packed_total_bytes"]:
            continue

        if r["family"] == "q3":
            alloc= f"q3_b{r['bits']}_bits"
            codes= sum(m["params"] * m[alloc] for m in map_rows) / 8
        else:
            codes= eligible * r["bits"] / 8

        scales= r["unfolded_scale_bytes"] or 0
        meta= r["packed_eligible_bytes"] - codes
        non_el= r["packed_total_bytes"] - r["packed_eligible_bytes"] - scales

        out.append({"condition": r["condition"], "packed_total_bytes": r["packed_total_bytes"], "code_bytes": int(codes),
                    "quantizer_metadata_bytes": int(meta), "unfolded_awq_scale_bytes": scales,
                    "non_eligible_fp16_bytes": int(non_el),
                    "metadata_share_of_eligible": rnd(meta / r["packed_eligible_bytes"]),
                    "effective_bits_per_eligible_weight": r["effective_bits_per_eligible_weight"],
                    "packed_over_dense_fp16": r["packed_over_dense_fp16"],
                    "eval_peak_alloc_MiB": r["eval_peak_alloc_MiB"], "latency_peak_alloc_MiB": r["latency_peak_alloc_MiB"],
                    "note": "evaluated as fake-quant fp16: allocated memory and latency are fp16 numbers"})

    return out


#---------------------------------------------------------------- figures
def quality_figures(rows, fig_dir):
    base= next(r for r in rows if r["family"] == "baseline")

    #quality vs sparsity
    fig, axes= plt.subplots(1, 3, figsize=(15, 4))

    for ax, key, lab, log in [(axes[0], "wt2_ppl", "WikiText-2 PPL", True), (axes[1], "c4_ppl", "C4-val PPL", True),
                              (axes[2], "hs512_acc_norm", "HellaSwag acc_norm (first 512)", False)]:
        for fam in ["p1", "p2", "p3", "p2_2of4"]:
            rs= sorted([r for r in rows if r["family"] == fam], key=lambda r: r["target_sparsity"])
            #p1/p2 start from the baseline at 0; p3 (50/75% only) and 2:4 (50% only) are not joined to it
            from_base= fam in ["p1", "p2"]
            xs= ([0.0] if from_base else []) + [r["target_sparsity"] for r in rs]
            ys= ([base[key]] if from_base else []) + [r[key] for r in rs]
            ax.plot(xs, ys, marker=MARKERS[fam], markersize=8, linewidth=2 if fam != "p2_2of4" else 0,
                    color=COLORS[fam], label=fam if fam != "p2_2of4" else "P2 2:4")

        ax.axhline(base[key], color="#999999", linewidth=1, linestyle="--")

        if log:
            ax.set_yscale("log")

        ax.set_xlabel("eligible-weight sparsity")
        ax.set_ylabel(lab)
        clean_axes(ax)

    axes[0].legend(frameon=False)
    fig.suptitle("Pruning: quality vs sparsity (grey dashed = FP16 baseline; PPL axes log)")
    save_fig(fig, os.path.join(fig_dir, "quality_vs_sparsity.png"))

    #quality vs bytes (quantization at packed bytes; pruning at the 16-bit sparse estimate)
    fig, axes= plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, key, lab in [(axes[0], "wt2_ppl", "WikiText-2 PPL"), (axes[1], "hs512_acc_norm", "HellaSwag acc_norm (first 512)")]:
        ax.scatter([base["dense_fp16_bytes"] / 1e9], [base[key]], color=COLORS["baseline"], marker="*", s=150, label="FP16 baseline")

        for fam in ["q1", "q2", "q3"]:
            rs= sorted([r for r in rows if r["family"] == fam], key=lambda r: r["packed_total_bytes"])
            ax.plot([r["packed_total_bytes"] / 1e9 for r in rs], [r[key] for r in rs], marker=MARKERS[fam], markersize=8,
                    linewidth=2, color=COLORS[fam], label=f"{fam.upper()} (packed bytes)")

        for fam in ["p1", "p2", "p3", "p2_2of4"]:
            rs= [r for r in rows if r["family"] == fam and r["target_sparsity"] <= 0.5]
            ax.scatter([r["sparse16_estimate_bytes"] / 1e9 for r in rs], [r[key] for r in rs], marker=MARKERS[fam], s=50,
                       facecolors="none", edgecolors=COLORS[fam], linewidths=1.5,
                       label=f"{'P2 2:4' if fam == 'p2_2of4' else fam.upper()} (16-bit sparse estimate, <=50%)")

        if key == "wt2_ppl":
            ax.set_yscale("log")

        ax.set_xlabel("model bytes (GB, incl. non-eligible params)")
        ax.set_ylabel(lab)
        clean_axes(ax)

    axes[1].legend(frameon=False, fontsize=7)
    fig.suptitle("Quality vs bytes (quantized = fake-quant FP16 evaluated; bytes are the packed-format count)")
    save_fig(fig, os.path.join(fig_dir, "quality_vs_bytes.png"))

    #calibration (layer-output) error vs held-out quality
    fig, ax= plt.subplots(figsize=(6.5, 4.5))

    for fam in ["p1", "p2", "p3", "p2_2of4", "q1", "q2", "q3"]:
        rs= [r for r in rows if r["family"] == fam]
        filled= fam.startswith("q")
        ax.scatter([r["layer_rel_err_mean"] for r in rs], [r["wt2_ppl"] for r in rs], marker=MARKERS[fam], s=55,
                   facecolors=COLORS[fam] if filled else "none", edgecolors=COLORS[fam], linewidths=1.5,
                   label=f"{'P2 2:4' if fam == 'p2_2of4' else fam.upper()}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("mean per-projection relative output error on calibration activations")
    ax.set_ylabel("WikiText-2 PPL")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    clean_axes(ax)
    save_fig(fig, os.path.join(fig_dir, "calibration_error_vs_ppl.png"))


def heat(ax, grid, title, cmap, norm=None, fmt=None, vmin=None, vmax=None):
    im= ax.imshow(grid.T, aspect="auto", cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
    ax.set_yticks(range(len(PROJS)), [p.replace("_proj", "") for p in PROJS], fontsize=7)
    ax.set_xlabel("layer")
    ax.set_title(title, fontsize=9)

    if fmt:
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                ax.text(i, j, fmt(grid[i, j]), ha="center", va="center", fontsize=5)

    return im


def map_figures(map_rows, fig_dir):
    n_layers= max(r["layer"] for r in map_rows) + 1

    def grid(key):
        g= np.zeros((n_layers, len(PROJS)))

        for r in map_rows:
            g[r["layer"], PROJS.index(r["proj"])]= r[key]

        return g

    from matplotlib.colors import LogNorm

    fig, axes= plt.subplots(3, 2, figsize=(14, 9))

    for ax, key, title in [(axes[0, 0], "taylor_pruned_mean_s0.5", "P3 sensitivity: Taylor |g w| over pruned weights, s=0.5"),
                           (axes[0, 1], "taylor_pruned_mean_s0.75", "P3 sensitivity: Taylor, s=0.75")]:
        g= grid(key)
        fig.colorbar(heat(ax, g, title, "Blues", norm=LogNorm(vmin=g[g > 0].min(), vmax=g.max())), ax=ax)

    for ax, key, title in [(axes[1, 0], "fisher_quant_b4", "Q3 sensitivity: Fisher g^2 (Q_b(w)-w)^2, b=4"),
                           (axes[1, 1], "fisher_quant_b3", "Q3 sensitivity: Fisher, b=3")]:
        g= grid(key)
        fig.colorbar(heat(ax, g, title, "Blues", norm=LogNorm(vmin=g[g > 0].min(), vmax=g.max())), ax=ax)

    for ax, s in [(axes[2, 0], "0.5"), (axes[2, 1], "0.75")]:
        fig.colorbar(heat(ax, grid(f"p3_s{s}_sparsity"), f"P3 final sparsity map (target {s})", "Blues", vmin=0.1, vmax=0.85), ax=ax)

    fig.tight_layout()
    save_fig(fig, os.path.join(fig_dir, "sensitivity_and_sparsity_maps.png"))

    fig, axes= plt.subplots(1, 2, figsize=(14, 3.2))
    cmap= matplotlib.colors.ListedColormap(["#cfe0f5", "#7fb0e6", "#2a78d6", "#123d73"])
    norm= matplotlib.colors.BoundaryNorm([1.5, 2.5, 3.5, 6, 9], cmap.N)

    for ax, b in zip(axes, ["4", "3"]):
        im= heat(ax, grid(f"q3_b{b}_bits"), f"Q3 final bit map (budget = Q2 INT{b} packed bytes)", cmap, norm=norm,
                 fmt=lambda v: f"{int(v)}")
        fig.colorbar(im, ax=ax, ticks=[2, 3, 4, 8])

    fig.tight_layout()
    save_fig(fig, os.path.join(fig_dir, "q3_bit_maps.png"))


#---------------------------------------------------------------- main
def main(args):
    root= args.results_dir
    tdir= os.path.join(root, "tables")
    fig_dir= os.path.join(root, "figures")

    conds= load_conditions(root)
    rows= main_table(conds)
    write_csv(os.path.join(tdir, "main.csv"), rows)
    write_csv(os.path.join(tdir, "layer_errors.csv"), layer_table(conds))

    map_rows, meta, sweep, sens= map_tables(args.splits_dir)
    write_csv(os.path.join(tdir, "sensitivity_and_allocation_maps.csv"), map_rows)
    write_csv(os.path.join(tdir, "allocation_meta.csv"), meta)
    write_csv(os.path.join(tdir, "p3_dev_sweep.csv"), sweep)

    write_csv(os.path.join(tdir, "paired_bootstrap.csv"), bootstrap_table(conds))
    cands= case_finder(conds, args.data_dir)
    write_csv(os.path.join(tdir, "qualitative_candidates.csv"), cands)
    write_csv(os.path.join(tdir, "qualitative_scores_all_conditions.csv"), cross_config_scores(conds, cands))
    write_csv(os.path.join(tdir, "baseline_deltas.csv"), baseline_deltas(conds))
    write_csv(os.path.join(tdir, "component_summary.csv"), component_summary(conds, map_rows))
    write_csv(os.path.join(tdir, "storage_breakdown.csv"), storage_breakdown(rows, map_rows))

    quality_figures(rows, fig_dir)
    map_figures(map_rows, fig_dir)

    keep= ["condition", "wt2_ppl", "c4_ppl", "hs_n", "hs_acc_norm", "hs512_acc_norm", "eligible_nonzero_fraction",
           "nominal_bits_per_eligible_weight", "packed_total_bytes", "sparse16_estimate_bytes", "prefill512_ms",
           "decode_ms_per_token"]
    write_csv(os.path.join(root, "summary.csv"), [{k: r[k] for k in keep} for r in rows])


if __name__ == "__main__":
    parser= argparse.ArgumentParser()

    parser.add_argument(
        "--results_dir",
        default="results/task1",
        help="folder holding one subfolder per condition"
    )

    parser.add_argument(
        "--splits_dir",
        default="splits/task1",
        help="sensitivity.json and frozen allocation files"
    )

    parser.add_argument(
        "--data_dir",
        default="atml-pa4-datasets/data/public_locked",
        help="locked hellaswag / wikitext2 files (text for the case finder only)"
    )

    args= parser.parse_args()
    main(args)
