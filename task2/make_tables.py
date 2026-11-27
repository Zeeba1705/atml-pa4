#task 2 tables and figures (handout pp.7-11, required evidence). cpu only, reads results/task2, writes
#results/task2/tables/*.csv, results/task2/summary.csv and results/task2/figures/*.png. numbers only, no interpretation.
#bootstrap: paired, 2,000 resamples, seed 6304. mqar: over test examples (per-example query accuracy), per seed.
#kv: over prompts (cases).
import os
import csv
import json
import glob
import gzip
import argparse
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.bootstrap import paired_bootstrap

SEEDS= [6304, 6305, 6306]
KS= [4, 8, 16]
REGIMES= ["near", "mixed", "far"]
DIST_BINS= ["[0,32)", "[32,64)", "[64,128)", "[128,256)"]
IMPLS_A= ["dense", "tiled", "sdpa", "sdpa_FLASH_ATTENTION", "sdpa_EFFICIENT_ATTENTION", "sdpa_CUDNN_ATTENTION", "sdpa_MATH"]
KV_METHODS= ["fp16", "evict_sink_recent", "evict_question", "int4_token", "int4_channel_k", "hybrid_int2"]
KV_PAIRS= [
    ("evict_question", "evict_sink_recent"),
    ("int4_channel_k", "int4_token"),
    ("hybrid_int2", "evict_question"),
    ("hybrid_int2", "int4_channel_k"),
    ("evict_question", "int4_channel_k"),
] + [(m, "fp16") for m in KV_METHODS[1:]]
#reference palette slots in fixed order (dataviz skill)
PALETTE= ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e55c9b", "#7a5cd6"]
ATTN_COLORS= {"dense": PALETTE[0], "window": PALETTE[1], "local_dilated": PALETTE[2], "linear": PALETTE[3],
              "gated_linear": PALETTE[4], "tiled": PALETTE[1], "sdpa": PALETTE[2], "linear_parallel": PALETTE[3],
              "linear_recurrent": PALETTE[4]}
KV_COLORS= dict(zip(KV_METHODS, PALETTE))


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


def mean_std(vals):
    vals= [v for v in vals if v is not None]

    if not vals:
        return None, None

    return rnd(np.mean(vals)), rnd(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)


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


#---------------------------------------------------------------- part A: operators
def attention_tables(root):
    rows, backends, tiles= [], [], []

    for f in sorted(glob.glob(os.path.join(root, "attention", "float*_L*.json"))):
        d= read_json(f)
        dtype= d["dtype"]

        backends.append({
            "dtype": dtype, "L": d["T"], "sdpa_default_backend": d.get("sdpa_default_backend"),
            "sdpa_default_ops": ";".join(d.get("sdpa_default_ops") or []),
            "forced_backends": ";".join(f"{r['impl'][5:]}={r['status']}" for r in d["rows"] if r["impl"].startswith("sdpa_")),
            "gpu": d["gpu_name"], "torch": d["torch"], "compute_capability": d["sdpa"]["compute_capability"]
        })

        for size, v in (d.get("tile_size_verification") or {}).items():
            tiles.append({"dtype": dtype, "L": d["T"], "tile": int(size), "max_abs_err": v["max_abs_err"],
                          "tiles_computed": v["tiles_computed"]})

        for r in d["rows"]:
            fw, fb= r.get("forward") or {}, r.get("forward_backward") or {}
            fm, fbm= r.get("forward_memory") or {}, r.get("forward_backward_memory") or {}
            ferr= (r.get("forward_errors_vs_reference") or {}).get("output") or {}
            err= r.get("errors_vs_reference") or {}

            row= {
                "dtype": dtype, "L": d["T"], "impl": r["impl"], "status": r.get("status"),
                "backward_status": r.get("backward_status"), "reference": d.get("reference"),
                "fwd_ms_median": rnd(fw.get("median")), "fwd_ms_iqr": rnd(fw.get("iqr")), "fwd_reps": fw.get("reps"),
                "fwd_peak_alloc_MiB": rnd(fm["peak_allocated_bytes"] / 2**20, 1) if fm else None,
                "fwd_peak_reserved_MiB": rnd(fm["peak_reserved_bytes"] / 2**20, 1) if fm else None,
                "fwbw_ms_median": rnd(fb.get("median")), "fwbw_ms_iqr": rnd(fb.get("iqr")),
                "fwbw_peak_alloc_MiB": rnd(fbm["peak_allocated_bytes"] / 2**20, 1) if fbm else None,
                "fwbw_peak_reserved_MiB": rnd(fbm["peak_reserved_bytes"] / 2**20, 1) if fbm else None,
                "fwd_out_max_abs_err": ferr.get("max_abs"), "fwd_out_mean_abs_err": ferr.get("mean_abs"),
                "finite": r.get("finite"), "dense_score_matrix_MiB": rnd(d["dense_score_matrix_bytes"] / 2**20, 1)
            }

            for g in ["output", "dq", "dk", "dv"]:
                row[f"{g}_max_abs_err"]= (err.get(g) or {}).get("max_abs")
                row[f"{g}_max_rel_to_ref_max"]= (err.get(g) or {}).get("max_rel_to_ref_max")

            rows.append(row)

    edge= read_json(os.path.join(root, "attention", "edge_cases.json"))
    edges= []

    for dtype in ["float16", "float32"]:
        for impl in ["dense", "tiled", "sdpa"]:
            e= edge["extreme_scores"][dtype][impl]
            c= edge["causality"][dtype][impl]
            edges.append({"dtype": dtype, "impl": impl, "extreme_q_scale": edge["extreme_scores"][dtype]["q_scale"],
                          "extreme_finite": e["finite"], "extreme_max_abs_err_vs_fp64": e["max_abs_err_vs_fp64"],
                          "causality_max_change_before_perturbed_pos": c["max_change_before"],
                          "causality_max_change_after_perturbed_pos": c["max_change_after"]})

        m= edge["all_masked_row"][dtype]
        edges.append({"dtype": dtype, "impl": "all_masked_row", "dense_row_is_zero": m["dense_row_is_zero"],
                      "sdpa_row_finite": m["sdpa_row_finite"], "sdpa_row_value_example": m["sdpa_row_value_example"],
                      "other_rows_max_abs_diff": m["other_rows_max_abs_diff"]})

    return rows, backends, tiles, edges


def attention_figures(rows, fig_dir):
    fig, axes= plt.subplots(2, 2, figsize=(11, 8), sharex=True)

    for j, dtype in enumerate(["float16", "float32"]):
        for i, (key, label) in enumerate([("fwd", "forward"), ("fwbw", "forward+backward")]):
            ax= axes[i, j]

            for impl in ["dense", "tiled", "sdpa"]:
                rs= sorted([r for r in rows if r["dtype"] == dtype and r["impl"] == impl and r[f"{key}_ms_median"]],
                           key=lambda r: r["L"])
                ax.plot([r["L"] for r in rs], [r[f"{key}_ms_median"] for r in rs], marker="o", markersize=5,
                        linewidth=2, color=ATTN_COLORS[impl], label=impl)

            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_title(f"{dtype}, {label}")
            ax.set_ylabel("median latency (ms)")
            clean_axes(ax)

    for ax in axes[1]:
        ax.set_xlabel("sequence length L")

    axes[0, 0].legend(frameon=False)
    fig.suptitle("Part A latency, B=2 H=4 d=64, T4 (tiled = Python reference, not FlashAttention)")
    save_fig(fig, os.path.join(fig_dir, "attention_latency.png"))

    fig, axes= plt.subplots(1, 2, figsize=(11, 4), sharey=True)

    for j, dtype in enumerate(["float16", "float32"]):
        ax= axes[j]

        for impl in ["dense", "tiled", "sdpa"]:
            for key, ls in [("fwd", "-"), ("fwbw", "--")]:
                rs= sorted([r for r in rows if r["dtype"] == dtype and r["impl"] == impl and r[f"{key}_peak_alloc_MiB"]],
                           key=lambda r: r["L"])
                ax.plot([r["L"] for r in rs], [r[f"{key}_peak_alloc_MiB"] for r in rs], marker="o", markersize=5,
                        linewidth=2, linestyle=ls, color=ATTN_COLORS[impl],
                        label=f"{impl} {'fwd' if key == 'fwd' else 'fwd+bwd'}")

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(dtype)
        ax.set_xlabel("sequence length L")
        clean_axes(ax)

    axes[0].set_ylabel("peak allocated (MiB)")
    axes[0].legend(frameon=False, fontsize=8)
    save_fig(fig, os.path.join(fig_dir, "attention_memory.png"))


#---------------------------------------------------------------- part B: MQAR
def load_mqar(root):
    runs= {}

    for d in sorted(glob.glob(os.path.join(root, "mqar", "*_s63*"))):
        name= os.path.basename(d)
        base, seed= name.rsplit("_s", 1)
        tm= read_json(os.path.join(d, "test_metrics.json"))

        runs[name]= {
            "dir": d, "base": base, "seed": int(seed), "attention": tm["run"]["attention"], "K": tm["run"]["K"],
            "regime": tm["run"]["regime"], "test": tm,
            "mask": read_json(os.path.join(d, "test_window_mask_metrics.json")) if os.path.exists(
                os.path.join(d, "test_window_mask_metrics.json")) else None,
            "history": read_json(os.path.join(d, f"{name}_history.json"))
        }

    return runs


def cell_row(run, metrics, condition):
    row= {"run": run["base"], "seed": run["seed"], "attention": run["attention"], "train_K": run["K"],
          "train_regime": run["regime"], "condition": condition, "n_params": metrics["n_params"]}

    for k in KS:
        for reg in REGIMES:
            row[f"K{k}_{reg}"]= rnd(metrics["cells"][f"K{k}_{reg}"]["accuracy"])

    for k in KS:
        row[f"chance_1_over_K{k}"]= rnd(1 / k)

    row["chance_label_space"]= metrics["cells"]["K4_mixed"]["chance_label_space"]

    return row


def mqar_tables(runs):
    test, bins, curves, costs, agg= [], [], [], [], []

    for name, r in runs.items():
        test.append(cell_row(r, r["test"], "trained mask"))

        if r["mask"]:
            test.append(cell_row(r, r["mask"], "dense-trained, W=64 mask at test only"))

        for cond, m in [("trained mask", r["test"]), ("test-time W=64", r["mask"])]:
            if m is None:
                continue

            for k in KS:
                for b in DIST_BINS:
                    c= m["bins"][f"K{k}_{b}"]
                    bins.append({"run": r["base"], "seed": r["seed"], "condition": cond, "test_K": k, "distance_bin": b,
                                 "n_queries": c["n_queries"], "accuracy": rnd(c["accuracy"]),
                                 "mean_loss": rnd(c["mean_loss"]), "chance_1_over_K": rnd(1 / k)})

        for h in r["history"]:
            row= {"run": r["base"], "seed": r["seed"], "update": h["update"],
                  "train_loss_last100": rnd(h["train_loss_last100"]),
                  "train_query_acc_last100": rnd(h["train_query_acc_last100"]),
                  "train_seconds": rnd(h.get("train_seconds"), 1)}

            for k in KS:
                for reg in REGIMES:
                    row[f"dev_K{k}_{reg}"]= rnd(h["dev"][f"K{k}_{reg}"]["accuracy"])

            curves.append(row)

        if r["seed"] == SEEDS[0]:
            p= r["test"]["theoretical_pairs_T256"]
            s= r["test"]["state_bytes_T256_fp32"]
            costs.append({"run": r["base"], "attention": r["attention"], "n_params": r["test"]["n_params"],
                          "pairs_T256_total": p.get("total"), "pairs_T256_per_head": p.get("per_head"),
                          "pairs_T256_per_query_max": p.get("per_query_max"), "pairs_note": p.get("note"),
                          "state_bytes_T256_fp32_per_layer": s["per_layer"], "state_bytes_T256_fp32_total": s["total"]})

    #seed mean / sd per run and condition
    groups= defaultdict(list)

    for row in test:
        groups[(row["run"], row["condition"])].append(row)

    for (run, cond), rs in groups.items():
        out= {"run": run, "condition": cond, "attention": rs[0]["attention"], "train_K": rs[0]["train_K"],
              "train_regime": rs[0]["train_regime"], "n_seeds": len(rs), "seeds": ";".join(str(x["seed"]) for x in rs)}

        for k in KS:
            for reg in REGIMES:
                out[f"K{k}_{reg}_mean"], out[f"K{k}_{reg}_sd"]= mean_std([x[f"K{k}_{reg}"] for x in rs])

        agg.append(out)

    return test, bins, curves, costs, agg


def mqar_inference_table(root):
    rows= []

    for f in sorted(glob.glob(os.path.join(root, "mqar_inference", "T*.json")), key=lambda p: int(p.split("T")[-1][:-5])):
        d= read_json(f)

        for kind, v in d["results"].items():
            lat= v.get("latency_ms") or {}
            rows.append({
                "T": d["T"], "batch_size": d["batch_size"], "kind": kind, "status": v["status"],
                "latency_ms_median": rnd(lat.get("median")), "latency_ms_iqr": rnd(lat.get("iqr")), "reps": lat.get("reps"),
                "peak_alloc_MiB": rnd(v["peak_allocated_bytes"] / 2**20, 1) if v.get("peak_allocated_bytes") else None,
                "alloc_before_call_MiB": rnd(v["allocated_before_call_bytes"] / 2**20, 1) if v.get(
                    "allocated_before_call_bytes") else None,
                "peak_reserved_MiB": rnd(v["peak_reserved_bytes"] / 2**20, 1) if v.get("peak_reserved_bytes") else None,
                "dense_mask_materialized_MiB": rnd(v["dense_mask_materialized_bytes"] / 2**20, 1) if v.get(
                    "dense_mask_materialized_bytes") else None,
                "linear_score_tensor_MiB": rnd(v["linear_score_tensor_bytes"] / 2**20, 1) if v.get(
                    "linear_score_tensor_bytes") else None,
                "theoretical_state_fp32_total_bytes": v["theoretical_state_fp32"]["total"],
                "pairs_total": v["pairs"].get("total"), "dtype": d["dtype"], "inputs": d["inputs"], "label": d["labels"]
            })

    return rows


def per_example_acc(run_dir, cell_prefix):
    out= {}

    for r in read_jsonl(os.path.join(run_dir, "test_predictions.jsonl.gz")):
        if r["id"].startswith(cell_prefix):
            out[r["id"]]= float(np.mean(r["correct"]))

    return out


def b3_bootstrap(runs, b3="b3_gated_linear_K16_mixed", base="linear_K16_mixed"):
    rows= []

    for k in KS:
        for reg in REGIMES:
            prefix= f"test-{reg}-K{k}-"
            pooled_a, pooled_b= defaultdict(list), defaultdict(list)

            for seed in SEEDS:
                ra, rb= runs.get(f"{b3}_s{seed}"), runs.get(f"{base}_s{seed}")

                if ra is None or rb is None:
                    continue

                a, b= per_example_acc(ra["dir"], prefix), per_example_acc(rb["dir"], prefix)
                ids= sorted(a)
                assert ids == sorted(b), "test example ids differ between runs"

                bs= paired_bootstrap([a[i] - b[i] for i in ids])
                rows.append({"comparison": f"{b3} - {base}", "test_cell": f"K{k}_{reg}", "seed_pairing": seed,
                             "n_examples": bs["n_units"], f"acc_{b3}": rnd(np.mean([a[i] for i in ids])),
                             f"acc_{base}": rnd(np.mean([b[i] for i in ids])), "mean_diff": rnd(bs["mean_diff"]),
                             "ci_low": rnd(bs["ci_low"]), "ci_high": rnd(bs["ci_high"]), "p_two_sided": rnd(bs["p_two_sided"])})

                for i in ids:
                    pooled_a[i].append(a[i])
                    pooled_b[i].append(b[i])

            ids= sorted(pooled_a)
            bs= paired_bootstrap([np.mean(pooled_a[i]) - np.mean(pooled_b[i]) for i in ids])
            rows.append({"comparison": f"{b3} - {base}", "test_cell": f"K{k}_{reg}", "seed_pairing": "mean over 3 seeds",
                         "n_examples": bs["n_units"], f"acc_{b3}": rnd(np.mean([np.mean(pooled_a[i]) for i in ids])),
                         f"acc_{base}": rnd(np.mean([np.mean(pooled_b[i]) for i in ids])),
                         "mean_diff": rnd(bs["mean_diff"]), "ci_low": rnd(bs["ci_low"]), "ci_high": rnd(bs["ci_high"]),
                         "p_two_sided": rnd(bs["p_two_sided"]),
                         "note": "examples resampled; seed variation not in this interval (see per-seed rows)"})

    return rows


def mqar_figures(runs, curves, bins, inf_rows, fig_dir):
    bases= list(dict.fromkeys(r["base"] for r in runs.values()))

    #learning curves: dev accuracy on the trained (K, regime) cell
    n= len(bases)
    cols= 4
    fig, axes= plt.subplots((n + cols - 1) // cols, cols, figsize=(14, 3 * ((n + cols - 1) // cols)), sharey=True)
    axes= axes.ravel()

    for ax, base in zip(axes, bases):
        r0= next(r for r in runs.values() if r["base"] == base)
        key= f"dev_K{r0['K']}_{r0['regime']}"

        for seed, ls in zip(SEEDS, ["-", "--", ":"]):
            rs= [c for c in curves if c["run"] == base and c["seed"] == seed]
            ax.plot([c["update"] for c in rs], [c[key] for c in rs], marker="o", markersize=4, linewidth=2,
                    linestyle=ls, color=ATTN_COLORS.get(r0["attention"], PALETTE[5]), label=f"s{seed}")

        ax.axhline(1 / r0["K"], color="#999999", linewidth=1, linestyle="--")
        ax.set_title(f"{base}\n(dev {key[4:]})", fontsize=9)
        ax.set_ylim(-0.02, 1.02)
        clean_axes(ax)

    for ax in axes[n:]:
        ax.axis("off")

    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("MQAR dev accuracy on the trained cell vs update (grey dashed = 1/K)")
    fig.tight_layout()
    save_fig(fig, os.path.join(fig_dir, "mqar_learning_curves.png"))

    #distance-by-load heatmaps (seed mean, trained mask)
    fig, axes= plt.subplots((n + cols - 1) // cols, cols, figsize=(14, 2.6 * ((n + cols - 1) // cols)))
    axes= axes.ravel()

    for ax, base in zip(axes, bases):
        grid= np.zeros((len(KS), len(DIST_BINS)))

        for i, k in enumerate(KS):
            for j, b in enumerate(DIST_BINS):
                vals= [x["accuracy"] for x in bins if x["run"] == base and x["condition"] == "trained mask"
                       and x["test_K"] == k and x["distance_bin"] == b]
                grid[i, j]= np.mean(vals)

        im= ax.imshow(grid, vmin=0, vmax=1, cmap="Blues", aspect="auto")

        for i in range(len(KS)):
            for j in range(len(DIST_BINS)):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if grid[i, j] > 0.6 else "#222222")

        ax.set_xticks(range(len(DIST_BINS)), DIST_BINS, fontsize=7)
        ax.set_yticks(range(len(KS)), [f"K={k}" for k in KS], fontsize=7)
        ax.set_title(base, fontsize=9)

    for ax in axes[n:]:
        ax.axis("off")

    fig.colorbar(im, ax=axes.tolist(), shrink=0.6, label="test accuracy (mean of 3 seeds)")
    fig.suptitle("MQAR test accuracy by load K (rows) and query-key distance (columns)")
    save_fig(fig, os.path.join(fig_dir, "mqar_distance_by_load.png"))

    #forward-only inference sweep
    fig, axes= plt.subplots(1, 2, figsize=(11, 4))

    for kind in dict.fromkeys(r["kind"] for r in inf_rows):
        rs= [r for r in inf_rows if r["kind"] == kind and r["status"] == "ok"]
        col= ATTN_COLORS.get(kind, PALETTE[5])
        axes[0].plot([r["T"] for r in rs], [r["latency_ms_median"] for r in rs], marker="o", markersize=5,
                     linewidth=2, color=col, label=kind)
        axes[1].plot([r["T"] for r in rs], [r["peak_alloc_MiB"] for r in rs], marker="o", markersize=5,
                     linewidth=2, color=col, label=kind)

    for ax, lab in zip(axes, ["median forward latency (ms)", "peak allocated (MiB)"]):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("T")
        ax.set_ylabel(lab)
        clean_axes(ax)

    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("MQAR forward-only sweep, batch 2, fp32, untrained models (window/dilated = dense boolean mask)")
    save_fig(fig, os.path.join(fig_dir, "mqar_inference_sweep.png"))


#---------------------------------------------------------------- part C: KV cache
def load_cases(root):
    return [read_json(f) for f in sorted(glob.glob(os.path.join(root, "kv", "cases", "*.json")))]


def kv_main_table(cases):
    rows= []
    fp16_ok= {c["id"]: c["methods"]["fp16"]["correct"] for c in cases}

    for m in KV_METHODS:
        ms= [c["methods"][m] for c in cases]
        cond= [c["methods"][m]["correct"] for c in cases if fp16_ok[c["id"]]]
        ev= [x["evidence_retained"]["mean"] for x in ms if x.get("evidence_retained")]
        rec= [x["reconstruction_rel_err"] for x in ms if x.get("reconstruction_rel_err")]

        rows.append({
            "method": m, "n_cases": len(ms), "accuracy": rnd(np.mean([x["correct"] for x in ms])),
            "n_fp16_correct": len(cond), "accuracy_given_fp16_correct": rnd(np.mean(cond)) if cond else None,
            "ref_answer_nll_mean": rnd(np.mean([x["ref_answer_nll"] for x in ms])),
            "ref_answer_nll_median": rnd(np.median([x["ref_answer_nll"] for x in ms])),
            "post_prefill_kl_mean": rnd(np.mean([x["post_prefill_kl"] for x in ms]), 6),
            "teacher_forced_kl_mean": rnd(np.mean([x["teacher_forced_kl_mean"] for x in ms]), 6),
            "evidence_retained_mean": rnd(np.mean(ev)) if ev else None,
            "evidence_retained_min_over_cases": rnd(np.min([x["evidence_retained"]["min"] for x in ms
                                                            if x.get("evidence_retained")])) if ev else None,
            "key_rel_err_mean": rnd(np.mean([x["keys"] for x in rec])) if rec else None,
            "value_rel_err_mean": rnd(np.mean([x["values"] for x in rec])) if rec else None,
            "fraction_of_fp16_mean": rnd(np.mean([x["fraction_of_fp16"] for x in ms])),
            "fraction_of_fp16_max": rnd(np.max([x["fraction_of_fp16"] for x in ms])),
            "all_within_budget": all(x["within_budget"] for x in ms),
            "bytes_at_budget_point_mean_MiB": rnd(np.mean([x["bytes_at_budget_point"] for x in ms]) / 2**20, 3),
            "shared_fp16_suffix_generated_mean_MiB": rnd(
                np.mean([x["shared_fp16_bytes_suffix_and_generated"] for x in ms]) / 2**20, 3),
            "fp16_positions_mean": rnd(np.mean([x["fp16_positions"] for x in ms]), 1),
            "compress_s_mean": rnd(np.mean([x["compress_seconds"] for x in ms])),
            "suffix_prefill_s_mean": rnd(np.mean([x["suffix_prefill_seconds"] for x in ms])),
            "decode_s_mean": rnd(np.mean([x["decode_seconds"] for x in ms])),
            "n_generated_mean": rnd(np.mean([x["n_generated"] for x in ms]), 2),
            "stopped_eos_fraction": rnd(np.mean([x["stopped_eos"] for x in ms])),
            "peak_alloc_MiB_max": rnd(np.max([x["peak_allocated_bytes"] for x in ms]) / 2**20, 1),
            "all_finite": all(x["finite"] for x in ms),
            "retention_granularity": ms[0].get("retention_granularity") or ms[0].get("layout")
        })

    return rows


def kv_breakdowns(cases):
    rows= []
    factors= {"length": "target_tokens", "position": "position", "distractors": "distractors"}

    #one-way margins and the full 3-way cell grid
    for fname, key in list(factors.items()) + [("cell", None)]:
        groups= defaultdict(list)

        for c in cases:
            g= f"{c['target_tokens']}/{c['position']}/{c['distractors']}" if key is None else c[key]
            groups[g].append(c)

        for g, cs in sorted(groups.items(), key=lambda x: str(x[0])):
            for m in KV_METHODS:
                ms= [c["methods"][m] for c in cs]
                ev= [x["evidence_retained"]["mean"] for x in ms if x.get("evidence_retained")]
                rows.append({"factor": fname, "level": g, "method": m, "n_cases": len(ms),
                             "accuracy": rnd(np.mean([x["correct"] for x in ms])),
                             "ref_answer_nll_mean": rnd(np.mean([x["ref_answer_nll"] for x in ms])),
                             "post_prefill_kl_mean": rnd(np.mean([x["post_prefill_kl"] for x in ms]), 6),
                             "evidence_retained_mean": rnd(np.mean(ev)) if ev else None,
                             "fraction_of_fp16_mean": rnd(np.mean([x["fraction_of_fp16"] for x in ms]))})

    return rows


def kv_bootstrap(cases):
    rows= []
    subsets= [("all", cases), ("2048", [c for c in cases if c["target_tokens"] == 2048]),
              ("4096", [c for c in cases if c["target_tokens"] == 4096])]

    for a, b in KV_PAIRS:
        for sub, cs in subsets:
            for metric in ["correct", "ref_answer_nll", "post_prefill_kl"]:
                diffs= [float(c["methods"][a][metric]) - float(c["methods"][b][metric]) for c in cs]
                bs= paired_bootstrap(diffs)
                rows.append({"comparison": f"{a} - {b}", "subset": sub, "metric": "accuracy" if metric == "correct" else metric,
                             "n_cases": bs["n_units"], "mean_a": rnd(np.mean([float(c["methods"][a][metric]) for c in cs]), 6),
                             "mean_b": rnd(np.mean([float(c["methods"][b][metric]) for c in cs]), 6),
                             "mean_diff": rnd(bs["mean_diff"], 6), "ci_low": rnd(bs["ci_low"], 6),
                             "ci_high": rnd(bs["ci_high"], 6), "p_two_sided": rnd(bs["p_two_sided"])})

    return rows


def kv_lengths_and_c1(cases):
    lens, c1= [], []

    for n in sorted({c["target_tokens"] for c in cases}):
        cs= [c for c in cases if c["target_tokens"] == n]
        lens.append({"nominal_tokens": n, "n_cases": len(cs),
                     "context_tokens_min": min(c["context_tokens"] for c in cs),
                     "context_tokens_mean": rnd(np.mean([c["context_tokens"] for c in cs]), 1),
                     "context_tokens_max": max(c["context_tokens"] for c in cs),
                     "suffix_tokens_min": min(c["suffix_tokens"] for c in cs),
                     "suffix_tokens_max": max(c["suffix_tokens"] for c in cs),
                     "evidence_tokens_mean": rnd(np.mean([c["evidence_tokens"] for c in cs]), 1),
                     "budget_MiB_mean": rnd(np.mean([c["budget_bytes"] for c in cs]) / 2**20, 3),
                     "fp16_context_MiB_mean": rnd(np.mean([c["fp16_context_bytes"] for c in cs]) / 2**20, 3),
                     "context_prefill_s_mean": rnd(np.mean([c["context_prefill_seconds"] for c in cs])),
                     "context_prefill_peak_MiB_max": rnd(max(c["context_prefill_peak_bytes"] for c in cs) / 2**20, 1),
                     "question_scoring_s_mean": rnd(np.mean([c["question_scoring_seconds"] for c in cs]))})

    for c in cases:
        if c.get("c1_checks"):
            c1.append({"case": c["id"], **c["c1_checks"]})

    return lens, c1


def kv_bench_table(root):
    rows= []

    for f in sorted(glob.glob(os.path.join(root, "kv_bench", "len*_b*.json"))):
        d= read_json(f)
        fp= d["results"]["fp16"]

        rows.append({"nominal_length": d["nominal_length"], "context_tokens": d["context_tokens"], "batch_size": d["batch_size"],
                     "layout": "fp16", "bits": 16, "serialized_MiB": rnd(fp["serialized_bytes"] / 2**20, 2),
                     "allocated_cache_MiB": rnd(fp["allocated_cache_bytes"] / 2**20, 2), "fraction_of_fp16": 1.0,
                     "decode_step_ms_median": rnd(fp["decode_step_ms"]["median"]),
                     "decode_step_ms_iqr": rnd(fp["decode_step_ms"]["iqr"]), "reps": fp["decode_step_ms"]["reps"],
                     "step_peak_alloc_MiB": rnd(fp["decode_step_peak_allocated_bytes"] / 2**20, 1), "label": d["labels"]})

        for lay, v in d["results"].items():
            if lay == "fp16":
                continue

            assert v["serialized_bytes"] == v["formula_bytes"], f"{f} {lay}: serialized bytes differ from formula"
            rows.append({"nominal_length": d["nominal_length"], "context_tokens": d["context_tokens"],
                         "batch_size": d["batch_size"], "layout": lay, "bits": v["bits"],
                         "serialized_MiB": rnd(v["serialized_bytes"] / 2**20, 2),
                         "formula_MiB": rnd(v["formula_bytes"] / 2**20, 2), "fraction_of_fp16": rnd(v["fraction_of_fp16"]),
                         "pack_ms_median": rnd(v["pack_ms"]["median"]), "pack_ms_iqr": rnd(v["pack_ms"]["iqr"]),
                         "unpack_dequant_ms_median": rnd(v["unpack_dequant_ms"]["median"]),
                         "unpack_dequant_ms_iqr": rnd(v["unpack_dequant_ms"]["iqr"]),
                         "unpack_plus_step_ms_median": rnd(v["unpack_plus_decode_step_ms"]["median"]),
                         "unpack_plus_step_ms_iqr": rnd(v["unpack_plus_decode_step_ms"]["iqr"]),
                         "reps": v["pack_ms"]["reps"],
                         "step_peak_alloc_MiB": rnd(v["unpack_plus_decode_step_peak_allocated_bytes"] / 2**20, 1),
                         "label": d["labels"]})

    return rows


def kv_generations(cases):
    rows= []

    for c in cases:
        if not c["id"].endswith("-2-00"):
            continue

        for m in KV_METHODS:
            x= c["methods"][m]
            rows.append({"case": c["id"], "code": c["code"], "method": m, "correct": x["correct"],
                         "prediction": x["prediction"], "response": x["response"], "n_generated": x["n_generated"],
                         "stopped_eos": x["stopped_eos"], "ref_answer_nll": rnd(x["ref_answer_nll"]),
                         "evidence_retained_mean": rnd(x["evidence_retained"]["mean"]) if x.get("evidence_retained") else None})

    return rows


def kv_figures(cases, main_rows, break_rows, fig_dir):
    #accuracy by evidence position and length
    fig, axes= plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    for ax, n in zip(axes, [2048, 4096]):
        x= np.arange(3)
        w= 0.13

        for i, m in enumerate(KV_METHODS):
            acc= [np.mean([c["methods"][m]["correct"] for c in cases if c["target_tokens"] == n and c["position"] == p])
                  for p in ["start", "middle", "end"]]
            ax.bar(x + (i - 2.5) * w, acc, width=w * 0.9, color=KV_COLORS[m], label=m)

        ax.set_xticks(x, ["start", "middle", "end"])
        ax.set_title(f"nominal {n} tokens (30 cases per bar)")
        ax.set_xlabel("evidence position")
        clean_axes(ax)

    axes[0].set_ylabel("exact-match accuracy")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle("KV cache at matched INT4 budget: accuracy by evidence position")
    save_fig(fig, os.path.join(fig_dir, "kv_accuracy_by_position.png"))

    #selection masks for the six fixed cases
    fixed= [c for c in cases if c["id"].endswith("-2-00")]
    fig, axes= plt.subplots(len(fixed), 1, figsize=(12, 1.6 * len(fixed)), sharex=False)

    for ax, c in zip(np.atleast_1d(axes), fixed):
        for m, ls in [("evict_question", "-"), ("hybrid_int2", "--")]:
            k= np.asarray(c["methods"][m]["fp16_keep_count"], dtype=float)
            ax.plot(np.arange(len(k)), k / k.max() if k.max() else k, color=KV_COLORS[m], linewidth=1, linestyle=ls,
                    label=f"{m} (FP16)")

        ev= c["methods"]["evict_question"]["evidence_token_positions"]
        ax.axvspan(min(ev), max(ev), color="#eda100", alpha=0.35, label="gold evidence (eval only)")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{c['id']}  ({c['context_tokens']} context tokens)", fontsize=8)
        ax.tick_params(labelsize=7)

    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=7, ncol=3, loc="upper center")
    np.atleast_1d(axes)[-1].set_xlabel("context position")
    fig.supylabel("fraction of (layer, KV head) keeping position")
    fig.tight_layout()
    save_fig(fig, os.path.join(fig_dir, "kv_selection_masks.png"))


#---------------------------------------------------------------- summary + main
def summary_rows(agg, kv_rows, inf_rows):
    rows= []

    for a in agg:
        if a["condition"] != "trained mask":
            continue

        k= a["train_K"]
        rows.append({"part": "B", "item": a["run"], "metric": f"test K{k}_mixed acc (mean of {a['n_seeds']} seeds)",
                     "value": a[f"K{k}_mixed_mean"], "sd": a[f"K{k}_mixed_sd"]})

    for r in kv_rows:
        rows.append({"part": "C", "item": r["method"], "metric": "accuracy", "value": r["accuracy"], "sd": None})
        rows.append({"part": "C", "item": r["method"], "metric": "fraction_of_fp16_mean", "value": r["fraction_of_fp16_mean"],
                     "sd": None})

    for r in inf_rows:
        if r["T"] == max(x["T"] for x in inf_rows):
            rows.append({"part": "B", "item": r["kind"], "metric": f"forward ms at T={r['T']}",
                         "value": r["latency_ms_median"], "sd": None})

    return rows


def main(args):
    root= args.results_dir
    tdir= os.path.join(root, "tables")
    fig_dir= os.path.join(root, "figures")

    att, backends, tiles, edges= attention_tables(root)
    write_csv(os.path.join(tdir, "attention.csv"), att)
    write_csv(os.path.join(tdir, "attention_sdpa_backends.csv"), backends)
    write_csv(os.path.join(tdir, "attention_tile_sizes.csv"), tiles)
    write_csv(os.path.join(tdir, "attention_edge_cases.csv"), edges)
    attention_figures(att, fig_dir)

    runs= load_mqar(root)
    test, bins, curves, costs, agg= mqar_tables(runs)
    inf_rows= mqar_inference_table(root)
    write_csv(os.path.join(tdir, "mqar_test_per_seed.csv"), test)
    write_csv(os.path.join(tdir, "mqar_test_seed_mean.csv"), agg)
    write_csv(os.path.join(tdir, "mqar_distance_bins.csv"), bins)
    write_csv(os.path.join(tdir, "mqar_dev_curves.csv"), curves)
    write_csv(os.path.join(tdir, "mqar_theoretical_costs.csv"), costs)
    write_csv(os.path.join(tdir, "mqar_inference_sweep.csv"), inf_rows)
    write_csv(os.path.join(tdir, "mqar_b3_vs_linear_bootstrap.csv"), b3_bootstrap(runs))
    mqar_figures(runs, curves, bins, inf_rows, fig_dir)

    cases= load_cases(root)
    assert len(cases) == read_json(os.path.join(root, "kv", "summary.json"))["n_cases"], "case files missing"
    kv_rows= kv_main_table(cases)
    break_rows= kv_breakdowns(cases)
    lens, c1= kv_lengths_and_c1(cases)
    write_csv(os.path.join(tdir, "kv_main.csv"), kv_rows)
    write_csv(os.path.join(tdir, "kv_breakdowns.csv"), break_rows)
    write_csv(os.path.join(tdir, "kv_bootstrap.csv"), kv_bootstrap(cases))
    write_csv(os.path.join(tdir, "kv_lengths.csv"), lens)
    write_csv(os.path.join(tdir, "kv_c1_checks.csv"), c1)
    write_csv(os.path.join(tdir, "kv_bench.csv"), kv_bench_table(root))
    write_csv(os.path.join(tdir, "kv_fixed_case_generations.csv"), kv_generations(cases))
    kv_figures(cases, kv_rows, break_rows, fig_dir)

    write_csv(os.path.join(root, "summary.csv"), summary_rows(agg, kv_rows, inf_rows))


if __name__ == "__main__":
    parser= argparse.ArgumentParser()

    parser.add_argument(
        "--results_dir",
        default="results/task2",
        help="folder holding attention/, mqar/, mqar_inference/, kv/, kv_bench/"
    )

    args= parser.parse_args()
    main(args)
