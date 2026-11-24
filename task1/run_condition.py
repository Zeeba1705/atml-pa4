import os
import json
import time
import argparse

import torch

from common.seed import set_seed
from common.io_utils import load_config, default_output_dir, run_info, write_json, write_jsonl, resolve_data_root
from task1.load import eligible_linears, load_model, load_tokenizer, load_windows, load_hellaswag, load_sequences, data_path
from task1.calibrate import sequential_compress
from task1.prune import make_pruner
from task1.quantize import make_quantizer
from task1.evaluate_ppl import evaluate_ppl
from task1.evaluate_hellaswag import evaluate_hellaswag
from task1.resources import storage_counts, measure_latency, reset_peak, peak_memory
from task1.allocate import splits_dir, p3_path, q3_path, per_block, row_counts

EVALS= ["wikitext2", "c4_val", "hellaswag", "resources"]
METHODS= ["baseline", "p1_magnitude", "p2_wanda", "p2_wanda_2of4", "p3_nonuniform", "q1_uniform", "q2_awq", "q3_mixed_budget"]
QUANT_METHODS= ["q1_uniform", "q2_awq", "q3_mixed_budget"]
CALIBRATION_BATCH= 8   #ours, sequences per calibration forward


def condition_name(args):
    if args.method == "baseline":
        name= "baseline"
    elif args.method in QUANT_METHODS:
        name= f"{args.method}_b{args.bits}"
    else:
        name= f"{args.method}_s{args.sparsity:g}"

    return "smoke_" + name if args.smoke else name


def quant_storage(model, records):
    #serialized size: packed codes + zero-points + fp16 group scales, awq's unfolded fp16 scales, all other params in fp16
    counts= storage_counts(model)
    eligible= sum(r["packed_bytes"] for r in records)
    extra= sum(r["extra_fp16_bytes"] for r in records)
    nominal= sum(r["bits"] * r["shape"][0] * r["shape"][1] for r in records) / counts["eligible_weights"]

    return {
        "nominal_bits_per_eligible_weight": nominal,
        "packed_eligible_bytes": eligible,
        "unfolded_scale_bytes": extra,
        "effective_bits_per_eligible_weight": 8 * (eligible + extra) / counts["eligible_weights"],
        "packed_total_bytes": eligible + extra + 2 * counts["non_eligible_params"]
    }


def load_frozen(path, records_key):
    #p3/q3 allocations are frozen (task1.allocate) before any held-out result exists
    assert os.path.exists(path), f"missing frozen allocation {path}: run python -m task1.allocate first (README)"

    with open(path) as f:
        frozen= json.load(f)

    return frozen, [u[records_key] for u in frozen["units"]]


def build_condition(model, args, cfg, data_root, device, out_dir):
    #each condition starts from a freshly loaded, unchanged model
    if args.method == "baseline":
        return {"calibration_seconds": 0.0, "compression_seconds": 0.0, "total_build_seconds": 0.0}

    pruning= cfg["pruning"]
    quant= cfg["quantization"]
    base= splits_dir(args.output_dir, args.smoke)
    frozen= None

    if args.method in QUANT_METHODS:
        bits= args.bits

        if args.method == "q3_mixed_budget":
            frozen, unit_bits= load_frozen(q3_path(base, args.bits), "bits")
            bits= per_block(frozen, unit_bits)
        else:
            assert args.bits in quant["q_bits"]

        compress_fn= make_quantizer(
            method="uniform" if args.method == "q1_uniform" else "awq",
            bits=bits,
            group_size=quant["group_size"],
            scale_dtype=getattr(torch, quant["scale_dtype"]),
            alpha_grid=quant["awq_alpha_grid"],
            folds=quant["awq_folds"],
            unfolded=quant["awq_unfolded_fp16_scales"]
        )
    elif args.method == "p1_magnitude":
        assert args.sparsity in pruning["p1_sparsities"]
        compress_fn= make_pruner("magnitude", args.sparsity)
    elif args.method == "p2_wanda":
        assert args.sparsity in pruning["p2_sparsities"]
        compress_fn= make_pruner("wanda", args.sparsity)
    elif args.method == "p3_nonuniform":
        assert args.sparsity in pruning["p3_sparsities"]
        frozen, zeros= load_frozen(p3_path(base, args.sparsity), "zeros")
        rows= [row_counts(z, u["shape"][0]) for z, u in zip(zeros, frozen["units"])]
        compress_fn= make_pruner("wanda", args.sparsity, row_zeros=per_block(frozen, rows))
    else:
        assert args.sparsity == pruning["two_four"]["sparsity"]
        compress_fn= make_pruner("wanda", args.sparsity, two_four=True)

    #calibration = the 128 c4 calibration sequences only (never dev or eval data)
    seqs= load_sequences(cfg, data_root, "calibration", cfg["calibration_sequences"])

    if args.smoke:
        seqs= [{"token_ids": s["token_ids"][:64]} for s in seqs[:cfg["smoke"]["max_examples"]]]

    start= time.perf_counter()
    records, times= sequential_compress(model, seqs, device, CALIBRATION_BATCH, compress_fn, cache_inputs=(args.method in ["q2_awq", "q3_mixed_budget"]))
    times["total_build_seconds"]= time.perf_counter() - start

    if args.method in QUANT_METHODS:
        times["storage_extra"]= quant_storage(model, records)

    if args.method == "q3_mixed_budget":
        #matched budget (p.4 step 7): realized bytes must fit the q2 uniform budget and equal the frozen plan
        realized= times["storage_extra"]["packed_total_bytes"]
        assert realized <= frozen["budget_total_bytes"], (realized, frozen["budget_total_bytes"])
        assert realized == frozen["packed_total_bytes"], (realized, frozen["packed_total_bytes"])

    if args.method == "p3_nonuniform":
        #matched to p1/p2 by the total number of zeroed eligible weights (p.4 step 4); make_pruner asserts
        #the pruned count per projection, zeros after the build can only add the base model's own exact zeros
        zeros_now= sum(int((m.weight == 0).sum().item()) for _, m in eligible_linears(model))
        times["matched_zero_check"]= {"allocated_zeros": frozen["total_zeros"], "zeros_after_build": zeros_now}
        assert zeros_now >= frozen["total_zeros"]

    if frozen is not None:
        times["frozen_allocation"]= {k: frozen[k] for k in ["sensitivity_sha256", "git_commit"]}

    write_jsonl(os.path.join(out_dir, "compression_layers.jsonl"), records)
    print(f"built {condition_name(args)}: {times['total_build_seconds']:.1f}s")

    return times


def main(args):
    cfg= load_config(args.config)
    set_seed(cfg["seed"])

    device= torch.device(args.device)
    print("Using device:", device)

    name= condition_name(args)

    #smoke results never go into results/
    results_dir= os.path.join(args.output_dir, "smoke", "task1") if args.smoke else args.results_dir
    out_dir= os.path.join(results_dir, name)
    todo= [e for e in args.evals if not os.path.exists(os.path.join(out_dir, f"{e}.json"))]

    print(f"condition {name}: done {[e for e in args.evals if e not in todo]}, to run {todo}")

    if not todo:
        return

    #check every data file before spending time on the model
    data_root= resolve_data_root(cfg["data_repo"], args.data_root)
    needed= {"wikitext2": ["wikitext2"], "c4_val": ["c4_val"], "hellaswag": ["hellaswag"], "resources": ["c4_val"]}

    for e in todo:
        for key in needed[e]:
            data_path(cfg, data_root, key)

    print("data root:", data_root)

    tok= load_tokenizer(cfg)
    model= load_model(cfg, device, args.smoke)
    dtype= next(model.parameters()).dtype

    info= run_info(cfg, device, dtype, model_key="task1_base")
    info["config"]= cfg
    info["condition"]= name
    info["smoke"]= args.smoke

    reset_peak(device)
    build= build_condition(model, args, cfg, data_root, device, out_dir)
    build.update({"build_" + k: v for k, v in peak_memory(device).items()})
    write_json(os.path.join(out_dir, "condition.json"), {**info, **build, "storage": storage_counts(model)})

    n_small= cfg["smoke"]["max_examples"]

    for eval_name in todo:
        reset_peak(device)
        start= time.perf_counter()

        if eval_name in ["wikitext2", "c4_val"]:
            windows= load_windows(cfg, data_root, eval_name)

            if args.smoke:
                windows= windows[:n_small]

            rows, summary= evaluate_ppl(model, windows, device)

            #the scored set must be identical across conditions (p.3)
            if not args.smoke:
                assert summary["n_scored"] == cfg["ppl_scored_tokens"]

            print(f"{eval_name}: ppl {summary['ppl']:.4f} over {summary['n_scored']} tokens")

        elif eval_name == "hellaswag":
            rows_in= load_hellaswag(cfg, data_root)
            full= name.removeprefix("smoke_") in cfg["hellaswag"]["full_conditions"]
            n= cfg["hellaswag"]["n_full"] if full else cfg["hellaswag"]["n_diagnostic"]

            if args.smoke:
                n= n_small

            rows, summary= evaluate_hellaswag(model, tok, rows_in, n, device)
            print(f"hellaswag: acc {summary['acc']:.4f}, acc_norm {summary['acc_norm']:.4f} on {summary['n_examples']}")

        else:
            windows= load_windows(cfg, data_root, "c4_val")
            res= cfg["resources"]
            lengths= [16, 32] if args.smoke else res["prefill_lengths"]
            decode= 4 if args.smoke else res["decode_tokens"]

            rows= None
            summary= {
                "latency": measure_latency(model, windows, cfg, device, lengths, decode),
                "batch_size": res["batch_size"],
                "storage": {**storage_counts(model), **build.get("storage_extra", {})}
            }

        summary["wall_clock_seconds"]= time.perf_counter() - start
        summary.update(peak_memory(device))

        if rows is not None:
            write_jsonl(os.path.join(out_dir, f"{eval_name}_examples.jsonl"), rows)

        #summary json last: its existence marks the eval as done
        write_json(os.path.join(out_dir, f"{eval_name}.json"), {**info, "eval": eval_name, **summary})


if __name__ == "__main__":
    parser= argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/task1/base.yaml"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=METHODS,
        default="baseline"
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.5
    )
    parser.add_argument(
        "--evals",
        type=str,
        nargs="+",
        choices=EVALS,
        default=EVALS
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
        default=os.path.join("results", "task1")
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
