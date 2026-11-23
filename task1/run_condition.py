import os
import time
import argparse

import torch

from common.seed import set_seed
from common.io_utils import load_config, default_output_dir, run_info, write_json, write_jsonl
from task1.load import load_model, load_tokenizer, load_windows, load_hellaswag
from task1.evaluate_ppl import evaluate_ppl
from task1.evaluate_hellaswag import evaluate_hellaswag
from task1.resources import storage_counts, measure_latency, reset_peak, peak_memory

EVALS= ["wikitext2", "c4_val", "hellaswag", "resources"]


def condition_name(args):
    name= args.method

    return "smoke_" + name if args.smoke else name


def build_condition(model, args, cfg):
    #each condition starts from a fresh copy of the unchanged weights; compression methods plug in here
    start= time.perf_counter()

    if args.method != "baseline":
        raise NotImplementedError(args.method)

    return {"compression_seconds": time.perf_counter() - start, "calibration_seconds": 0.0}


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

    tok= load_tokenizer(cfg)
    model= load_model(cfg, device, args.smoke)
    dtype= next(model.parameters()).dtype

    info= run_info(cfg, device, dtype, model_key="task1_base")
    info["config"]= cfg
    info["condition"]= name
    info["smoke"]= args.smoke

    build= build_condition(model, args, cfg)
    write_json(os.path.join(out_dir, "condition.json"), {**info, **build, "storage": storage_counts(model)})

    n_small= cfg["smoke"]["max_examples"]

    for eval_name in todo:
        reset_peak(device)
        start= time.perf_counter()

        if eval_name in ["wikitext2", "c4_val"]:
            windows= load_windows(cfg, args.data_root, eval_name)

            if args.smoke:
                windows= windows[:n_small]

            rows, summary= evaluate_ppl(model, windows, device)

            #the scored set must be identical across conditions (p.3)
            if not args.smoke:
                assert summary["n_scored"] == cfg["ppl_scored_tokens"]

            print(f"{eval_name}: ppl {summary['ppl']:.4f} over {summary['n_scored']} tokens")

        elif eval_name == "hellaswag":
            rows_in= load_hellaswag(cfg, args.data_root)
            full= args.method in cfg["hellaswag"]["full_conditions"]
            n= cfg["hellaswag"]["n_full"] if full else cfg["hellaswag"]["n_diagnostic"]

            if args.smoke:
                n= n_small

            rows, summary= evaluate_hellaswag(model, tok, rows_in, n, device)
            print(f"hellaswag: acc {summary['acc']:.4f}, acc_norm {summary['acc_norm']:.4f} on {summary['n_examples']}")

        else:
            windows= load_windows(cfg, args.data_root, "c4_val")
            res= cfg["resources"]
            lengths= [16, 32] if args.smoke else res["prefill_lengths"]
            decode= 4 if args.smoke else res["decode_tokens"]

            rows= None
            summary= {
                "latency": measure_latency(model, windows, cfg, device, lengths, decode),
                "batch_size": res["batch_size"],
                "storage": storage_counts(model)
            }

        summary["wall_clock_seconds"]= time.perf_counter() - start
        summary.update(peak_memory(device))

        if rows is not None:
            write_jsonl(os.path.join(out_dir, f"{eval_name}_examples.jsonl"), rows)

        #summary json last: its existence marks the eval as done
        write_json(os.path.join(out_dir, f"{eval_name}.json"), {**info, "eval": eval_name, **summary})


if __name__ == "__main__":
    defaults= load_config(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "task1", "base.yaml"))
    parser= argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/task1/base.yaml"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["baseline"],
        default="baseline"
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
        default=defaults["data_repo"]
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
