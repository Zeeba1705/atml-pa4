#task 3 generated prefix banks (handout p.13; decisions G1, 3.1, 3.8). same prompts, position rule and rng string as
#setup_data.py's supplied banks: positions from bins [1,16],[17,32],[33,64],[65,192] (8, 8, 16, 16 per prompt),
#record = response_prefix_token_ids = seq[:pos-1]. banks live in outputs/ (never committed); results/ gets an info file
import os
import time
import random
import hashlib
import argparse

import yaml
import torch

from common.seed import set_seed
from common.io_utils import load_config, write_json, write_jsonl, resolve_data_root, default_output_dir, run_info
from task3.load import load_tokenizer, load_model, prompt_ids, eval_prompts
from task3.spec_sampler import HFRunner, speculative_generate, target_generate
from task3.spec_eval import load_cfgs, load_merged


def select_positions(seq_len, pid, bank, bins):
    #identical to setup_data.prefix_banks
    rng= random.Random(f"PA4-banks-v1:6304:{pid}:{bank}")
    positions= []

    for (lo, hi), n in bins:
        candidates= list(range(lo, min(hi, seq_len) + 1))
        positions.extend(sorted(rng.sample(candidates, min(n, len(candidates)))))

    return positions


def bank_records(pid, domain, bank, prompt, seq, seed, bins, producer):
    return [
        {"prompt_id": pid, "domain": domain, "bank": bank, "position": pos, "prompt": prompt,
         "response_prefix_token_ids": seq[:pos - 1], "source_length": len(seq), "source_generation_seed": seed,
         "producing_checkpoint": producer}
        for pos in select_positions(len(seq), pid, bank, bins)
    ]


def bank_path(output_dir, bank):
    return os.path.join(output_dir, "task3", "banks", f"{bank}.jsonl.gz")


def file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main(args):
    cfg= load_cfgs(args)
    set_seed(cfg["seed"])
    device= torch.device(args.device)
    print("Using device:", device)

    b= cfg["banks"]
    bins= list(zip([tuple(x) for x in b["position_bins"]], b["positions_per_bin"]))
    max_new= 8 if args.smoke else b["max_new_tokens"]
    out_dir= os.path.join(args.output_dir, "smoke") if args.smoke else args.output_dir
    res_dir= os.path.join(out_dir, "results") if args.smoke else args.results_dir

    path= bank_path(out_dir, args.bank)

    if os.path.exists(path):
        print(f"{path} exists, skipping")
        return

    data_root= resolve_data_root(cfg["data_repo"], args.data_root)
    prompts= eval_prompts(cfg, data_root)

    if args.smoke:
        prompts= [prompts[0], prompts[60], prompts[90]]

    tok= load_tokenizer(cfg, "target")
    eos_ids= cfg["eos_token_ids"]

    #which policy produces the prefixes: target alone, exact speculation with the original draft, or a student alone
    if args.bank == "teacher_pure":
        model= load_model(cfg, "target", device, args.smoke)
        producer= cfg["models"]["target"]
    elif args.bank == "exact_spec_original_gamma4":
        model= load_model(cfg, "target", device, args.smoke)
        draft= load_model(cfg, "draft", device, args.smoke)
        producer= f"{cfg['models']['target']} + {cfg['models']['draft']} (gamma {b['spec_bank_gamma']})"
    else:
        #student own-rollout bank: --bank student_<run>, --draft_path = that run's merged checkpoint
        assert args.bank.startswith("student_") and args.draft_path, "student banks need --draft_path"
        model= load_merged(args.draft_path, device)
        producer= args.draft_path

    offset= b["seed_offsets"].get(args.bank, args.seed_offset)
    records= []
    lengths= []
    stops= []
    start= time.perf_counter()

    for j, p in enumerate(prompts):
        ids= prompt_ids(tok, p["text"])
        seed= 6304 + 10007 * j + offset
        gen= torch.Generator(device=device).manual_seed(seed)

        if args.bank == "exact_spec_original_gamma4":
            seq, stats= speculative_generate(HFRunner(model, device), HFRunner(draft, device), ids, b["spec_bank_gamma"], max_new, eos_ids, gen)
        else:
            seq, stats= target_generate(HFRunner(model, device), ids, max_new, eos_ids, gen)

        records+= bank_records(p["id"], p["domain"], args.bank, p["text"], seq, seed, bins, producer)
        lengths.append(len(seq))
        stops.append(stats["stop"])

        if (j + 1) % 20 == 0:
            print(f"{args.bank}: {j + 1}/{len(prompts)} prompts, {time.perf_counter() - start:.0f}s")

    write_jsonl(path, records)

    info= run_info(cfg, device, next(model.parameters()).dtype, model_key="target")
    info.update({
        "bank": args.bank, "producing_checkpoint": producer, "n_prompts": len(prompts), "n_records": len(records),
        "seed_offset": offset, "max_new_tokens": max_new, "mean_response_length": sum(lengths) / len(lengths),
        "eos_fraction": stops.count("eos") / len(stops), "generation_seconds": time.perf_counter() - start,
        "path": path, "sha256": file_sha256(path), "smoke": args.smoke
    })
    write_json(os.path.join(res_dir, "task3", "banks", f"{args.bank}_info.json"), info)
    print(f"{args.bank}: {len(records)} records from {len(prompts)} prompts -> {path}")


if __name__ == "__main__":
    parser= argparse.ArgumentParser()

    parser.add_argument(
        "--bank",
        type=str,
        required=True
    )
    parser.add_argument(
        "--draft_path",
        type=str,
        default=None
    )
    parser.add_argument(
        "--seed_offset",
        type=int,
        default=0
    )
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
