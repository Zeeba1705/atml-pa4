#task 3 step 3 off-policy pool (handout p.12, p.14; decision 3.4): one frozen teacher response per training prompt
#(500 gsm8k train + 300 short-form train), temperature 1, top-p 1, caps 192 math / 32 short, eos-aware.
#training prompts only; held-out prompts never enter (asserted in task3.load.train_prompts)
import os
import time
import argparse

import torch

from common.seed import set_seed
from common.io_utils import load_config, write_json, write_jsonl, resolve_data_root, default_output_dir, run_info
from task3.load import load_tokenizer, load_model, prompt_ids, train_prompts
from task3.spec_sampler import HFRunner, target_generate

POOL_SEED_OFFSET= 999   #ours: rollout seed = 6304 + 10007 * i + 999 for training prompt i


def pool_path(output_dir):
    return os.path.join(output_dir, "task3", "teacher_pool.jsonl.gz")


def main(args):
    cfg= load_config(args.config)
    set_seed(cfg["seed"])
    device= torch.device(args.device)
    print("Using device:", device)

    out_dir= os.path.join(args.output_dir, "smoke") if args.smoke else args.output_dir
    res_dir= os.path.join(out_dir, "results") if args.smoke else args.results_dir
    path= pool_path(out_dir)

    if os.path.exists(path):
        print(f"{path} exists, skipping")
        return

    data_root= resolve_data_root(cfg["data_repo"], args.data_root)
    prompts= train_prompts(cfg, data_root)
    caps= dict(cfg["response_caps"])

    if args.smoke:
        prompts= prompts[:2] + prompts[500:502]
        caps= {"math": 6, "short": 4}

    tok= load_tokenizer(cfg, "target")
    teacher= load_model(cfg, "target", device, args.smoke)
    eos_ids= cfg["eos_token_ids"]
    rows= []
    start= time.perf_counter()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for i, p in enumerate(prompts):
        ids= prompt_ids(tok, p["text"])
        seed= 6304 + 10007 * i + POOL_SEED_OFFSET
        gen= torch.Generator(device=device).manual_seed(seed)

        t= time.perf_counter()
        out, stats= target_generate(HFRunner(teacher, device, cfg["sampling"]["temperature"]), ids, caps[p["category"]], eos_ids, gen)

        rows.append({
            "source_id": p["id"], "category": p["category"], "rollout_seed": seed, "prompt_ids": ids,
            "response_ids": out, "termination": stats["stop"], "n_tokens": len(out), "seconds": time.perf_counter() - t
        })

        if (i + 1) % 100 == 0:
            print(f"teacher pool {i + 1}/{len(prompts)}, {time.perf_counter() - start:.0f}s")

    write_jsonl(path, rows)
    total= time.perf_counter() - start

    info= run_info(cfg, device, next(teacher.parameters()).dtype, model_key="target")
    info.update({
        "n_prompts": len(rows), "caps": caps, "seed_offset": POOL_SEED_OFFSET, "path": path, "smoke": args.smoke,
        "tokens": {c: sum(r["n_tokens"] for r in rows if r["category"] == c) for c in caps},
        "eos_fraction": {c: sum(r["termination"] == "eos" for r in rows if r["category"] == c) / max(1, sum(r["category"] == c for r in rows)) for c in caps},
        "teacher_generation_seconds": total,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    })
    write_json(os.path.join(res_dir, "task3", "teacher_pool_info.json"), info)
    print(f"teacher pool: {len(rows)} responses, {info['tokens']} tokens, {total:.0f}s -> {path}")


if __name__ == "__main__":
    parser= argparse.ArgumentParser()

    parser.add_argument(
        "--config",
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
