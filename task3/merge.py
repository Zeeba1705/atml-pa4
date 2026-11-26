#task 3 step 6 (handout p.14): merge each lora adapter into a fresh fp16 0.5B backbone, check merged vs unmerged
#logits on a fixed prompt batch, save one merged checkpoint per (method, seed) to outputs/ (never committed)
import os
import time
import argparse

import torch
from peft import PeftModel

from common.io_utils import load_config, write_json, resolve_data_root, default_output_dir, run_info
from task3.load import load_tokenizer, load_model, prompt_ids, eval_prompts

N_CHECK_PROMPTS= 8   #ours: first 8 held-out prompts by order, logits only (no sampling)


def logits_for(model, tok, prompts, device):
    out= []

    with torch.no_grad():
        for p in prompts:
            ids= torch.tensor([prompt_ids(tok, p["text"])], device=device)
            out.append(model(input_ids=ids).logits[0].float())

    return out


def merged_dir(output_dir, name):
    return os.path.join(output_dir, "task3", "merged", name)


def main(args):
    cfg= load_config(args.config)
    device= torch.device(args.device)
    print("Using device:", device)

    out_dir= os.path.join(args.output_dir, "smoke") if args.smoke else args.output_dir
    res_root= os.path.join(out_dir, "results") if args.smoke else args.results_dir
    adapter= os.path.join(out_dir, "task3", "adapters", args.name)
    dest= merged_dir(out_dir, args.name)
    res_path= os.path.join(res_root, "task3", "merge", f"{args.name}.json")

    if os.path.exists(res_path) and os.path.exists(dest):
        print(f"{args.name}: merged, skipping")
        return

    data_root= resolve_data_root(cfg["data_repo"], args.data_root)
    prompts= eval_prompts(cfg, data_root)[:N_CHECK_PROMPTS]
    tok= load_tokenizer(cfg, "draft")

    start= time.perf_counter()
    base= load_model(cfg, "draft", device, args.smoke)
    model= PeftModel.from_pretrained(base, adapter).eval()
    unmerged= logits_for(model, tok, prompts, device)

    t= time.perf_counter()
    merged= model.merge_and_unload()
    merge_seconds= time.perf_counter() - t
    after= logits_for(merged, tok, prompts, device)

    errs= [(a - b).abs().max().item() for a, b in zip(unmerged, after)]
    merged.save_pretrained(dest)
    tok.save_pretrained(dest)

    info= run_info(cfg, device, next(merged.parameters()).dtype, model_key="draft")
    info.update({
        "name": args.name, "adapter_dir": adapter, "merged_dir": dest, "smoke": args.smoke,
        "check_prompts": [p["id"] for p in prompts],
        "max_abs_logit_error": max(errs), "per_prompt_max_abs_logit_error": errs,
        "merge_seconds": merge_seconds, "total_seconds": time.perf_counter() - start,
        "merged_params": sum(p.numel() for p in merged.parameters())
    })
    write_json(res_path, info)
    print(f"{args.name}: merged in {merge_seconds:.2f}s, max |logit diff| merged vs unmerged {max(errs):.3e} -> {dest}")


if __name__ == "__main__":
    parser= argparse.ArgumentParser()

    parser.add_argument(
        "--name",
        type=str,
        required=True
    )
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
