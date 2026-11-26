#task 3 step 7 (handout p.14): standalone draft quality, the draft as a model (not speculative output). 120 held-out
#prompts, temperature 1 / top-p 1 sampling, 2 paired reps (decision 3.7), caps 512 math / 32 short, eos-aware.
#scoring rules are ours (decision 3.16): math answer = last \boxed{}, else after ####, else last number; short-form
#compliance = whole reply (trimmed) is only the requested field
import os
import re
import time
import argparse

import torch

from common.seed import set_seed
from common.io_utils import write_json, write_jsonl, read_jsonl, resolve_data_root, default_output_dir, run_info
from task3.load import load_tokenizer, load_model, prompt_ids, eval_prompts
from task3.spec_sampler import HFRunner, target_generate
from task3.spec_eval import load_cfgs, load_merged

STANDALONE_SEED_OFFSET= 7000   #ours: seed = 6304 + 10007 * j + 7000 + 100 * rep, identical for every model (paired)
MONTHS= ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
FAMILY_PATTERNS= {0: re.compile(r"(?i)(" + "|".join(MONTHS) + r")"), 1: re.compile(r"\d+"), 2: re.compile(r"[A-Z]{2}")}
NUMBER= re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def to_number(s):
    s= s.replace(",", "").replace("$", "").strip().rstrip(".")

    try:
        return float(s)
    except ValueError:
        return None


def extract_math(text):
    #returns (number or None, how it was found)
    boxed= re.findall(r"\\boxed\{([^{}]*)\}", text)

    if boxed:
        nums= NUMBER.findall(boxed[-1])
        if nums:
            return to_number(nums[-1]), "boxed"

    if "####" in text:
        nums= NUMBER.findall(text.split("####")[-1])
        if nums:
            return to_number(nums[0]), "hashes"

    nums= NUMBER.findall(text)

    if nums:
        return to_number(nums[-1]), "last_number"

    return None, "none"


def score_math(text, answer):
    pred, how= extract_math(text)
    gold= to_number(answer)
    assert gold is not None, answer

    return {"extracted": pred, "extraction": how, "extractable": pred is not None,
            "correct": pred is not None and abs(pred - gold) < 1e-6}


def clean_short(text):
    #trim whitespace, one trailing period, bold markers, quotes/backticks, then a trailing period inside them
    s= text.strip()
    s= s[:-1].strip() if s.endswith(".") else s
    s= re.sub(r"^\*\*(.*)\*\*$", r"\1", s).strip()
    s= s.strip("\"'`").strip()

    return s[:-1].strip() if s.endswith(".") else s


def score_short(text, answer, family):
    s= clean_short(text)
    compliant= FAMILY_PATTERNS[family].fullmatch(s) is not None
    lenient= re.search(r"(?<![A-Za-z0-9])" + re.escape(answer) + r"(?![A-Za-z0-9])", text, flags=re.IGNORECASE) is not None

    return {"cleaned": s, "compliant": compliant, "correct": compliant and s.lower() == answer.lower(), "correct_lenient": lenient}


def score(p, text, stop):
    out= score_math(text, p["answer"]) if p["domain"] == "math" else score_short(text, p["answer"], p["family"])
    out["truncated"]= stop == "length"

    return out


def summarize(rows):
    def mean(xs):
        return sum(xs) / len(xs) if xs else None

    out= {}

    for split in ["math", "familiar", "shifted"]:
        rs= [r for r in rows if r["split"] == split]
        s= {"n": len(rs), "accuracy": mean([r["correct"] for r in rs]), "truncated": mean([r["truncated"] for r in rs]),
            "mean_tokens": mean([r["n_out"] for r in rs])}

        if split == "math":
            s["extractable"]= mean([r["extractable"] for r in rs])
            s["correct_and_not_truncated"]= mean([r["correct"] and not r["truncated"] for r in rs])
        else:
            s["compliant"]= mean([r["compliant"] for r in rs])
            s["accuracy_lenient"]= mean([r["correct_lenient"] for r in rs])

        out[split]= s

    return out


def main(args):
    cfg= load_cfgs(args)
    set_seed(cfg["seed"])
    device= torch.device(args.device)
    print("Using device:", device)

    out_dir= os.path.join(args.output_dir, "smoke") if args.smoke else args.output_dir
    res_root= os.path.join(out_dir, "results") if args.smoke else args.results_dir
    res_dir= os.path.join(res_root, "task3", "standalone", args.model_name)
    caps= {"math": 8, "short": 4} if args.smoke else dict(cfg["quality_max_new_tokens"])
    reps= cfg["paired_reps"]

    todo= [r for r in range(reps) if not os.path.exists(os.path.join(res_dir, f"r{r}.jsonl"))]

    if not todo:
        print(f"{args.model_name}: standalone done, skipping")
        return

    data_root= resolve_data_root(cfg["data_repo"], args.data_root)
    prompts= eval_prompts(cfg, data_root)

    if args.smoke:
        prompts= [prompts[0], prompts[60], prompts[90]]

    tok= load_tokenizer(cfg, "target")
    eos_ids= cfg["eos_token_ids"]

    #which model: the original draft, the target (reference), or a merged student checkpoint
    if args.draft_path:
        model= load_merged(args.draft_path, device)
    elif args.model_name == "target":
        model= load_model(cfg, "target", device, args.smoke)
    else:
        assert args.model_name == "original", "merged students need --draft_path"
        model= load_model(cfg, "draft", device, args.smoke)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start= time.perf_counter()

    for rep in todo:
        rows= []

        for j, p in enumerate(prompts):
            ids= prompt_ids(tok, p["text"])
            seed= 6304 + 10007 * j + STANDALONE_SEED_OFFSET + 100 * rep
            gen= torch.Generator(device=device).manual_seed(seed)

            t= time.perf_counter()
            out, stats= target_generate(HFRunner(model, device, cfg["temperature"]), ids, caps[p["domain"]], eos_ids, gen)
            seconds= time.perf_counter() - t
            text= tok.decode(out, skip_special_tokens=True)

            row= {"id": p["id"], "domain": p["domain"], "split": p["split"], "family": p.get("family"), "rep": rep,
                  "seed": seed, "cap": caps[p["domain"]], "n_out": len(out), "stop": stats["stop"], "seconds": seconds,
                  "answer": p["answer"], "response": text, "response_ids": out}
            row.update(score(p, text, stats["stop"]))
            rows.append(row)

            if (j + 1) % 20 == 0:
                print(f"{args.model_name} rep {rep}: {j + 1}/{len(prompts)} prompts, {time.perf_counter() - start:.0f}s")

        write_jsonl(os.path.join(res_dir, f"r{rep}.jsonl"), rows)
        s= summarize(rows)
        print(f"{args.model_name} rep {rep}: " + ", ".join(f"{k} acc {v['accuracy']:.3f}" for k, v in s.items()))

    all_rows= [r for rep in range(reps) for r in read_jsonl(os.path.join(res_dir, f"r{rep}.jsonl"))]
    info= run_info(cfg, device, next(model.parameters()).dtype, model_key="draft")
    info.update({
        "model_name": args.model_name, "draft_path": args.draft_path, "smoke": args.smoke, "caps": caps, "reps": reps,
        "seed_rule": f"6304 + 10007*j + {STANDALONE_SEED_OFFSET} + 100*rep", "temperature": cfg["temperature"],
        "top_p": cfg["top_p"], "scoring": "decision 3.16", "generation_seconds_this_run": time.perf_counter() - start,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "by_split": summarize(all_rows),
        "by_split_per_rep": {rep: summarize([r for r in all_rows if r["rep"] == rep]) for rep in range(reps)}
    })
    write_json(os.path.join(res_dir, "summary.json"), info)


if __name__ == "__main__":
    parser= argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        required=True
    )
    parser.add_argument(
        "--draft_path",
        type=str,
        default=None
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
