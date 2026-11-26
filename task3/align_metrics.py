#task 3 step 5 (handout p.14): target vs draft next-token distributions on fixed prefix banks. for each saved prefix
#h = response_prefix_token_ids (= seq[:pos-1]) the target p(.|h) and each draft q(.|h) give forward kl(p||q),
#reverse kl(q||p), tv, greedy agreement and expected one-token acceptance 1 - tv, at temperature 1 over the full vocab.
#every draft is scored on every bank (same prefix ids, positions and one-token-per-record denominators)
import os
import time
import argparse
from collections import defaultdict

import torch
import torch.nn.functional as F

from common.seed import set_seed
from common.io_utils import write_json, write_jsonl, read_jsonl, resolve_data_root, default_output_dir, run_info
from task3.load import load_tokenizer, load_model, prompt_ids, eval_prompts
from task3.spec_eval import load_cfgs, load_merged
from task3.prefix_banks import bank_path

POSITION_BINS= [(1, 16), (17, 32), (33, 64), (65, 192)]
STUDENTS= ["off_policy_s6304", "off_policy_s6305", "on_policy_s6304", "on_policy_s6305"]


def position_bin(pos):
    for lo, hi in POSITION_BINS:
        if lo <= pos <= hi:
            return f"{lo}-{hi}"

    raise ValueError(pos)


def pair_metrics(logits_p, logits_q):
    #rows = prefixes, columns = full vocab; fp32, temperature 1
    logp= F.log_softmax(logits_p.float(), dim=-1)
    logq= F.log_softmax(logits_q.float(), dim=-1)
    p, q= logp.exp(), logq.exp()

    return {
        "fkl": (p * (logp - logq)).sum(-1),
        "rkl": (q * (logq - logp)).sum(-1),
        "tv": 0.5 * (p - q).abs().sum(-1),
        "greedy_agree": (logp.argmax(-1) == logq.argmax(-1)).float(),
        "target_top": logp.argmax(-1),
        "draft_top": logq.argmax(-1)
    }


def group_prefixes(records):
    #all records of one (bank, prompt) are prefixes of one source sequence: keep the longest, check the others
    longest= max(records, key=lambda r: len(r["response_prefix_token_ids"]))["response_prefix_token_ids"]

    for r in records:
        pre= r["response_prefix_token_ids"]
        assert len(pre) == r["position"] - 1 and longest[:len(pre)] == pre, (r["prompt_id"], r["position"])

    return longest


def rows_at(model, ids, rows, device):
    #logits at absolute rows only (lm_head applied to the selected hidden states)
    with torch.no_grad():
        h= model.model(input_ids=torch.tensor([ids], device=device)).last_hidden_state[0]

        return model.lm_head(h[rows])


def load_banks(cfg, data_root, out_dir, wanted):
    #supplied banks (teacher, original_draft) from the data repo + our generated banks from outputs/
    banks= defaultdict(list)
    missing= []
    supplied= os.path.join(data_root, cfg["data"]["prefix_banks"])

    if os.path.exists(supplied):
        for r in read_jsonl(supplied):
            banks[f"supplied_{r['bank']}"].append(r)
    else:
        missing.append(supplied)

    for name in wanted:
        path= bank_path(out_dir, name)

        if os.path.exists(path):
            banks[name]= list(read_jsonl(path))
        else:
            missing.append(path)

    return dict(banks), missing


def main(args):
    cfg= load_cfgs(args)
    set_seed(cfg["seed"])
    device= torch.device(args.device)
    print("Using device:", device)

    out_dir= os.path.join(args.output_dir, "smoke") if args.smoke else args.output_dir
    res_root= os.path.join(out_dir, "results") if args.smoke else args.results_dir
    res_dir= os.path.join(res_root, "task3", "align")
    students= ["smoke_student"] if args.smoke else STUDENTS
    drafts= ["original"] + students

    data_root= resolve_data_root(cfg["data_repo"], args.data_root)
    prompts= {p["id"]: p for p in eval_prompts(cfg, data_root)}
    generated= list(cfg["banks"]["generated"]) + [f"student_{s}" for s in (STUDENTS if not args.smoke else [])]
    banks, missing= load_banks(cfg, data_root, out_dir, generated)

    for m in missing:
        print("MISSING BANK (skipped, rerun when present):", m)

    todo= {b: rs for b, rs in banks.items() if not os.path.exists(os.path.join(res_dir, f"{b}.jsonl.gz"))}

    if not todo:
        print("all available banks scored, skipping")
        return

    tok= load_tokenizer(cfg, "target")
    target= load_model(cfg, "target", device, args.smoke)
    models= {"original": load_model(cfg, "draft", device, args.smoke)}

    for s in students:
        models[s]= load_model(cfg, "draft", device, True, seed=3) if args.smoke else load_merged(os.path.join(out_dir, "task3", "merged", s), device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for bank, records in todo.items():
        start= time.perf_counter()
        by_prompt= defaultdict(list)

        for r in records:
            #held-out prompts only, identical text to our formatting input
            assert r["prompt_id"] in prompts and r["prompt"] == prompts[r["prompt_id"]]["text"], r["prompt_id"]
            by_prompt[r["prompt_id"]].append(r)

        out= []

        for k, (pid, rs) in enumerate(by_prompt.items()):
            rs= sorted(rs, key=lambda r: r["position"])
            prompt= prompt_ids(tok, prompts[pid]["text"])
            longest= group_prefixes(rs)
            ids= prompt + longest
            #response token at 1-based position pos sits at index len(prompt) + pos - 1, predicted by the row before it
            rows= [len(prompt) + r["position"] - 2 for r in rs]
            lp= rows_at(target, ids, rows, device)

            for name in drafts:
                m= pair_metrics(lp, rows_at(models[name], ids, rows, device))

                for i, r in enumerate(rs):
                    out.append({
                        "bank": bank, "prompt_id": pid, "split": prompts[pid]["split"], "position": r["position"],
                        "position_bin": position_bin(r["position"]), "draft": name,
                        "producing_checkpoint": r.get("producing_checkpoint"),
                        **{key: (int(v[i]) if key.endswith("top") else float(v[i])) for key, v in m.items()}
                    })

            if (k + 1) % 30 == 0:
                print(f"{bank}: {k + 1}/{len(by_prompt)} prompts, {time.perf_counter() - start:.0f}s")

        write_jsonl(os.path.join(res_dir, f"{bank}.jsonl.gz"), out)
        n= len(records)

        for name in drafts:
            rs= [o for o in out if o["draft"] == name]
            print(f"{bank} / {name}: n={len(rs)} fkl {sum(o['fkl'] for o in rs) / n:.4f} tv {sum(o['tv'] for o in rs) / n:.4f} "
                  f"1-tv {1 - sum(o['tv'] for o in rs) / n:.4f} greedy {sum(o['greedy_agree'] for o in rs) / n:.4f}")

    info= run_info(cfg, device, next(target.parameters()).dtype, model_key="target")
    info.update({
        "drafts": drafts, "banks_scored": sorted(todo), "banks_missing": missing, "smoke": args.smoke,
        "metrics": "fkl = KL(p_target || q_draft), rkl = KL(q || p), tv, greedy agreement, 1 - tv; temperature 1, full vocab, fp32",
        "denominator": "one next-token distribution per bank record (token-weighted means)",
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    })
    write_json(os.path.join(res_dir, f"run_info_{int(time.time())}.json"), info)


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
