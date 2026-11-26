#task 3 steps 3-4 (handout pp.12-14): off-policy and on-policy forward-kl distillation of the 0.5B student into lora
#adapters. 40,000 teacher-scored response tokens (30,000 math + 10,000 short), exactly 256 per update (short final
#update), kl over the full vocabulary normalized by the update's supervised tokens (decisions 3.3-3.5)
import os
import json
import time
import random
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

from common.seed import set_seed
from common.io_utils import load_config, read_jsonl, write_json, write_jsonl, resolve_data_root, default_output_dir, run_info
from task3.load import load_tokenizer, load_model, prompt_ids, train_prompts
from task3.spec_sampler import HFRunner, target_generate
from task3.teacher_pool import pool_path

CATS= ["math", "short"]
CKPT_EVERY= 10   #updates between resume checkpoints


def forward_kl_sum(p, logq):
    #sum over positions of KL(p_T || q) = sum_v p (log p - log q); 0 log 0 = 0
    return (torch.special.xlogy(p, p) - p * logq).sum()


def category_perms(categories, seed):
    #fixed seeded permutation of prompt indices within each quota category (decision 3.3)
    perms= {}

    for c in CATS:
        idx= [i for i, x in enumerate(categories) if x == c]
        random.Random(f"PA4-distill-schedule:{seed}:{c}").shuffle(idx)
        perms[c]= idx

    return perms


def pick_presentation(state, perms, quotas):
    #alternate math/short; once a category's quota is met only the other continues; cycle each permutation
    open_cats= [c for c in CATS if state["scored"][c] < quotas[c]]

    if not open_cats:
        return None

    c= state["next"] if state["next"] in open_cats else open_cats[0]
    idx= perms[c][state["pos"][c] % len(perms[c])]
    state["pos"][c]+= 1
    state["next"]= "short" if c == "math" else "math"

    return c, idx


def next_chunk(remaining, chunk, total, size):
    #next piece of a response for the open update: exactly `size` scored tokens per update, the last one shorter
    update_size= min(size, total - (chunk["done"] - chunk["acc"]))
    k= min(remaining, update_size - chunk["acc"])
    out= {"k": k, "update": chunk["update"], "update_size": update_size, "closes": chunk["acc"] + k == update_size}

    chunk["acc"]+= k
    chunk["done"]+= k

    if out["closes"]:
        chunk["acc"]= 0
        chunk["update"]+= 1

    return out


def update_chunks(lengths, total, size):
    chunk= {"acc": 0, "done": 0, "update": 0}
    out= []

    for i, n in enumerate(lengths):
        off= 0

        while off < n:
            c= next_chunk(n - off, chunk, total, size)
            out.append({"presentation": i, "offset": off, **c})
            off+= c["k"]

    return out


def add_lora(model, lcfg, seed):
    #lora init seeded by the training seed; peft keeps adapters in fp32 on an fp16 backbone (autocast_adapter_dtype)
    torch.manual_seed(seed)
    config= LoraConfig(
        r=lcfg["r"],
        lora_alpha=lcfg["alpha"],
        lora_dropout=lcfg["dropout"],
        target_modules=lcfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    return get_peft_model(model, config)


def teacher_probs(teacher, prompt, resp, device):
    #row i = teacher distribution for response token i given prompt + resp[:i]
    ids= torch.tensor([(prompt + resp)[:-1]], device=device)

    with torch.no_grad():
        logits= teacher(input_ids=ids, logits_to_keep=len(resp)).logits[0]

    return torch.softmax(logits.float(), dim=-1)


def student_log_probs(student, prompt, resp, offset, k, device, autocast):
    #log q for response tokens offset .. offset+k-1 (needs the prefix up to the token before the last one)
    ids= torch.tensor([(prompt + resp)[:len(prompt) + offset + k - 1]], device=device)

    with torch.autocast(device.type, dtype=torch.float16, enabled=autocast):
        logits= student(input_ids=ids, logits_to_keep=k).logits[0]

    return F.log_softmax(logits.float(), dim=-1)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_state(s):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])

    if s["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def main(args):
    cfg= load_config(args.config)
    assert args.method in cfg["methods"] and args.seed in cfg["seeds"]
    set_seed(args.seed)
    device= torch.device(args.device)
    print("Using device:", device)

    name= f"{args.method}_s{args.seed}"
    out_dir= os.path.join(args.output_dir, "smoke") if args.smoke else args.output_dir
    res_root= os.path.join(out_dir, "results") if args.smoke else args.results_dir
    res_dir= os.path.join(res_root, "task3", "distill", name)
    ckpt_path= os.path.join(out_dir, "task3", "distill", name, "last.pt")
    adapter_dir= os.path.join(out_dir, "task3", "adapters", name)

    if os.path.exists(os.path.join(res_dir, "summary.json")):
        print(f"{name}: done, skipping")
        return

    data_root= resolve_data_root(cfg["data_repo"], args.data_root)
    prompts= train_prompts(cfg, data_root)
    quotas= {"math": cfg["quota"]["math_tokens"], "short": cfg["quota"]["short_tokens"]}
    caps= dict(cfg["response_caps"])

    size= cfg["update_every_scored_tokens"]
    ckpt_every= CKPT_EVERY

    if args.smoke:
        #same 4 prompts as the smoke teacher pool; several small updates so checkpoint/resume runs
        prompts= prompts[:2] + prompts[500:502]
        quotas, caps, size, ckpt_every= {"math": 20, "short": 8}, {"math": 6, "short": 4}, 8, 2

    total= sum(quotas.values())
    assert total == cfg["quota"]["total_tokens"] or args.smoke

    tok= load_tokenizer(cfg, "target")
    teacher= load_model(cfg, "target", device, args.smoke)
    student= add_lora(load_model(cfg, "draft", device, args.smoke), cfg["lora"], args.seed)
    params= [p for p in student.parameters() if p.requires_grad]
    assert all(p.dtype == torch.float32 for p in params), "adapters must be fp32"

    pool= None

    if args.method == "off_policy":
        pool= {r["source_id"]: r for r in read_jsonl(pool_path(out_dir))}
        assert all(p["id"] in pool for p in prompts) or args.smoke

    use_amp= device.type == "cuda"
    optimizer= torch.optim.AdamW(params, lr=cfg["optimizer"]["lr"], weight_decay=cfg["optimizer"]["weight_decay"])
    scaler= torch.amp.GradScaler(device.type, enabled=use_amp)
    perms= category_perms([p["category"] for p in prompts], args.seed)

    state= {"scored": {c: 0 for c in CATS}, "next": "math", "pos": {c: 0 for c in CATS},
            "chunk": {"acc": 0, "done": 0, "update": 0}, "k": 0, "current": None, "kl_acc": 0.0}
    presentations= []
    updates= []
    times= {"teacher_scoring": 0.0, "student_rollout": 0.0, "student_forward_backward": 0.0, "optimizer_step": 0.0}

    if os.path.exists(ckpt_path):
        ckpt= torch.load(ckpt_path, map_location=device, weights_only=False)
        set_peft_model_state_dict(student, ckpt["adapter"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        set_rng_state(ckpt["rng"])
        state, presentations, updates, times= ckpt["state"], ckpt["presentations"], ckpt["updates"], ckpt["times"]
        print(f"resumed {name} at update {state['chunk']['update']}, {state['chunk']['done']} scored tokens")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start= time.perf_counter()
    eos_ids= cfg["eos_token_ids"]
    P= None

    while True:
        cur= state["current"]

        if cur is None:
            pick= pick_presentation(state, perms, quotas)

            if pick is None:
                break

            c, idx= pick
            p= prompts[idx]
            ids= prompt_ids(tok, p["text"])

            if args.method == "off_policy":
                row= pool[p["id"]]
                resp, term, rseed= row["response_ids"], row["termination"], row["rollout_seed"]
            else:
                #fresh rollout from the current student, no gradient through generation
                rseed= args.seed * 1000003 + state["k"]
                gen= torch.Generator(device=device).manual_seed(rseed)
                t= time.perf_counter()
                student.eval()
                resp, st= target_generate(HFRunner(student, device, cfg["sampling"]["temperature"]), ids, caps[c], eos_ids, gen)
                term= st["stop"]
                times["student_rollout"]+= time.perf_counter() - t

            #a quota reached mid-response: score only the permitted prefix
            n= min(len(resp), quotas[c] - state["scored"][c])
            state["scored"][c]+= n
            presentations.append({"k": state["k"], "source_id": p["id"], "category": c, "rollout_seed": rseed,
                                  "response_ids": resp, "termination": term, "n_response": len(resp), "n_scored": n,
                                  "truncated_by_quota": n < len(resp), "update_at_start": state["chunk"]["update"]})
            cur= {"ids": ids, "resp": resp[:n], "offset": 0}
            state["current"]= cur
            state["k"]+= 1
            P= None

        if P is None:
            t= time.perf_counter()
            P= teacher_probs(teacher, cur["ids"], cur["resp"], device)
            times["teacher_scoring"]+= time.perf_counter() - t

        while cur["offset"] < len(cur["resp"]):
            ch= next_chunk(len(cur["resp"]) - cur["offset"], state["chunk"], total, size)
            off, k= cur["offset"], ch["k"]

            t= time.perf_counter()
            student.train()
            logq= student_log_probs(student, cur["ids"], cur["resp"], off, k, device, use_amp)
            kl= forward_kl_sum(P[off:off + k], logq)
            scaler.scale(kl / ch["update_size"]).backward()
            times["student_forward_backward"]+= time.perf_counter() - t

            state["kl_acc"]+= kl.item()
            cur["offset"]+= k

            if ch["closes"]:
                t= time.perf_counter()
                scaler.unscale_(optimizer)
                grad_norm= torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
                scale_before= scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                times["optimizer_step"]+= time.perf_counter() - t

                updates.append({"update": ch["update"], "kl_per_token": state["kl_acc"] / ch["update_size"],
                                "supervised_tokens": ch["update_size"], "grad_norm": float(grad_norm),
                                "loss_scale": scale_before, "skipped": bool(use_amp and scaler.get_scale() < scale_before),
                                "lr": optimizer.param_groups[0]["lr"], "scored_math": state["scored"]["math"],
                                "scored_short": state["scored"]["short"], "scored_total": state["chunk"]["done"],
                                "elapsed_seconds": time.perf_counter() - start})
                state["kl_acc"]= 0.0
                u= len(updates)

                if u == 5:
                    per= (time.perf_counter() - start) / state["chunk"]["done"]
                    print(f"speed {1000 * per:.1f} ms per scored token, about {per * (total - state['chunk']['done']) / 60:.1f} min left")

                if u % 20 == 0:
                    recent= updates[-20:]
                    print(f"{name} update {u}: kl/token={sum(r['kl_per_token'] for r in recent) / 20:.4f}, scored {state['chunk']['done']}/{total}, {time.perf_counter() - start:.0f}s")

                if u % ckpt_every == 0:
                    if cur["offset"] >= len(cur["resp"]):
                        state["current"]= None

                    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                    torch.save({"adapter": get_peft_model_state_dict(student), "optimizer": optimizer.state_dict(),
                                "scaler": scaler.state_dict(), "rng": rng_state(), "state": state,
                                "presentations": presentations, "updates": updates, "times": times}, ckpt_path + ".tmp")
                    os.replace(ckpt_path + ".tmp", ckpt_path)
                    state["current"]= cur

                    if args.stop_after_updates and u >= args.stop_after_updates:
                        #debug: simulate a dead session right after a checkpoint (resume test)
                        print(f"stopping after update {u}")
                        return

        state["current"]= None

    assert state["chunk"]["done"] == total and state["chunk"]["acc"] == 0, state["chunk"]
    assert state["scored"] == quotas
    wall= time.perf_counter() - start

    student.save_pretrained(adapter_dir)

    info= run_info(cfg, device, next(teacher.parameters()).dtype, model_key="target")
    info.update({"seed": args.seed, "method": args.method, "smoke": args.smoke, "lora": cfg["lora"]})
    pool_info= os.path.join(res_root, "task3", "teacher_pool_info.json")
    summary= {
        **info,
        "n_updates": len(updates),
        "skipped_updates": sum(r["skipped"] for r in updates),
        "scored_tokens": state["scored"],
        "n_presentations": len(presentations),
        "truncated_presentations": sum(r["truncated_by_quota"] for r in presentations),
        "eos_fraction_by_category": {c: sum(r["termination"] == "eos" for r in presentations if r["category"] == c) / max(1, sum(r["category"] == c for r in presentations)) for c in CATS},
        "mean_response_length_by_category": {c: sum(r["n_response"] for r in presentations if r["category"] == c) / max(1, sum(r["category"] == c for r in presentations)) for c in CATS},
        "times_seconds": times,
        "teacher_pool_generation_seconds": None,
        "wall_clock_seconds": wall,
        "student_params": sum(p.numel() for p in student.parameters()),
        "adapter_params": sum(p.numel() for p in params),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "final_loss_scale": scaler.get_scale() if use_amp else None,
        "adapter_dir": adapter_dir
    }

    if args.method == "off_policy" and os.path.exists(pool_info):
        #the shared offline teacher pool is part of the off-policy cost (p.12)
        with open(pool_info) as f:
            summary["teacher_pool_generation_seconds"]= json.load(f)["teacher_generation_seconds"]

    write_json(os.path.join(res_dir, "config.json"), {**info, "config": cfg, "quotas": quotas, "caps": caps})
    write_jsonl(os.path.join(res_dir, "presentations.jsonl.gz"), presentations)
    write_jsonl(os.path.join(res_dir, "train_log.jsonl.gz"), updates)
    write_json(os.path.join(res_dir, "summary.json"), summary)
    print(f"{name}: {len(updates)} updates ({summary['skipped_updates']} skipped), {len(presentations)} presentations, {wall:.0f}s -> {adapter_dir}")


if __name__ == "__main__":
    parser= argparse.ArgumentParser()

    parser.add_argument(
        "--method",
        type=str,
        choices=["off_policy", "on_policy"],
        required=True
    )
    parser.add_argument(
        "--seed",
        type=int,
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
    parser.add_argument(
        "--stop_after_updates",
        type=int,
        default=None
    )

    args= parser.parse_args()
    main(args)
