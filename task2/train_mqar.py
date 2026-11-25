#task 2b training (handout p.7): AdamW 8e-4 wd 0.01, batch 32, 4,500 updates, 200 warmup then linear to 15%, clip 1.0
#loss = cross-entropy at query positions only (not next-token lm). dev evals at fixed updates; test once, final checkpoint
import os
import time
import random
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from common.seed import set_seed
from common.io_utils import load_config, write_json, write_jsonl, resolve_data_root, default_output_dir, run_info
from task2.mqar_model import MQARModel, attended_pairs, state_bytes
from task2.mqar_data import load_generator, make_batch, build_stream, stream_examples, stream_path, load_stream, save_stream, stream_n, load_split
from task2.eval_mqar import evaluate_split


def lr_at(update, cfg):
    #update is 1-based; linear warmup to peak, then linear decay to final_lr_fraction * peak at the last update
    peak= cfg["optimizer"]["peak_lr"]
    w= cfg["warmup_updates"]
    total= cfg["updates"]

    if update <= w:
        return peak * update / w

    return peak * (1.0 - (1.0 - cfg["final_lr_fraction"]) * (update - w) / (total - w))


def query_loss(logits, labels):
    #mean over supervised query positions; every other position is -100
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)


def mqar_train(model, tokens, labels, optimizer, device, lr, grad_clip):
    model.train()

    for group in optimizer.param_groups:
        group["lr"]= lr

    tokens= tokens.to(device)
    labels= labels.to(device)

    optimizer.zero_grad()

    logits= model(tokens)
    loss= query_loss(logits, labels)
    loss.backward()

    grad_norm= torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    with torch.no_grad():
        mask= labels != -100
        acc= (logits.argmax(dim=-1)[mask] == labels[mask]).float().mean()

    return (loss.item(), acc.item(), grad_norm.item())


def smoke_cfg(cfg):
    #a few updates and examples; same model and optimizer
    c= dict(cfg)
    c.update({k: cfg["smoke"][k] for k in ["updates", "batch_size", "warmup_updates", "eval_at"]})

    return c


def subset(rows, per_cell):
    out= {}

    for r in rows:
        cell= out.setdefault((r["K"], r["regime"]), [])

        if len(cell) < per_cell:
            cell.append(r)

    return [r for cell in out.values() for r in cell]


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    }


def set_rng_state(s):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])

    if s["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def get_stream(cfg, args, run, data_root, out_dir):
    n= stream_n(cfg["full"], args.smoke)
    path= stream_path(out_dir, run["K"], run["regime"], n)

    if not os.path.exists(path):
        #normally built by task2.mqar_data first (with its frozen-data and leakage checks)
        print(f"{path} missing, building it")
        save_stream(path, build_stream(load_generator(data_root), run["K"], run["regime"], n))

    stream= load_stream(path)
    assert len(stream["tokens"]) >= cfg["updates"] * cfg["batch_size"]

    return stream


def main(args):
    full= load_config(args.config)
    cfg= smoke_cfg(full) if args.smoke else dict(full)
    cfg["full"]= full

    runs= {r["name"]: r for r in full["runs"]}
    assert args.run_name in runs, f"unknown run {args.run_name}"
    run= runs[args.run_name]
    assert args.seed in full["seeds"], "seeds are fixed by the handout"
    #the last update is always an eval/checkpoint point, so the final state and timing are saved
    assert cfg["eval_at"][-1] == cfg["updates"]

    set_seed(args.seed)
    device= torch.device(args.device)
    print("Using device:", device)

    out_dir= os.path.join(args.output_dir, "smoke") if args.smoke else args.output_dir
    results_root= os.path.join(out_dir, "results") if args.smoke else args.results_dir
    name= f"{run['name']}_s{args.seed}"
    res_dir= os.path.join(results_root, "task2", "mqar", name)
    ckpt_dir= os.path.join(out_dir, "task2", "mqar", name)
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    #test-time W=64 control on the dense models (decision 2.5)
    controls= ["window"] if run["name"] in full["test_time_window_control"] else []
    needed= [os.path.join(res_dir, "test_metrics.json")] + [os.path.join(res_dir, f"test_{a}_mask_metrics.json") for a in controls]

    if all(os.path.exists(p) for p in needed):
        print(f"{name}: done, skipping")
        return

    data_root= resolve_data_root(full["data_repo"], args.data_root)
    stream= get_stream(cfg, args, run, data_root, out_dir)
    dev= load_split(full, data_root, "development")

    if args.smoke:
        dev= subset(dev, full["smoke"]["eval_examples_per_cell"])

    model= MQARModel(full["model"], full["attention"], full["seq_len"], full["num_classes"], attention=run["attention"]).to(device)
    optimizer= torch.optim.AdamW(model.parameters(), lr=lr_at(1, cfg), weight_decay=full["optimizer"]["weight_decay"])
    n_params= sum(p.numel() for p in model.parameters())

    #resume from the last checkpoint: model, optimizer, rng, update count (= data position), logs
    last= os.path.join(ckpt_dir, "last.pt")
    start_update= 0
    train_log= []
    history= []
    train_seconds= 0.0

    if os.path.exists(last):
        ckpt= torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        set_rng_state(ckpt["rng"])
        start_update= ckpt["update"]
        train_log= ckpt["train_log"]
        history= ckpt["history"]
        train_seconds= ckpt["train_seconds"]
        print(f"resumed {name} at update {start_update}")

    info= run_info(full, device, torch.float32)
    info.update({
        "seed": args.seed,
        "run": run,
        "smoke": args.smoke,
        "n_params": n_params,
        "train_stream": f"make_example('train', {run['K']}, '{run['regime']}', update * {cfg['batch_size']} + b)",
        "theoretical_pairs_T256": attended_pairs(run["attention"], full["seq_len"], full["attention"], full["model"]["num_heads"]),
        "state_bytes_T256_fp32": state_bytes(run["attention"], full["seq_len"], full["model"], full["attention"], 4)
    })
    write_json(os.path.join(res_dir, "config.json"), {**info, "config": {k: v for k, v in cfg.items() if k != "full"}})
    print(f"{name}: {run['attention']} K={run['K']} {run['regime']}, {n_params} params")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t0= time.perf_counter()

    for update in range(start_update + 1, cfg["updates"] + 1):
        tokens, labels, _= make_batch(stream_examples(stream, update - 1, cfg["batch_size"]))
        lr= lr_at(update, cfg)

        loss, acc, grad_norm= mqar_train(
            model=model,
            tokens=tokens,
            labels=labels,
            optimizer=optimizer,
            device=device,
            lr=lr,
            grad_clip=full["grad_clip"]
        )

        train_log.append({"update": update, "loss": loss, "query_acc": acc, "lr": lr, "grad_norm": grad_norm})

        if update == start_update + 20:
            per= (time.perf_counter() - t0) / 20
            print(f"speed {per * 1000:.1f} ms/update, about {per * (cfg['updates'] - update) / 60:.1f} min of training left")

        if update % 500 == 0:
            recent= train_log[-100:]
            print(
                f"update {update}: "
                f"loss={sum(r['loss'] for r in recent) / len(recent):.4f}, "
                f"query_acc={sum(r['query_acc'] for r in recent) / len(recent):.4f}, "
                f"lr={lr:.2e}"
            )

        if update in cfg["eval_at"]:
            train_seconds+= time.perf_counter() - t0
            summary, _= evaluate_split(model, dev, device, run["attention"], full)
            recent= train_log[-100:]
            history.append({
                "update": update,
                "train_loss_last100": sum(r["loss"] for r in recent) / len(recent),
                "train_query_acc_last100": sum(r["query_acc"] for r in recent) / len(recent),
                "dev": summary["cells"],
                "dev_bins": summary["bins"],
                "train_seconds": train_seconds
            })
            own= summary["cells"][f"K{run['K']}_{run['regime']}"]
            print(f"dev at {update}: own cell K{run['K']}_{run['regime']} acc={own['accuracy']:.4f} loss={own['mean_loss']:.4f}")

            write_json(os.path.join(res_dir, f"{name}_history.json"), history)
            write_jsonl(os.path.join(res_dir, "train_log.jsonl.gz"), train_log)
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng": rng_state(),
                "update": update,
                "train_log": train_log,
                "history": history,
                "train_seconds": train_seconds,
                "run": run,
                "seed": args.seed
            }, last + ".tmp")
            os.replace(last + ".tmp", last)
            t0= time.perf_counter()

    #final checkpoint is the one tested (no best-checkpoint choice)
    final= os.path.join(ckpt_dir, "final.pt")
    torch.save({"model": model.state_dict(), "update": cfg["updates"], "run": run, "seed": args.seed}, final)

    peak= torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    test= load_split(full, data_root, "test")

    if args.smoke:
        test= subset(test, full["smoke"]["eval_examples_per_cell"])

    for attention, tag in [(run["attention"], "test")] + [(a, f"test_{a}_mask") for a in controls]:
        summary, preds= evaluate_split(model, test, device, attention, full)
        write_jsonl(os.path.join(res_dir, f"{tag}_predictions.jsonl.gz"), preds)
        write_json(os.path.join(res_dir, f"{tag}_metrics.json"), {
            **info,
            **summary,
            "trained_attention": run["attention"],
            "final_update": cfg["updates"],
            "train_wall_clock_seconds": train_seconds,
            "ms_per_update": 1000 * train_seconds / cfg["updates"],
            "peak_allocated_bytes_train": peak
        })
        print(f"{tag} ({attention}): " + ", ".join(f"{c}={v['accuracy']:.3f}" for c, v in summary["cells"].items()))


if __name__ == "__main__":
    parser= argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/task2/mqar.yaml"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=True
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=6304
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
