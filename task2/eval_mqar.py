#task 2b evaluation: accuracy over supervised query positions by (K, regime) and (K, distance bin) (handout pp.7-8)
#uses the saved source position of every query, never the query position alone
import os
import time
import argparse

import torch
import torch.nn.functional as F

from common.io_utils import load_config, write_json, write_jsonl, resolve_data_root, default_output_dir, run_info
from task2.mqar_model import MQARModel
from task2.mqar_data import make_batch, load_split


def evaluate_model(model, examples, device, attention, batch_size):
    #one row per supervised query: prediction, correctness and loss
    model.eval()
    rows= []

    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            exs= examples[i:i + batch_size]
            tokens, labels, _= make_batch(exs)
            labels= labels.to(device)
            logits= model(tokens.to(device), attention=attention).float()

            #boolean selection is row-major, query positions ascending (the generator stores them sorted)
            mask= labels != -100
            y= labels[mask]
            lq= logits[mask]
            preds= lq.argmax(dim=-1).tolist()
            losses= F.cross_entropy(lq, y, reduction="none").tolist()
            y= y.tolist()
            j= 0

            for e in exs:
                assert e["query_positions"] == sorted(e["query_positions"])

                for q, s, d in zip(e["query_positions"], e["source_positions"], e["distances"]):
                    rows.append({
                        "id": e.get("id"),
                        "K": e.get("K", len(e["query_positions"])),
                        "regime": e.get("regime"),
                        "query_position": q,
                        "source_position": s,
                        "distance": d,
                        "answer": y[j],
                        "pred": preds[j],
                        "correct": int(preds[j] == y[j]),
                        "loss": losses[j]
                    })
                    j+= 1

            assert j == len(y)

    return rows


def bin_name(d, bins):
    for lo, hi in bins:
        if lo <= d < hi:
            return f"[{lo},{hi})"

    raise ValueError(d)


def group_stats(rows, k):
    n= len(rows)

    return {
        "n_queries": n,
        "accuracy": sum(r["correct"] for r in rows) / n,
        "mean_loss": sum(r["loss"] for r in rows) / n,
        "chance_label_space": 1 / 64,
        "chance_1_over_k": 1 / k
    }


def summarize_queries(rows, bins):
    #micro-averaged over queries; cells = (K, regime); bins = (K, distance bin) pooled over regimes
    cells= {}
    by_bin= {}

    for r in rows:
        cells.setdefault((r["K"], r["regime"]), []).append(r)
        by_bin.setdefault((r["K"], bin_name(r["distance"], bins)), []).append(r)

    return {
        "cells": {f"K{k}_{reg}": group_stats(v, k) for (k, reg), v in sorted(cells.items())},
        "bins": {f"K{k}_{b}": group_stats(v, k) for (k, b), v in sorted(by_bin.items())}
    }


def prediction_rows(rows):
    #compact per-example record (details are recoverable from the frozen file by id)
    out= {}

    for r in rows:
        e= out.setdefault(r["id"], {"id": r["id"], "query_positions": [], "pred": [], "correct": [], "loss": []})
        e["query_positions"].append(r["query_position"])
        e["pred"].append(r["pred"])
        e["correct"].append(r["correct"])
        e["loss"].append(round(r["loss"], 5))

    return list(out.values())


def evaluate_split(model, examples, device, attention, cfg):
    start= time.perf_counter()
    rows= evaluate_model(model, examples, device, attention, cfg["eval_batch_size"])
    summary= summarize_queries(rows, cfg["distance_bins"])
    summary["attention_at_eval"]= attention
    summary["n_examples"]= len(examples)
    summary["eval_seconds"]= time.perf_counter() - start

    return summary, prediction_rows(rows)


def main(args):
    #re-evaluate a saved final checkpoint on the frozen test set, optionally under a different (test-time) mask
    cfg= load_config(args.config)
    device= torch.device(args.device)
    print("Using device:", device)

    ckpt= torch.load(args.checkpoint, map_location=device)
    run= ckpt["run"]
    assert ckpt["update"] == cfg["updates"], "only the final checkpoint is evaluated on test"

    model= MQARModel(cfg["model"], cfg["attention"], cfg["seq_len"], cfg["num_classes"], attention=run["attention"]).to(device)
    model.load_state_dict(ckpt["model"])

    data_root= resolve_data_root(cfg["data_repo"], args.data_root)
    test= load_split(cfg, data_root, "test")
    attention= args.attention or run["attention"]

    summary, preds= evaluate_split(model, test, device, attention, cfg)
    info= run_info(cfg, device, torch.float32)
    info.update({"seed": ckpt["seed"], "run": run, "checkpoint": args.checkpoint})

    write_json(os.path.join(args.out_dir, f"test_{attention}_metrics.json"), {**info, **summary})
    write_jsonl(os.path.join(args.out_dir, f"test_{attention}_predictions.jsonl.gz"), preds)
    print(f"test under {attention}: " + ", ".join(f"{c}={v['accuracy']:.3f}" for c, v in summary["cells"].items()))


if __name__ == "__main__":
    parser= argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/task2/mqar.yaml"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True
    )
    parser.add_argument(
        "--attention",
        type=str,
        choices=["dense", "window", "local_dilated", "linear"],
        default=None
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
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu"
    )

    args= parser.parse_args()
    main(args)
