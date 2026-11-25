#task 2b data: supplied mqar generator (training stream), frozen development/test files (handout p.7)
#prep: verify the generator's invariants on the frozen files, build each training stream once, check no leakage
import os
import json
import time
import hashlib
import argparse
import importlib.util
from multiprocessing import Pool

import numpy as np
import torch

from common.io_utils import load_config, read_jsonl, write_json, resolve_data_root, default_output_dir, git_commit

MAX_K= 16
WORKER_GEN= None


def load_generator(data_root):
    #the supplied scripts/mqar_generator.py, imported by path (the data repo is not a package)
    path= os.path.join(data_root, "scripts", "mqar_generator.py")
    spec= importlib.util.spec_from_file_location("mqar_generator", path)
    gen= importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)

    return gen


def make_batch(examples):
    #tokens (B, T); labels = value id - 128 at query positions, -100 elsewhere; distances at query positions, -1 elsewhere
    tokens= torch.tensor([e["tokens"] for e in examples], dtype=torch.long)
    labels= torch.full(tokens.shape, -100, dtype=torch.long)
    dist= torch.full(tokens.shape, -1, dtype=torch.long)

    for b, e in enumerate(examples):
        q= torch.tensor(e["query_positions"])
        labels[b, q]= torch.tensor(e["answer_ids"]) - 128
        dist[b, q]= torch.tensor(e["distances"])

    return tokens, labels, dist


def init_worker(data_root):
    global WORKER_GEN
    WORKER_GEN= load_generator(data_root)


def make_train(args):
    k, regime, i= args

    return WORKER_GEN.make_example("train", k, regime, i)


def pack(examples):
    n= len(examples)
    T= len(examples[0]["tokens"])
    out= {
        "tokens": np.zeros((n, T), dtype=np.uint8),
        "query_positions": np.full((n, MAX_K), -1, dtype=np.int16),
        "answer_ids": np.full((n, MAX_K), -1, dtype=np.int16),
        "source_positions": np.full((n, MAX_K), -1, dtype=np.int16),
        "K": np.array([examples[0]["K"]]),
    }

    for i, e in enumerate(examples):
        k= len(e["query_positions"])
        out["tokens"][i]= e["tokens"]
        out["query_positions"][i, :k]= e["query_positions"]
        out["answer_ids"][i, :k]= e["answer_ids"]
        out["source_positions"][i, :k]= e["source_positions"]

    return out


def build_stream(gen, k, regime, n, workers=1, data_root=None):
    #training index i = update * batch_size + b (decision 2.4), same stream for every seed and method
    if workers == 1:
        examples= [gen.make_example("train", k, regime, i) for i in range(n)]
    else:
        with Pool(workers, initializer=init_worker, initargs=(data_root,)) as pool:
            examples= pool.map(make_train, [(k, regime, i) for i in range(n)], chunksize=2000)

    return pack(examples)


def stream_examples(stream, update, batch_size):
    #update is 0-based here; rows update*batch_size ... +batch_size-1
    k= int(stream["K"][0])
    out= []

    for i in range(update * batch_size, (update + 1) * batch_size):
        q= stream["query_positions"][i, :k].tolist()
        s= stream["source_positions"][i, :k].tolist()
        out.append({
            "tokens": stream["tokens"][i].tolist(),
            "query_positions": q,
            "answer_ids": stream["answer_ids"][i, :k].tolist(),
            "source_positions": s,
            "distances": [a - b for a, b in zip(q, s)]
        })

    return out


def stream_path(output_dir, k, regime, n):
    return os.path.join(output_dir, "task2", "mqar_streams", f"train_K{k}_{regime}_n{n}.npz")


def load_stream(path):
    with np.load(path) as f:
        return {key: f[key] for key in f.files}


def save_stream(path, stream):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp= path + ".tmp.npz"
    np.savez(tmp, **stream)
    os.replace(tmp, path)


def split_path(cfg, data_root, split):
    return os.path.join(data_root, cfg["data"][split])


def load_split(cfg, data_root, split):
    rows= read_jsonl(split_path(cfg, data_root, split))
    assert all(r["split"] == split for r in rows), split

    return rows


def row_hash(tokens):
    return hashlib.sha1(bytes(np.asarray(tokens, dtype=np.uint8))).hexdigest()


def stream_n(cfg, smoke):
    s= cfg["smoke"] if smoke else cfg

    return s["updates"] * s["batch_size"]


def file_sha256(path):
    h= hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)

    return h.hexdigest()


def main(args):
    cfg= load_config(args.config)
    data_root= resolve_data_root(cfg["data_repo"], args.data_root)
    gen= load_generator(data_root)
    start= time.perf_counter()

    #1. the frozen files pass the supplied invariant checks and equal the deterministic generator (never rewritten)
    held= {}

    for split in ["development", "test"]:
        rows= load_split(cfg, data_root, split)
        per_cell= cfg["data"]["test_per_cell" if split == "test" else "dev_per_cell"]
        assert len(rows) == 9 * per_cell, (split, len(rows))

        for r in rows:
            gen.validate(r)
            idx= int(r["id"].rsplit("-", 1)[1])

            if not args.smoke or idx < 2:
                assert gen.make_example(split, r["K"], r["regime"], idx) == r, r["id"]

        held[split]= {row_hash(r["tokens"]) for r in rows}
        print(f"{split}: {len(rows)} records validated")

    #2. one training stream per (K, regime) used by any run
    n= stream_n(cfg, args.smoke)
    out_dir= os.path.join(args.output_dir, "smoke") if args.smoke else args.output_dir
    streams= sorted({(r["K"], r["regime"]) for r in cfg["runs"]})
    check= {"streams": {}}

    for k, regime in streams:
        path= stream_path(out_dir, k, regime, n)

        if os.path.exists(path):
            stream= load_stream(path)
            print(f"{path} exists")
        else:
            t= time.perf_counter()
            stream= build_stream(gen, k, regime, n, workers=args.workers, data_root=data_root)
            save_stream(path, stream)
            print(f"built K={k} {regime}: {n} examples in {time.perf_counter() - t:.0f}s -> {path}")

        #3. no training sequence is a development or test sequence
        hashes= {row_hash(t) for t in stream["tokens"]}
        overlap= {s: len(hashes & held[s]) for s in held}
        assert all(v == 0 for v in overlap.values()), overlap

        check["streams"][f"K{k}_{regime}"]= {"n": n, "unique_sequences": len(hashes), "overlap_with": overlap, "sha256": file_sha256(path)}

    check.update({
        "git_commit": git_commit(),
        "smoke": args.smoke,
        "generator_sha256": file_sha256(os.path.join(data_root, "scripts", "mqar_generator.py")),
        "development_sha256": file_sha256(split_path(cfg, data_root, "development")),
        "test_sha256": file_sha256(split_path(cfg, data_root, "test")),
        "frozen_records_equal_generator": "all" if not args.smoke else "first 2 per cell",
        "wall_clock_seconds": time.perf_counter() - start
    })

    #committed copy lives in splits/task2; on kaggle write outside the repo so the code tree stays clean
    if args.check_path:
        path= args.check_path
    elif args.smoke:
        path= os.path.join(out_dir, "task2", "mqar_data_check.json")
    else:
        path= os.path.join("splits", "task2", "mqar_data_check.json")
    write_json(path, check)
    print(f"data check ok -> {path}")


if __name__ == "__main__":
    parser= argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/task2/mqar.yaml"
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
        "--workers",
        type=int,
        default=os.cpu_count() or 1
    )
    parser.add_argument(
        "--check_path",
        type=str,
        default=None
    )
    parser.add_argument(
        "--smoke",
        action="store_true"
    )

    args= parser.parse_args()
    main(args)
