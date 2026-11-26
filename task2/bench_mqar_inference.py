#task 2 part B forward-only sweep (handout p.8): every mqar attention operator at T in {256, 1024, 4096, 8192} with
#fixed random embeddings and matched B, H, d (an untrained model, seed 6304, fp32 as in training): isolates kernel cost
#from learnability. records latency, peak allocated/reserved memory, materialized dense masks and the theoretical
#pair counts / state bytes. linear attention is timed in both forms: the parallel form used in training (materializes
#a T x T score tensor) and the constant-state recurrence (python loop over T). ooms are recorded, not skipped silently
import os
import argparse
import contextlib

import torch

import task2.mqar_model as mm
from common.seed import set_seed
from common.timing import time_fn
from common.io_utils import load_config, write_json, default_output_dir, run_info
from task2.mqar_model import MQARModel, attended_pairs, state_bytes, linear_attention_recurrent

KINDS= [("dense", "dense"), ("window", "window"), ("local_dilated", "local_dilated"), ("linear", "linear_parallel"), ("linear", "linear_recurrent")]


@contextlib.contextmanager
def recurrent_linear():
    #route the model's linear operator through the exact recurrence for this measurement only
    parallel= mm.linear_attention
    mm.linear_attention= lambda q, k, v, eps: linear_attention_recurrent(q, k, v, eps)[0]

    try:
        yield
    finally:
        mm.linear_attention= parallel


def measure(model, attention, impl, T, B, d, device, warmup, reps):
    gen= torch.Generator(device="cpu").manual_seed(6304 + T)
    h= torch.randn(B, T, d, generator=gen).to(device)
    ctx= recurrent_linear() if impl == "linear_recurrent" else contextlib.nullcontext()

    with ctx, torch.no_grad():
        mask= model.mask_for(attention, T, device)
        times= time_fn(lambda _: model.forward_embeds(h, attention), device, warmup, reps)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            base= torch.cuda.memory_allocated(device)
            model.forward_embeds(h, attention)
            torch.cuda.synchronize(device)
            peak= torch.cuda.max_memory_allocated(device)
            reserved= torch.cuda.max_memory_reserved(device)
        else:
            base= peak= reserved= None

    return {"latency_ms": times, "peak_allocated_bytes": peak, "allocated_before_call_bytes": base, "peak_reserved_bytes": reserved,
            "dense_mask_materialized_bytes": None if mask is None else mask.numel() * mask.element_size(),
            "linear_score_tensor_bytes": B * model.num_heads * T * T * 4 if impl == "linear_parallel" else None}


def main(args):
    cfg= load_config(args.config)
    set_seed(cfg["seed"])
    device= torch.device(args.device)
    print("Using device:", device)

    sw= cfg["forward_sweep"]
    lengths= [64, 256] if args.smoke else sw["seq_lengths"]
    warmup, reps= (1, 2) if args.smoke else (sw["warmup_reps"], sw["timing_reps"])
    B= sw["batch_size"]
    mcfg, acfg= cfg["model"], cfg["attention"]
    d= mcfg["hidden_dim"]
    out_dir= os.path.join(args.output_dir, "smoke") if args.smoke else args.output_dir
    res_root= os.path.join(out_dir, "results") if args.smoke else args.results_dir
    res_dir= os.path.join(res_root, "task2", "mqar_inference")

    for T in lengths:
        path= os.path.join(res_dir, f"T{T}.json")

        if os.path.exists(path):
            print(f"T={T} done, skipping")
            continue

        torch.manual_seed(cfg["seed"])
        model= MQARModel(mcfg, acfg, T, cfg["num_classes"]).to(device).eval()
        rows= {}

        for attention, impl in KINDS:
            r= {"attention": attention, "impl": impl, "T": T, "batch_size": B,
                "pairs": attended_pairs(attention, T, acfg, mcfg["num_heads"]),
                "theoretical_state_fp32": state_bytes(attention, T, mcfg, acfg, 4)}

            try:
                r.update(measure(model, attention, impl, T, B, d, device, warmup, reps))
                r["status"]= "ok"
                print(f"T={T} {impl}: {r['latency_ms']['median']:.2f} ms, peak {r['peak_allocated_bytes']}")
            except torch.OutOfMemoryError as e:
                r["status"]= "oom"
                r["error"]= str(e).splitlines()[0]
                print(f"T={T} {impl}: OOM")

            if device.type == "cuda":
                torch.cuda.empty_cache()

            rows[impl]= r

        info= run_info(cfg, device, torch.float32)
        info.update({"T": T, "batch_size": B, "smoke": args.smoke, "warmup": warmup, "reps": reps,
                     "inputs": "fixed random embeddings (seed 6304 + T), untrained model (seed 6304), fp32",
                     "labels": "window/local_dilated = sdpa with a dense boolean mask (correctness implementation, not a sparse kernel)",
                     "results": rows})
        write_json(path, info)


if __name__ == "__main__":
    parser= argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/task2/mqar.yaml"
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
