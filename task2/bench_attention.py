#task 2a benchmark (handout p.7, pp.8-9): dense vs tiled vs sdpa on identical random q, k, v, B=2 H=4 d=64
#per (dtype, L): forward and forward+backward time, peak allocated/reserved memory, output and gradient error vs a
#dense fp32 reference (tiled fp32 where dense fp32 does not fit; the reference used is recorded), forced sdpa backends
import os
import time
import contextlib
import argparse

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend

from common.seed import set_seed
from common.timing import time_fn, sync
from common.io_utils import load_config, write_json, default_output_dir, run_info
from task2.attention_ops import dense_attention, tiled_attention, sdpa_attention, tile_count

#aten op names the dispatcher calls for each backend (read from the profiler, decision 2.2)
BACKEND_OPS= {
    "aten::_scaled_dot_product_flash_attention": "FLASH_ATTENTION",
    "aten::_scaled_dot_product_efficient_attention": "EFFICIENT_ATTENTION",
    "aten::_scaled_dot_product_cudnn_attention": "CUDNN_ATTENTION",
    "aten::_scaled_dot_product_attention_math": "MATH",
    "aten::_scaled_dot_product_flash_attention_for_cpu": "FLASH_ATTENTION_CPU"
}


def make_inputs(cfg, T, dtype, device, seed):
    g= torch.Generator(device="cpu").manual_seed(seed)
    shape= (cfg["batch_size"], cfg["num_heads"], T, cfg["head_dim"])
    q, k, v, do= (torch.randn(shape, generator=g) for _ in range(4))

    return [x.to(device=device, dtype=dtype) for x in (q, k, v, do)]


def dispatched_backend(q, k, v):
    #which sdpa implementation actually ran (default dispatch, no forcing)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        with torch.no_grad():
            sdpa_attention(q, k, v)

    names= {e.name for e in prof.events()}
    hit= [BACKEND_OPS[n] for n in BACKEND_OPS if n in names]

    return hit[0] if hit else "unknown", sorted(n for n in names if "scaled_dot_product" in n)


def impl_fn(name, cfg, block):
    if name == "dense":
        return dense_attention

    if name == "tiled":
        return lambda q, k, v: tiled_attention(q, k, v, block, block)

    return sdpa_attention


def backend_ctx(name):
    #sdpa_<BACKEND> forces one backend; plain sdpa uses the default dispatch
    if name.startswith("sdpa_"):
        return sdpa_kernel([getattr(SDPBackend, name[len("sdpa_"):])])

    return contextlib.nullcontext()


def peak(device, fn):
    #peak allocated / reserved during one call, also relative to what was allocated before it
    if device.type != "cuda":
        fn()
        return {"peak_allocated_bytes": None, "peak_reserved_bytes": None, "peak_above_start_bytes": None}

    sync(device)
    torch.cuda.reset_peak_memory_stats(device)
    start= torch.cuda.memory_allocated(device)
    fn()
    sync(device)

    return {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "peak_above_start_bytes": torch.cuda.max_memory_allocated(device) - start
    }


def errors(out, grads, ref_out, ref_grads):
    def err(a, b):
        d= (a.double() - b.double()).abs()

        return {"max_abs": d.max().item(), "mean_abs": d.mean().item(), "max_rel_to_ref_max": (d.max() / b.double().abs().max()).item()}

    e= {"output": err(out, ref_out)}

    if grads is not None:
        for n, g, r in zip(["dq", "dk", "dv"], grads, ref_grads):
            e[n]= err(g, r)

    return e


def forward_backward(fn, q, k, v, do):
    a, b, c= (x.detach().clone().requires_grad_(True) for x in (q, k, v))
    out= fn(a, b, c)
    out.backward(do)

    return out.detach(), [a.grad, b.grad, c.grad]


def is_oom(e):
    return isinstance(e, torch.OutOfMemoryError) or "out of memory" in str(e).lower()


def reference(q, k, v, do, cfg):
    #dense fp32 if it fits, else the exact tiled algorithm in fp32 (tested equal to dense)
    q32, k32, v32, do32= (x.float() for x in (q, k, v, do))

    for name, fn in [("dense_fp32", dense_attention), ("tiled_fp32", lambda a, b, c: tiled_attention(a, b, c, cfg["tiled"]["default_block_q"], cfg["tiled"]["default_block_k"]))]:
        try:
            out, grads= forward_backward(fn, q32, k32, v32, do32)
            return name, out, grads
        except RuntimeError as e:
            if not is_oom(e):
                raise
            torch.cuda.empty_cache()

    raise RuntimeError("no reference fits")


def failure(e, device):
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {"status": "oom" if is_oom(e) else "unsupported", "message": str(e).split("\n")[0][:300]}


def bench_case(name, cfg, q, k, v, do, ref, device, warmup, reps):
    #forward-only and forward+backward measured separately: a backward oom keeps the forward result
    fn= impl_fn(name, cfg, cfg["tiled"]["default_block_q"])
    row= {"impl": name}

    def fwd():
        with torch.no_grad():
            return fn(q, k, v)

    with backend_ctx(name):
        try:
            out= fwd()
            row["forward"]= time_fn(lambda _: fwd(), device, warmup, reps)
            row["forward_memory"]= peak(device, fwd)
            row["forward_errors_vs_reference"]= errors(out, None, ref[1], None)
            row["finite"]= bool(torch.isfinite(out).all())
            row["status"]= "ok"
            del out
        except RuntimeError as e:
            row.update(failure(e, device))
            return row

        try:
            out, grads= forward_backward(fn, q, k, v, do)
            row["errors_vs_reference"]= errors(out, grads, ref[1], ref[2])
            del out, grads
            row["forward_backward"]= time_fn(lambda _: forward_backward(fn, q, k, v, do), device, warmup, reps)
            row["forward_backward_memory"]= peak(device, lambda: forward_backward(fn, q, k, v, do))
            row["backward_status"]= "ok"
        except RuntimeError as e:
            f= failure(e, device)
            row["backward_status"]= f["status"]
            row["backward_message"]= f["message"]

    return row


def edge_cases(cfg, device, dtypes):
    #extreme scores, an all-masked row, and future perturbation, at a modest length
    T= 1024 if device.type == "cuda" else 64
    out= {"T": T, "extreme_scores": {}, "all_masked_row": {}, "causality": {}}

    for dt in dtypes:
        dtype= getattr(torch, dt)
        q, k, v, _= make_inputs(cfg, T, dtype, device, 11)
        ref= dense_attention((q * 300).double(), k.double(), v.double())
        res= {}

        for name, fn in [("dense", dense_attention), ("tiled", lambda a, b, c: tiled_attention(a, b, c, 128, 128)), ("sdpa", sdpa_attention)]:
            with torch.no_grad():
                o= fn(q * 300, k, v)

            res[name]= {"finite": bool(torch.isfinite(o).all()), "max_abs_err_vs_fp64": (o.double() - ref).abs().max().item()}

            #causality: change keys/values after position T/2, earlier outputs must not move
            k2, v2= k.clone(), v.clone()
            k2[:, :, T // 2:]+= 3.0
            v2[:, :, T // 2:]-= 2.0

            with torch.no_grad():
                a= fn(q, k, v)
                b= fn(q, k2, v2)

            out["causality"].setdefault(dt, {})[name]= {"max_change_before": (a[:, :, :T // 2] - b[:, :, :T // 2]).abs().max().item(),
                                                       "max_change_after": (a[:, :, T // 2:] - b[:, :, T // 2:]).abs().max().item()}

        out["extreme_scores"][dt]= {"q_scale": 300, **res}

        #a row with no visible key: our dense returns 0; record what sdpa does with the same boolean mask
        mask= torch.ones(T, T, dtype=torch.bool, device=device).tril()
        mask[5]= False

        with torch.no_grad():
            d= dense_attention(q, k, v, mask=mask)
            s= torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        out["all_masked_row"][dt]= {
            "dense_row_is_zero": bool((d[:, :, 5] == 0).all()),
            "sdpa_row_finite": bool(torch.isfinite(s[:, :, 5]).all()),
            "sdpa_row_value_example": s[0, 0, 5, :4].float().tolist(),
            "other_rows_max_abs_diff": (d[:, :, 6:] - s[:, :, 6:]).abs().max().item()
        }

    return out


def main(args):
    cfg= load_config(args.config)
    set_seed(cfg["seed"])
    device= torch.device(args.device)
    print("Using device:", device)

    lengths= [64, 128] if args.smoke else cfg["seq_lengths"]
    dtypes= ["float32"] if args.smoke else cfg["dtypes"]
    reps= 2 if args.smoke else cfg["timing_reps"]
    warmup= 1 if args.smoke else cfg["warmup_reps"]
    tiles= [16, 32] if args.smoke else cfg["tiled"]["verify_tile_sizes"]
    out_dir= os.path.join(args.output_dir, "smoke", "results", "task2", "attention") if args.smoke else os.path.join(args.results_dir, "task2", "attention")
    cases= ["dense", "tiled", "sdpa"] + [f"sdpa_{b}" for b in cfg["sdpa_backends"]]

    if device.type == "cpu":
        cases= ["dense", "tiled", "sdpa", "sdpa_MATH"]

    info= run_info(cfg, device, None)
    info["dtype"]= "per file"

    path= os.path.join(out_dir, "edge_cases.json")

    if not os.path.exists(path):
        write_json(path, {**info, **edge_cases(cfg, device, dtypes)})
        print(f"edge cases -> {path}")

    for dt in dtypes:
        dtype= getattr(torch, dt)

        for T in lengths:
            path= os.path.join(out_dir, f"{dt}_L{T}.json")

            if os.path.exists(path):
                print(f"{path} exists, skipping")
                continue

            start= time.perf_counter()
            q, k, v, do= make_inputs(cfg, T, dtype, device, cfg["seed"] + T)
            ref= reference(q, k, v, do, cfg)
            backend, ops= dispatched_backend(q, k, v)

            rows= [bench_case(name, cfg, q, k, v, do, ref, device, warmup, reps) for name in cases]

            #tiled output across tile sizes (forward only, vs the same reference)
            verify= {}

            for b in tiles:
                with torch.no_grad():
                    o= tiled_attention(q, k, v, b, b)

                verify[str(b)]= {"max_abs_err": (o.double() - ref[1].double()).abs().max().item(), "tiles_computed": tile_count(T, b, b)}

            result= {
                **info,
                "dtype": dt,
                "T": T,
                "shape": [cfg["batch_size"], cfg["num_heads"], T, cfg["head_dim"]],
                "reference": ref[0],
                "sdpa_default_backend": backend,
                "sdpa_default_ops": ops,
                "tiled_block": [cfg["tiled"]["default_block_q"], cfg["tiled"]["default_block_k"]],
                "tiled_note": "python reference; backward is autograd and stores every tile (not a fused flashattention kernel)",
                "tile_size_verification": verify,
                "dense_score_matrix_bytes": cfg["batch_size"] * cfg["num_heads"] * T * T * 4,
                "rows": rows,
                "warmup_reps": warmup,
                "timing_reps": reps,
                "wall_clock_seconds": time.perf_counter() - start
            }
            write_json(path, result)

            line= ", ".join(f"{r['impl']}={r['forward']['median']:.2f}ms" if r["status"] == "ok" else f"{r['impl']}={r['status']}" for r in rows)
            print(f"{dt} L={T} (ref {ref[0]}, sdpa default {backend}): {line}")

            del q, k, v, do, ref

            if device.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    parser= argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/task2/attention.yaml"
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
