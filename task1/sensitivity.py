#decision 1.7: one gradient pass over the 128 calibration sequences gives
#  p3: first-order taylor |g w| (g = mean gradient), averaged over the weights wanda would prune
#  q3: fisher-weighted quantization error sum_i sum_w g_i^2 dw_b^2 (hawq-v2 style), per allowed bit width
import os
import time
import argparse

#less fragmentation in the long fp16 backward loop; read when cuda first initializes
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F

from common.seed import set_seed
from common.io_utils import load_config, default_output_dir, run_info, write_json, resolve_data_root
from task1.load import load_model, load_sequences, eligible_linears, unique_param_count
from task1.prune import wanda_scores, row_prune_mask
from task1.quantize import fake_quantize


def taylor_scores(w, g):
    #first-order estimate of the loss change from zeroing each weight: |dL/dw * w|
    return (g.float() * w.float()).abs()


def pruned_set_sensitivity(w, g, x_norm, sparsity):
    #mean taylor score over the weights wanda prunes in this projection at this sparsity (fixed per-row count)
    n_prune= int(round(sparsity * w.shape[1]))
    keep= row_prune_mask(wanda_scores(w, x_norm), n_prune)

    return taylor_scores(w, g)[~keep].mean().item()


def fisher_quant_error(g, w, bits, group_size, scale_dtype):
    #second-order (diagonal fisher) estimate of the loss change from quantizing w to b bits, one sample's gradient
    dw= fake_quantize(w, bits, group_size, scale_dtype) - w.float()

    return (g.float().pow(2) * dw.pow(2)).sum().item()


#fraction of gpu memory the estimate may use: the oom'd first run lost ~1.9 GiB of 14.56 to fragmentation
GPU_FRACTION= 0.75


def gradient_pass_gpu_bytes(n_unique, n_eligible, seq_len, vocab, hidden, n_layers):
    #resident: fp16 params + fp16 grads of eligible weights (the fp32 gradient sum lives in cpu ram);
    #transient: fp32 logits + their grad + softmax, and per-layer saved activations (~24 hidden-sized fp16 tensors)
    resident= 2 * n_unique + 2 * n_eligible
    transient= 3 * 4 * seq_len * vocab + n_layers * 24 * 2 * seq_len * hidden

    return resident + transient


def gradient_pass(model, sequences, device, bits, group_size, scale_dtype, loss_scale=1.0, sum_device="cpu"):
    #per sequence (batch 1): mean next-token loss, backward to the eligible weights only
    #the fp32 gradient sum (4 bytes per eligible weight, 5.2 GB for qwen2.5-1.5b) is kept on sum_device, not the gpu
    linears= eligible_linears(model)
    flags= [p.requires_grad for p in model.parameters()]

    for p in model.parameters():
        p.requires_grad_(False)

    units= {}

    for name, lin in linears:
        lin.weight.requires_grad_(True)
        units[name]= {
            "g_sum": torch.zeros(lin.weight.shape, dtype=torch.float32, device=sum_device),
            "sumsq": torch.zeros(lin.weight.shape[1], dtype=torch.float32, device=lin.weight.device),
            "n": 0,
            "fisher": {b: 0.0 for b in bits}
        }

    def stats_hook(name):
        def hook(module, inp, out):
            x= inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            units[name]["sumsq"]+= x.pow(2).sum(dim=0)
            units[name]["n"]+= x.shape[0]
        return hook

    handles= [lin.register_forward_hook(stats_hook(name)) for name, lin in linears]
    scale= loss_scale
    rescaled= 0

    try:
        for i, s in enumerate(sequences):
            ids= torch.tensor([s["token_ids"]], device=device)

            while True:
                snapshot= {name: units[name]["n"] for name, _ in linears}
                saved= {name: units[name]["sumsq"].clone() for name, _ in linears}

                with torch.enable_grad():
                    logits= model(ids, use_cache=False).logits[0, :-1]
                    loss= F.cross_entropy(logits.float(), ids[0, 1:])
                    (loss * scale).backward()

                grads= [lin.weight.grad for _, lin in linears]

                if all(torch.isfinite(g).all() for g in grads):
                    break

                #fp16 overflow: undo this sequence's stats, halve the loss scale and redo it
                for name, lin in linears:
                    lin.weight.grad= None
                    units[name]["n"]= snapshot[name]
                    units[name]["sumsq"]= saved[name]

                scale/= 2
                rescaled+= 1
                assert scale > 1e-8, "gradients are not finite at any loss scale"

            with torch.no_grad():
                for name, lin in linears:
                    g= lin.weight.grad.float() / scale

                    for b in bits:
                        units[name]["fisher"][b]+= fisher_quant_error(g, lin.weight, b, group_size, scale_dtype)

                    #copy the fp16 grad (half the bytes of g) and accumulate in fp32 where the sum lives
                    units[name]["g_sum"].add_(lin.weight.grad.to(sum_device).float(), alpha=1.0 / scale)

                    del g
                    lin.weight.grad= None

            if (i + 1) % 16 == 0:
                print(f"gradient pass {i + 1}/{len(sequences)}")
    finally:
        for h in handles:
            h.remove()

        for p, f in zip(model.parameters(), flags):
            p.requires_grad_(f)
            p.grad= None

    return {"units": units, "n_sequences": len(sequences), "final_loss_scale": scale, "loss_scale_halvings": rescaled}


def summarize(model, out, sparsities, bits):
    #small per-unit record (no tensors): what p3/q3 allocation and the sensitivity plots need
    n= out["n_sequences"]
    units= []

    for name, lin in eligible_linears(model):
        u= out["units"][name]
        w= lin.weight.detach()
        g_mean= u["g_sum"].to(w.device) / n
        layer= int(name.split(".")[2])

        units.append({
            "name": name,
            "layer": layer,
            "proj": name.split(".")[-1],
            "shape": list(w.shape),
            "taylor_mean_all": taylor_scores(w, g_mean).mean().item(),
            "taylor_pruned_mean": {str(s): pruned_set_sensitivity(w, g_mean, u["sumsq"].sqrt(), s) for s in sparsities},
            "fisher_quant": {str(b): u["fisher"][b] for b in bits}
        })

    n_eligible= sum(lin.weight.numel() for _, lin in eligible_linears(model))

    return {
        "units": units,
        "n_sequences": n,
        "final_loss_scale": out["final_loss_scale"],
        "loss_scale_halvings": out["loss_scale_halvings"],
        "eligible_weights": n_eligible,
        "non_eligible_params": unique_param_count(model) - n_eligible
    }


def sensitivity_path(args):
    #frozen allocation inputs live in splits/task1 (committed); smoke ones never do
    base= os.path.join(args.output_dir, "smoke", "task1", "splits") if args.smoke else os.path.join("splits", "task1")

    return os.path.join(base, "sensitivity.json")


def main(args):
    cfg= load_config(args.config)
    set_seed(cfg["seed"])

    device= torch.device(args.device)
    print("Using device:", device)

    path= sensitivity_path(args)

    if os.path.exists(path):
        print(f"{path} exists, skipping")
        return

    data_root= resolve_data_root(cfg["data_repo"], args.data_root)

    #calibration sequences only (p.3): never dev or evaluation data
    seqs= load_sequences(cfg, data_root, "calibration", cfg["calibration_sequences"])

    if args.smoke:
        seqs= [{"token_ids": s["token_ids"][:64]} for s in seqs[:cfg["smoke"]["max_examples"]]]

    model= load_model(cfg, device, args.smoke)
    dtype= next(model.parameters()).dtype
    quant= cfg["quantization"]
    sens_cfg= cfg["sensitivity"]

    if device.type == "cuda":
        #fail fast (before 128 backward passes) if this design can't fit the gpu
        c= model.config
        need= gradient_pass_gpu_bytes(unique_param_count(model), sum(m.weight.numel() for _, m in eligible_linears(model)),
                                      len(seqs[0]["token_ids"]), c.vocab_size, c.hidden_size, c.num_hidden_layers)
        total= torch.cuda.get_device_properties(device).total_memory
        print(f"gradient pass estimate {need / 2**30:.2f} GiB of {total / 2**30:.2f} GiB")
        assert need < GPU_FRACTION * total, "gradient pass would not fit this gpu"
        torch.cuda.reset_peak_memory_stats(device)

    start= time.perf_counter()
    out= gradient_pass(
        model, seqs, device,
        bits=quant["q3_allowed_bits"],
        group_size=quant["group_size"],
        scale_dtype=getattr(torch, quant["scale_dtype"]),
        loss_scale=sens_cfg["loss_scale"] if dtype == torch.float16 else 1.0
    )
    elapsed= time.perf_counter() - start
    print(f"gradient pass done in {elapsed:.0f}s")

    summary= summarize(model, out, cfg["pruning"]["p3_sparsities"], quant["q3_allowed_bits"])

    info= run_info(cfg, device, dtype, model_key="task1_base")
    info.update({
        "smoke": args.smoke,
        "data": cfg["data"]["calibration"],
        "measure_p3": "mean |g w| over wanda-pruned weights at the target sparsity, g = mean gradient over calibration sequences",
        "measure_q3": "sum over sequences and weights of g_i^2 * (Q_b(w) - w)^2, q1 quantizer on unchanged weights",
        "wall_clock_seconds": elapsed,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    })

    write_json(path, {**info, **summary})
    print(f"sensitivity: {len(summary['units'])} units, {elapsed:.1f}s, loss-scale halvings {out['loss_scale_halvings']} -> {path}")


if __name__ == "__main__":
    parser= argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/task1/base.yaml"
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
    parser.add_argument(
        "--smoke",
        action="store_true"
    )

    args= parser.parse_args()
    main(args)
