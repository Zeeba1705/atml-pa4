#task 2 c5 systems benchmark (handout p.10): actual packed-cache encode and decode at batch 1 and 8 and both context
#lengths, >= 10 warmed cuda-synchronized reps. the cache is a real pre-question qwen cache (first case of each length,
#repeated over the batch). per layout: packing time (quantize + pack, all layers k and v), unpacking/dequantization
#time, serialized bytes vs the formula, and one decode step with unpack-then-sdpa vs the fp16 reference at matched
#shapes. packed bytes and allocated gpu memory are reported separately (dequantized fp16 is what sdpa reads)
import os
import argparse

import torch
from transformers import AutoTokenizer, DynamicCache

from common.seed import set_seed
from common.timing import time_fn, sync
from common.io_utils import load_config, write_json, resolve_data_root, default_output_dir, run_info, model_revision
from task2.kv_quant import encode, decode, encoded_nbytes, mb_bytes, fp16_bytes
from task2.kv_cache import encode_case, prefill
from task2.kv_eval import load_model, read_cases
from task2.kv_policies import shape_of

LAYOUTS= [("int4_channel_k", 4, "channel"), ("int4_token", 4, "token"), ("int2_channel_k", 2, "channel")]


def batch_cache(cache, B):
    new= DynamicCache()

    for i, layer in enumerate(cache.layers):
        new.update(layer.keys.expand(B, -1, -1, -1).contiguous(), layer.values.expand(B, -1, -1, -1).contiguous(), i)

    return new


def encode_all(cache, bits, layout_k):
    return [(encode(l.keys, bits, layout_k), encode(l.values, bits, "token")) for l in cache.layers]


def decode_all(encs):
    new= DynamicCache()

    for i, (k, v) in enumerate(encs):
        new.update(decode(k), decode(v), i)

    return new


def clone(cache):
    new= DynamicCache()

    for i, l in enumerate(cache.layers):
        new.update(l.keys.clone(), l.values.clone(), i)

    return new


def step(model, cache, B, pos, device):
    with torch.no_grad():
        return model(input_ids=torch.full((B, 1), 220, device=device), position_ids=torch.full((B, 1), pos, device=device),
                     cache_position=torch.tensor([cache.get_seq_length()], device=device), past_key_values=cache,
                     use_cache=True, logits_to_keep=1).logits


def peak_of(fn, device):
    if device.type != "cuda":
        return None

    sync(device)
    torch.cuda.reset_peak_memory_stats(device)
    fn()
    sync(device)

    return torch.cuda.max_memory_allocated(device)


def main(args):
    cfg= load_config(args.config)
    set_seed(cfg["seed"])
    device= torch.device(args.device)
    print("Using device:", device)

    b= cfg["bench"]
    warmup, reps= (1, 2) if args.smoke else (b["warmup_reps"], b["timing_reps"])
    out_dir= os.path.join(args.output_dir, "smoke") if args.smoke else args.output_dir
    res_root= os.path.join(out_dir, "results") if args.smoke else args.results_dir
    res_dir= os.path.join(res_root, "task2", "kv_bench")

    data_root= resolve_data_root(cfg["data_repo"], args.data_root)
    cases= read_cases(os.path.join(data_root, cfg["data"]))
    tok= AutoTokenizer.from_pretrained(cfg["models"]["target"], revision=model_revision(cfg, "target"))
    model= load_model(cfg, device, args.smoke)
    nl, hkv, dh= shape_of(model)

    for length in b["context_lengths"]:
        case= next(c for c in cases if c["target_tokens"] == length)
        enc= encode_case(tok, case, cfg["question_marker"])
        L= enc["L"]
        base, _= prefill(model, enc["ids"][:L])

        for B in b["batch_sizes"]:
            path= os.path.join(res_dir, f"len{length}_b{B}.json")

            if os.path.exists(path):
                print(f"{path} exists, skipping")
                continue

            cache= batch_cache(base, B)
            fp16_step= time_fn(lambda c: step(model, c, B, L, device), device, warmup, reps, setup=lambda: clone(cache))
            fp16_peak= peak_of(lambda: step(model, clone(cache), B, L, device), device)
            rows= {"fp16": {"decode_step_ms": fp16_step, "decode_step_peak_allocated_bytes": fp16_peak,
                            "serialized_bytes": fp16_bytes(L, nl, hkv, dh) * B,
                            "allocated_cache_bytes": sum(l.keys.numel() * 2 + l.values.numel() * 2 for l in cache.layers)}}

            for name, bits, layout_k in LAYOUTS:
                encs= encode_all(cache, bits, layout_k)
                real= sum(encoded_nbytes(k) + encoded_nbytes(v) for k, v in encs)
                formula= mb_bytes(L, bits, nl, hkv, dh) * B
                assert real == formula, (name, real, formula)

                rows[name]= {
                    "bits": bits, "layout": f"{layout_k}-wise K / token-wise V",
                    "pack_ms": time_fn(lambda _: encode_all(cache, bits, layout_k), device, warmup, reps),
                    "unpack_dequant_ms": time_fn(lambda _: decode_all(encs), device, warmup, reps),
                    "unpack_plus_decode_step_ms": time_fn(lambda _: step(model, decode_all(encs), B, L, device), device, warmup, reps),
                    "unpack_plus_decode_step_peak_allocated_bytes": peak_of(lambda: step(model, decode_all(encs), B, L, device), device),
                    "serialized_bytes": real, "formula_bytes": formula, "fraction_of_fp16": real / rows["fp16"]["serialized_bytes"]
                }
                print(f"len {length} (L={L}) B={B} {name}: pack {rows[name]['pack_ms']['median']:.2f} ms, unpack {rows[name]['unpack_dequant_ms']['median']:.2f} ms, "
                      f"step+unpack {rows[name]['unpack_plus_decode_step_ms']['median']:.2f} ms vs fp16 step {fp16_step['median']:.2f} ms")
                del encs

            info= run_info(cfg, device, next(model.parameters()).dtype, model_key="target")
            info.update({"case": case["id"], "nominal_length": length, "context_tokens": L, "batch_size": B, "smoke": args.smoke,
                         "warmup": warmup, "reps": reps, "results": rows,
                         "labels": "python/pytorch packing (uint8 nibbles/crumbs), not a fused low-bit attention kernel; sdpa reads the dequantized fp16 copy"})
            write_json(path, info)
            del cache

            if device.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    parser= argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/task2/kv.yaml"
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
