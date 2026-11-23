import math

import torch

from common.timing import time_fn
from task1.load import eligible_linears, unique_param_count


def storage_counts(model):
    #dense fp16 accounting; quantized conditions add their packed bytes on top of this later
    eligible= eligible_linears(model)

    n_eligible= sum(linear.weight.numel() for _, linear in eligible)
    nnz= sum(int((linear.weight != 0).sum().item()) for _, linear in eligible)
    n_unique= unique_param_count(model)
    n_other= n_unique - n_eligible

    #p.4 sparse estimate: 1 mask bit per eligible weight + 16 bits per nonzero, byte aligned per tensor
    sparse_eligible= sum(math.ceil(linear.weight.numel() / 8) + 2 * int((linear.weight != 0).sum().item()) for _, linear in eligible)

    return {
        "eligible_weights": n_eligible,
        "eligible_nonzeros": nnz,
        "eligible_nonzero_fraction": nnz / n_eligible,
        "unique_params": n_unique,
        "non_eligible_params": n_other,
        "dense_fp16_bytes": 2 * n_unique,
        "sparse16_estimate_bytes": sparse_eligible + 2 * n_other
    }


def latency_prompt(windows, length):
    #first tokens of the concatenated c4-val windows (decision 1.9), timing only
    ids= []

    for w in windows:
        ids.extend(w["token_ids"])

        if len(ids) >= length:
            break

    assert len(ids) >= length

    return torch.tensor([ids[:length]])


def reset_peak(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory(device):
    if device.type != "cuda":
        return {"peak_allocated_bytes": None, "peak_reserved_bytes": None}

    return {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)
    }


@torch.no_grad()
def measure_latency(model, windows, cfg, device, prefill_lengths, decode_tokens):
    res= cfg["resources"]
    out= {}

    for length in prefill_lengths:
        ids= latency_prompt(windows, length).to(device)
        reset_peak(device)

        #prefill as in generation: build the cache, lm head only on the last position
        timing= time_fn(
            lambda _: model(ids, use_cache=True, logits_to_keep=1),
            device,
            res["warmup_reps"],
            res["timing_reps"]
        )

        out[f"prefill_{length}"]= {**timing, **peak_memory(device)}
        print(f"prefill {length}: median {timing['median']:.1f} ms")

    #decode: greedy, ignores eos, after an untimed prefill of the first length
    ids= latency_prompt(windows, prefill_lengths[0]).to(device)

    def prefill():
        result= model(ids, use_cache=True, logits_to_keep=1)
        return result.past_key_values, result.logits[:, -1].argmax(-1, keepdim=True)

    def decode(state):
        cache, token= state

        for _ in range(decode_tokens):
            result= model(token, past_key_values=cache, use_cache=True)
            cache= result.past_key_values
            token= result.logits[:, -1].argmax(-1, keepdim=True)

    reset_peak(device)
    timing= time_fn(decode, device, res["warmup_reps"], res["timing_reps"], setup=prefill)

    out[f"decode_{decode_tokens}_after_{prefill_lengths[0]}"]= {**timing, **peak_memory(device), "ms_per_token": timing["median"] / decode_tokens}
    print(f"decode {decode_tokens} tokens: median {timing['median']:.1f} ms")

    return out
