import time

import torch
import torch.nn as nn

from task1.load import ELIGIBLE_ATTN, ELIGIBLE_MLP


class StopForward(Exception):
    pass


class Catcher(nn.Module):
    #replaces block 0 to grab its inputs and the kwargs the model passes to every block
    def __init__(self, layer):
        super().__init__()
        self.attention_type= layer.attention_type
        self.inputs= []
        self.kwargs= []

    def forward(self, hidden_states, **kwargs):
        #kwargs (rotary cos/sin, mask) carry the batch dimension, so keep them per batch
        self.inputs.append(hidden_states)
        self.kwargs.append({k: v for k, v in kwargs.items() if k != "past_key_values"})
        raise StopForward


def layer_linears(layer):
    linears= [(name, getattr(layer.self_attn, name)) for name in ELIGIBLE_ATTN]
    linears+= [(name, getattr(layer.mlp, name)) for name in ELIGIBLE_MLP]

    return linears


@torch.no_grad()
def block_inputs(model, sequences, device, batch_size):
    layers= model.model.layers
    first= layers[0]
    catcher= Catcher(first)
    layers[0]= catcher

    try:
        for i in range(0, len(sequences), batch_size):
            ids= torch.tensor([s["token_ids"] for s in sequences[i:i + batch_size]], device=device)

            try:
                model(ids, use_cache=False)
            except StopForward:
                pass
    finally:
        layers[0]= first

    return catcher.inputs, catcher.kwargs


def run_block(layer, inputs, kwargs):
    out= layer(inputs, **kwargs)

    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def sequential_compress(model, sequences, device, batch_size, compress_fn, cache_inputs=False):
    #decision 1.5: block l is calibrated on outputs of the already-compressed blocks < l
    hidden, kwargs= block_inputs(model, sequences, device, batch_size)
    records= []
    t_stats= 0.0
    t_compress= 0.0
    t_error= 0.0

    for idx, layer in enumerate(model.model.layers):
        linears= layer_linears(layer)

        #1. input statistics of every projection, from the uncompressed block
        start= time.perf_counter()
        stats= {name: {"sumsq": 0.0, "sumabs": 0.0, "n": 0, "x": []} for name, _ in linears}

        def stats_hook(name):
            def hook(module, inp, out):
                x= inp[0].reshape(-1, inp[0].shape[-1])
                xf= x.float()
                stats[name]["sumsq"]= stats[name]["sumsq"] + xf.pow(2).sum(dim=0)
                stats[name]["sumabs"]= stats[name]["sumabs"] + xf.abs().sum(dim=0)
                stats[name]["n"]+= x.shape[0]

                #awq needs the inputs themselves for its scale search
                if cache_inputs:
                    stats[name]["x"].append(x.detach())
            return hook

        handles= [linear.register_forward_hook(stats_hook(name)) for name, linear in linears]

        for x, kw in zip(hidden, kwargs):
            run_block(layer, x, kw)

        for h in handles:
            h.remove()

        t_stats+= time.perf_counter() - start

        #2. compress the block
        start= time.perf_counter()
        original= {name: (linear.weight.detach().clone(), None if linear.bias is None else linear.bias.detach().clone()) for name, linear in linears}
        infos= compress_fn(idx, linears, stats, layer)
        del stats
        t_compress+= time.perf_counter() - start

        #modules may have been wrapped (awq o_proj) and inputs/outputs rescaled by folds:
        #in_scale/out_scale map a module's actual input/output back to the original space
        in_scale= {name: info.pop("in_scale", None) for name, info in infos.items()}
        out_scale= {name: info.pop("out_scale", None) for name, info in infos.items()}
        linears= layer_linears(layer)

        #3. relative output error per projection, and outputs for the next block
        start= time.perf_counter()
        err= {name: [0.0, 0.0] for name, _ in linears}

        def error_hook(name):
            def hook(module, inp, out):
                #relative error of the module's real output vs the original projection on the same (original-space) input
                x= inp[0].reshape(-1, inp[0].shape[-1]).float()
                y= out.reshape(-1, out.shape[-1]).float()
                w, b= original[name]

                if in_scale.get(name) is not None:
                    x= x * in_scale[name].float()

                if out_scale.get(name) is not None:
                    y= y * out_scale[name].float()

                ref= x @ w.float().T

                if b is not None:
                    ref= ref + b.float()

                err[name][0]+= (y - ref).pow(2).sum().item()
                err[name][1]+= ref.pow(2).sum().item()
            return hook

        handles= [linear.register_forward_hook(error_hook(name)) for name, linear in linears]
        hidden= [run_block(layer, x, kw) for x, kw in zip(hidden, kwargs)]

        for h in handles:
            h.remove()

        t_error+= time.perf_counter() - start

        for name, linear in linears:
            records.append({
                "layer": idx,
                "proj": name,
                "rel_output_error": err[name][0] / max(err[name][1], 1e-30),
                **infos.get(name, {})
            })

        del original

        if (idx + 1) % 7 == 0:
            print(f"calibrated block {idx + 1}/{len(model.model.layers)}")

    times= {"calibration_seconds": t_stats, "compression_seconds": t_compress, "error_pass_seconds": t_error}

    return records, times
