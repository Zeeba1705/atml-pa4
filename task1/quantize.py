import math

import torch
import torch.nn.functional as F

from task1.awq import awq_search, fold_into_norm, fold_into_linear, InputScaledLinear


def quantize_groups(w, bits, group_size, scale_dtype=torch.float32):
    #asymmetric min/max per group of input columns, range widened to include 0 (decision 1.2)
    dout, din= w.shape
    n_groups= math.ceil(din / group_size)
    pad= n_groups * group_size - din
    max_q= 2 ** bits - 1

    #zero padding never changes a zero-inclusive min/max, so padded entries don't affect the scale
    wg= F.pad(w.float(), (0, pad)).reshape(dout, n_groups, group_size)

    w_min= wg.min(dim=2).values.clamp(max=0.0)
    w_max= wg.max(dim=2).values.clamp(min=0.0)

    scale= (w_max - w_min) / max_q
    scale= torch.where(scale > 0, scale, torch.ones_like(scale))

    #round the scale to its storage dtype first so q is computed with the scale we actually store
    scale= scale.to(scale_dtype)
    s= scale.float()

    zero= torch.round(-w_min / s).clamp(0, max_q)
    q= torch.round(wg / s[:, :, None]) + zero[:, :, None]
    q= q.clamp(0, max_q)

    return q.reshape(dout, -1).to(torch.int32), scale, zero.to(torch.int32)


def dequantize_groups(q, scale, zero, din):
    dout, n_groups= scale.shape
    group_size= q.shape[1] // n_groups

    qg= q.reshape(dout, n_groups, group_size).float()
    w_hat= (qg - zero.float()[:, :, None]) * scale.float()[:, :, None]

    return w_hat.reshape(dout, -1)[:, :din]


def packed_bytes(dout, din, bits, group_size):
    #codes and zero-points are separate b-bit streams, each byte-padded; fp16 scale per group
    n_groups= math.ceil(din / group_size)

    code_bytes= math.ceil(dout * n_groups * group_size * bits / 8)
    zero_bytes= math.ceil(dout * n_groups * bits / 8)
    scale_bytes= dout * n_groups * 2

    return code_bytes + zero_bytes + scale_bytes


def fake_quantize(w, bits, group_size, scale_dtype):
    #quantize then dequantize: weights stay fp16 in memory (fake-quant), bytes are counted from packed_bytes
    q, scale, zero= quantize_groups(w, bits, group_size, scale_dtype)

    return dequantize_groups(q, scale, zero, din=w.shape[1])


def make_quantizer(method, bits, group_size, scale_dtype, alpha_grid=None, folds=None, unfolded=None):
    #returns a compress_fn for calibrate.sequential_compress; method is "uniform" (q1) or "awq" (q2, q3)
    #bits is one int for every projection, or (q3) a list with one {proj: bits} dict per block
    def bits_of(idx, name):
        return bits if isinstance(bits, int) else bits[idx][name]

    def quantizer(b):
        return lambda w: fake_quantize(w, b, group_size, scale_dtype)

    def compress(idx, linears, stats, layer):
        mods= dict(linears)
        infos= {name: {"bits": bits_of(idx, name), "extra_fp16_bytes": 0} for name in mods}

        if method == "awq":
            #search one alpha per group of projections sharing an input, using the uncompressed block's inputs
            groups= [(prev, names) for prev, names in folds.items()] + [(None, [name]) for name in unfolded]
            chosen= []

            for prev, names in groups:
                weights= [mods[n].weight for n in names]
                xs= stats[names[0]]["x"]
                x_mean_abs= stats[names[0]]["sumabs"] / stats[names[0]]["n"]

                fns= [quantizer(bits_of(idx, n)) for n in names]
                alpha, s, losses= awq_search(xs, x_mean_abs, weights, alpha_grid, fns)
                chosen.append((prev, names, s))

                for n in names:
                    infos[n].update({"awq_alpha": alpha, "awq_loss": min(losses), "awq_loss_alpha0": losses[alpha_grid.index(0.0)]})

            #apply all scales after the search (as in the llm-awq reference)
            for prev, names, s in chosen:
                if prev is None:
                    #not foldable (o_proj under gqa): keep fp16 input scales and count them
                    name= names[0]
                    wrapped= InputScaledLinear(mods[name], s)
                    setattr(layer.self_attn, name, wrapped)
                    infos[name]["extra_fp16_bytes"]= 2 * s.numel()
                elif prev in mods:
                    fold_into_linear(mods[prev], mods[names[0]], s)
                    infos[prev]["out_scale"]= s
                    infos[names[0]]["in_scale"]= s
                else:
                    fold_into_norm(getattr(layer, prev), [mods[n] for n in names], s)

                    for n in names:
                        infos[n]["in_scale"]= s

        for name, linear in linears:
            w= linear.weight
            dout, din= w.shape
            b= bits_of(idx, name)
            w.copy_(quantizer(b)(w).to(w.dtype))

            infos[name].update({
                "shape": [dout, din],
                "packed_bytes": packed_bytes(dout, din, b, group_size),
                "storage": "fake-quant fp16 in memory; packed_bytes is the serialized size"
            })

        return infos

    return compress
