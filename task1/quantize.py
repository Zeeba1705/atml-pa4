import math

import torch
import torch.nn.functional as F


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
