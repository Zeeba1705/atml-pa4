#task 2 part C kv quantization (handout p.8, p.9 C3/C4; decisions 2.10, 2.11, 2.21). asymmetric b-bit groups of 64
#elements, fp16 scale + fp16 offset (= group min) per group, round-half-to-even, codes clamped to [0, 2^b - 1].
#layouts: token-wise = groups of 64 channels inside one token; channel-wise = groups of 64 tokens inside one channel.
#only complete 64-token groups are quantized, the trailing L mod 64 tokens stay fp16 (both layouts, k and v)
import torch

GROUP= 64
META_BYTES= 4   #fp16 scale + fp16 offset


def mb_bytes(L, bits, nl, hkv, dh):
    #handout p.8: M_b(L) = 2 nl hkv dh [floor(L/64)(64b/8 + 4) + 2 (L mod 64)]
    return 2 * nl * hkv * dh * ((L // GROUP) * (GROUP * bits // 8 + META_BYTES) + 2 * (L % GROUP))


def fp16_bytes(L, nl, hkv, dh):
    #M_16(L) = 4 nl hkv dh L
    return 4 * nl * hkv * dh * L


def quantize_last(x, bits):
    #x [..., 64] -> codes uint8 [..., 64], scale and offset fp16 [..., 1]; codes use the stored (fp16) metadata
    assert x.shape[-1] == GROUP, x.shape
    xf= x.float()
    mn= xf.amin(-1, keepdim=True)
    mx= xf.amax(-1, keepdim=True)
    levels= 2 ** bits - 1
    scale= ((mx - mn) / levels).half()
    scale= torch.where(scale == 0, torch.ones_like(scale), scale)
    offset= mn.half()
    codes= torch.round((xf - offset.float()) / scale.float()).clamp(0, levels).to(torch.uint8)

    return codes, scale, offset


def dequantize_last(codes, scale, offset, dtype=torch.float16):
    return (codes.float() * scale.float() + offset.float()).to(dtype)


def to_groups(t, layout):
    #t [B, H, N, D] -> [B, H, a, b, 64] with groups on the last axis
    B, H, N, D= t.shape

    if layout == "token":
        assert D % GROUP == 0, D
        return t.reshape(B, H, N, D // GROUP, GROUP)

    assert layout == "channel" and N % GROUP == 0, (layout, N)

    return t.reshape(B, H, N // GROUP, GROUP, D).transpose(-1, -2)


def from_groups(g, layout, shape):
    B, H, N, D= shape

    if layout == "token":
        return g.reshape(B, H, N, D)

    return g.transpose(-1, -2).reshape(B, H, N, D)


def fake_quant(t, bits, layout):
    #quantize -> dequantize, same shape and dtype (decision 2.11: quality runs store dequantized fp16 values)
    codes, scale, offset= quantize_last(to_groups(t, layout), bits)

    return from_groups(dequantize_last(codes, scale, offset, t.dtype), layout, t.shape)


def fake_quant_span(t, start, end, bits, layout):
    #quantize tokens [start, end) of t [B, H, N, D] in place; end - start must be a multiple of 64 for channel layout
    t[:, :, start:end]= fake_quant(t[:, :, start:end].contiguous(), bits, layout)


def pack(codes, bits):
    #codes uint8 [..., 64] -> uint8 [..., 64 * bits / 8], little-end first
    per= 8 // bits
    c= codes.reshape(*codes.shape[:-1], codes.shape[-1] // per, per).to(torch.int32)
    shifts= torch.arange(per, device=codes.device, dtype=torch.int32) * bits

    return (c << shifts).sum(-1).to(torch.uint8)


def unpack(packed, bits):
    per= 8 // bits
    shifts= torch.arange(per, device=packed.device, dtype=torch.int32) * bits
    c= (packed.to(torch.int32).unsqueeze(-1) >> shifts) & (2 ** bits - 1)

    return c.reshape(*packed.shape[:-1], packed.shape[-1] * per).to(torch.uint8)


def encode(t, bits, layout):
    #real packed encoding of t [B, H, N, D]: complete groups packed, trailing N mod 64 tokens kept fp16 (both layouts)
    n_full= (t.shape[2] // GROUP) * GROUP
    body= t[:, :, :n_full].contiguous()
    codes, scale, offset= quantize_last(to_groups(body, layout), bits)

    return {"payload": pack(codes, bits), "scale": scale, "offset": offset, "residual": t[:, :, n_full:].half().contiguous(),   #the format stores the trailing tokens as fp16
            "shape": tuple(body.shape), "bits": bits, "layout": layout}


def decode(enc, dtype=torch.float16):
    codes= unpack(enc["payload"], enc["bits"])
    body= from_groups(dequantize_last(codes, enc["scale"], enc["offset"], dtype), enc["layout"], enc["shape"])

    return torch.cat([body, enc["residual"].to(dtype)], dim=2)


def encoded_nbytes(enc):
    #actual serialized bytes of the tensors (payload + metadata + fp16 residual)
    return sum(enc[k].numel() * enc[k].element_size() for k in ["payload", "scale", "offset", "residual"])
