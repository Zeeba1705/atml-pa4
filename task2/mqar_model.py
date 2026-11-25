#task 2b mqar model (handout p.7): 2 pre-norm blocks, d=128, 4 heads x 32, 4x mlp, learned token emb + sinusoidal pos
#only the attention operator changes between methods; every method has the same parameters
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

ATTENTIONS= ["dense", "window", "local_dilated", "linear"]


def sinusoidal_positions(T, d):
    #vaswani et al. 2017: sin on even dims, cos on odd dims
    pos= torch.arange(T, dtype=torch.float32)[:, None]
    freq= torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe= torch.zeros(T, d)
    pe[:, 0::2]= torch.sin(pos * freq)
    pe[:, 1::2]= torch.cos(pos * freq)

    return pe


def window_mask(T, W):
    #True = query i may attend key j, j in [i-W+1, i]
    i= torch.arange(T)[:, None]
    j= torch.arange(T)[None, :]

    return (i - j >= 0) & (i - j < W)


def dilated_mask(T, step, max_offset):
    #offsets {0, step, ..., max_offset} back from the query, causal
    off= torch.arange(T)[:, None] - torch.arange(T)[None, :]

    return (off >= 0) & (off <= max_offset) & (off % step == 0)


def head_masks(attention, T, acfg, num_heads):
    #(H, T, T) boolean masks; None where the operator needs no explicit mask
    if attention in ["dense", "linear"]:
        return None

    win= window_mask(T, acfg["window"])

    if attention == "window":
        return win[None].expand(num_heads, T, T).clone()

    dil= dilated_mask(T, acfg["dilated_offsets_step"], acfg["dilated_offsets_max"])

    return torch.stack([dil if h in acfg["dilated_heads"] else win for h in range(num_heads)])


def elu_feature(x):
    return F.elu(x) + 1


def linear_attention(q, k, v, eps):
    #exact parallel form of the causal recurrence: a_i = sum_{j<=i} (phi(q_i).phi(k_j)) v_j / (sum_{j<=i} phi(q_i).phi(k_j) + eps)
    #materializes a (T, T) causal score tensor; fine at T=256 (the recurrent form is the constant-state version)
    fq= elu_feature(q)
    fk= elu_feature(k)
    T= q.shape[-2]

    scores= (fq @ fk.transpose(-1, -2)) * torch.ones(T, T, device=q.device, dtype=torch.bool).tril()
    den= scores.sum(dim=-1, keepdim=True) + eps

    return (scores @ v) / den


def linear_attention_recurrent(q, k, v, eps):
    #S_t = S_{t-1} + phi(k_t) v_t^T, z_t = z_{t-1} + phi(k_t), a_t = phi(q_t)^T S_t / (phi(q_t)^T z_t + eps)
    fq= elu_feature(q)
    fk= elu_feature(k)
    B, H, T, d= q.shape

    s= torch.zeros(B, H, d, v.shape[-1], dtype=q.dtype, device=q.device)
    z= torch.zeros(B, H, d, dtype=q.dtype, device=q.device)
    out= []

    for t in range(T):
        s= s + fk[:, :, t, :, None] * v[:, :, t, None, :]
        z= z + fk[:, :, t]
        num= (fq[:, :, t, None, :] @ s).squeeze(-2)
        den= (fq[:, :, t] * z).sum(dim=-1, keepdim=True) + eps
        out.append(num / den)

    return torch.stack(out, dim=2), (s, z)


class Attention(nn.Module):
    def __init__(self, d, num_heads, eps):
        super().__init__()

        self.num_heads= num_heads
        self.head_dim= d // num_heads
        self.eps= eps
        self.qkv= nn.Linear(d, 3 * d)
        self.out= nn.Linear(d, d)

    def forward(self, x, attention, mask):
        B, T, d= x.shape

        #(B, T, 3d) -> 3 x (B, H, T, dh)
        q, k, v= self.qkv(x).view(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        if attention == "dense":
            a= F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif attention in ["window", "local_dilated"]:
            #dense boolean mask: a correctness implementation, not a sparse kernel
            a= F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        elif attention == "linear":
            a= linear_attention(q, k, v, self.eps)
        else:
            raise ValueError(attention)

        return self.out(a.transpose(1, 2).reshape(B, T, d))


class Block(nn.Module):
    def __init__(self, d, num_heads, mlp_mult, eps):
        super().__init__()

        self.ln1= nn.LayerNorm(d)
        self.attn= Attention(d, num_heads, eps)
        self.ln2= nn.LayerNorm(d)
        self.mlp= nn.Sequential(nn.Linear(d, mlp_mult * d), nn.GELU(), nn.Linear(mlp_mult * d, d))

    def forward(self, x, attention, mask):
        x= x + self.attn(self.ln1(x), attention, mask)

        return x + self.mlp(self.ln2(x))


class MQARModel(nn.Module):
    def __init__(self, mcfg, acfg, T, num_classes, attention="dense"):
        super().__init__()

        d= mcfg["hidden_dim"]
        assert d == mcfg["num_heads"] * mcfg["head_dim"]

        self.acfg= acfg
        self.num_heads= mcfg["num_heads"]
        self.attention= attention
        self.tok_emb= nn.Embedding(256, d)
        self.register_buffer("pos", sinusoidal_positions(T, d), persistent=False)
        self.blocks= nn.ModuleList([Block(d, mcfg["num_heads"], mcfg["mlp_mult"], acfg["linear_eps"]) for _ in range(mcfg["num_layers"])])
        self.ln_f= nn.LayerNorm(d)
        #64-way head over value ids 128..191 (decision 2.3)
        self.head= nn.Linear(d, num_classes)
        self.masks= {}

    def embed(self, tokens):
        return self.tok_emb(tokens) + self.pos[:tokens.shape[1]]

    def mask_for(self, attention, T, device):
        #cached per (operator, length, device)
        key= (attention, T, str(device))

        if key not in self.masks:
            m= head_masks(attention, T, self.acfg, self.num_heads)
            self.masks[key]= None if m is None else m.to(device)

        return self.masks[key]

    def forward_embeds(self, h, attention=None):
        attention= attention or self.attention
        mask= self.mask_for(attention, h.shape[1], h.device)

        for block in self.blocks:
            h= block(h, attention, mask)

        return self.head(self.ln_f(h))

    def forward(self, tokens, attention=None):
        #logits (B, T, num_classes) at every position; the loss reads only query positions
        return self.forward_embeds(self.embed(tokens), attention)


def attended_pairs(attention, T, acfg, num_heads):
    #theoretical (query, key) pairs per head over a length-T sequence, causal
    if attention == "linear":
        return {"per_head": None, "total": None, "note": "no pairwise scores in the recurrent form; state is d x d per head"}

    if attention == "dense":
        per= [T * (T + 1) // 2] * num_heads
    else:
        m= head_masks(attention, T, acfg, num_heads)
        per= [int(m[h].sum()) for h in range(num_heads)]

    return {"per_head": per, "total": sum(per), "per_query_max": [max_positions(attention, T, acfg, h) for h in range(num_heads)]}


def max_positions(attention, T, acfg, h):
    #most key/value positions one query can access in one layer
    if attention == "dense":
        return T

    if attention == "window" or (attention == "local_dilated" and h not in acfg["dilated_heads"]):
        return min(T, acfg["window"])

    return min(T, acfg["dilated_offsets_max"] // acfg["dilated_offsets_step"] + 1)


def state_bytes(attention, T, mcfg, acfg, bytes_per):
    #persistent decode state per layer at length T: k/v cache positions that future queries may still need,
    #or the linear state (d x d + d per head)
    H= mcfg["num_heads"]
    dh= mcfg["head_dim"]

    if attention == "linear":
        per_layer= H * (dh * dh + dh) * bytes_per
    else:
        keep= []

        for h in range(H):
            if attention == "dense":
                keep.append(T)
            elif attention == "window" or h not in acfg["dilated_heads"]:
                keep.append(min(T, acfg["window"]))
            else:
                #queries at t, t+1, ... need every offset residue, so the last max_offset+1 positions are kept
                keep.append(min(T, acfg["dilated_offsets_max"] + 1))

        per_layer= 2 * sum(keep) * dh * bytes_per

    return {"per_layer": per_layer, "total": per_layer * mcfg["num_layers"], "bytes_per_value": bytes_per}
