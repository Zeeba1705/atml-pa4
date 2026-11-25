#task 2a attention operators (handout p.6, p.8): dense explicit reference, exact tiled online softmax, pytorch sdpa
#all causal, inputs (B, H, T, d). matmuls run in the input dtype, softmax statistics in fp32 (decision 2.19)
import math

import torch
import torch.nn.functional as F


def dense_attention(q, k, v, mask=None):
    #explicit (T, T) scores; mask True = may attend (default causal). a row with no visible key returns 0
    T= q.shape[-2]
    stat= torch.float64 if q.dtype == torch.float64 else torch.float32

    if mask is None:
        mask= torch.ones(T, T, dtype=torch.bool, device=q.device).tril()

    s= (q @ k.transpose(-1, -2)).to(stat) / math.sqrt(q.shape[-1])
    s= s.masked_fill(~mask, float("-inf"))

    #stable softmax: subtract the row max; all-masked rows get max 0 so exp(-inf) = 0 and the sum is 0
    m= s.amax(dim=-1, keepdim=True)
    m= torch.where(torch.isinf(m), torch.zeros_like(m), m)
    p= torch.exp(s - m)
    denom= p.sum(dim=-1, keepdim=True)
    p= p / denom.clamp(min=torch.finfo(stat).tiny)

    return (p.to(v.dtype) @ v)


def tile_count(T, block_q, block_k):
    #tiles at or below the causal diagonal (some key index <= some query index in the tile)
    n= 0

    for qs in range(0, T, block_q):
        qe= min(qs + block_q, T)
        n+= math.ceil(qe / block_k)

    return n


def tiled_attention(q, k, v, block_q=128, block_k=128, return_stats=False):
    #online softmax over key tiles (handout p.6): m' = max(m, max s), l' = e^(m-m') l + sum e^(s-m'), o' = e^(m-m') o + sum e^(s-m') v
    #never materializes the (T, T) scores; tiles fully above the diagonal are skipped.
    #python reference: the backward is autograd, which keeps every tile (not a fused flashattention kernel)
    B, H, T, d= q.shape
    stat= torch.float64 if q.dtype == torch.float64 else torch.float32
    scale= 1.0 / math.sqrt(d)
    outs= []
    tiles= 0
    max_tile= 0

    for qs in range(0, T, block_q):
        qe= min(qs + block_q, T)
        qt= q[:, :, qs:qe]
        q_idx= torch.arange(qs, qe, device=q.device)[:, None]

        m= torch.full((B, H, qe - qs, 1), float("-inf"), dtype=stat, device=q.device)
        l= torch.zeros((B, H, qe - qs, 1), dtype=stat, device=q.device)
        o= torch.zeros((B, H, qe - qs, v.shape[-1]), dtype=stat, device=q.device)

        #causal: only key tiles starting at or before the last query of this tile
        for ks in range(0, qe, block_k):
            ke= min(ks + block_k, T)
            s= (qt @ k[:, :, ks:ke].transpose(-1, -2)).to(stat) * scale

            if ke - 1 > qs:
                #tile crosses the diagonal: hide keys after each query
                k_idx= torch.arange(ks, ke, device=q.device)[None, :]
                s= s.masked_fill(k_idx > q_idx, float("-inf"))

            m_new= torch.maximum(m, s.amax(dim=-1, keepdim=True))
            #a row with nothing visible yet keeps m = -inf; use 0 there so no inf - inf
            m_safe= torch.where(torch.isinf(m_new), torch.zeros_like(m_new), m_new)
            corr= torch.exp(m - m_safe)
            p= torch.exp(s - m_safe)

            l= corr * l + p.sum(dim=-1, keepdim=True)
            o= corr * o + p @ v[:, :, ks:ke].to(stat)
            m= m_new

            tiles+= 1
            max_tile= max(max_tile, s.numel())

        outs.append((o / l.clamp(min=torch.finfo(stat).tiny)).to(v.dtype))

    out= torch.cat(outs, dim=2)

    if return_stats:
        return out, {"tiles_computed": tiles, "max_score_tile_elements": max_tile}

    return out


def sdpa_attention(q, k, v):
    #pytorch picks the backend; the bench records which one ran
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)
