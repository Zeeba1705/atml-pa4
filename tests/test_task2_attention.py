#task 2a: dense reference, tiled online softmax, sdpa (handout p.6, pp.8-9)
import math

import torch

from task2.attention_ops import dense_attention, tiled_attention, sdpa_attention, tile_count


def qkv(B=2, H=3, T=37, d=8, dtype=torch.float64, seed=0, scale=1.0):
    g= torch.Generator().manual_seed(seed)

    return [(scale * torch.randn(B, H, T, d, generator=g)).to(dtype) for _ in range(3)]


def test_dense_hand_example():
    #one head, two tokens: token 0 sees only itself, token 1 sees softmax([s10, s11]) over v0, v1
    q= torch.tensor([[[[1.0, 0.0], [0.0, 2.0]]]], dtype=torch.float64)
    k= torch.tensor([[[[1.0, 1.0], [2.0, 0.0]]]], dtype=torch.float64)
    v= torch.tensor([[[[1.0, 2.0], [3.0, -1.0]]]], dtype=torch.float64)

    out= dense_attention(q, k, v)
    s10= (0 * 1 + 2 * 1) / math.sqrt(2)
    s11= (0 * 2 + 2 * 0) / math.sqrt(2)
    w0= math.exp(s10) / (math.exp(s10) + math.exp(s11))

    assert torch.allclose(out[0, 0, 0], v[0, 0, 0])
    assert torch.allclose(out[0, 0, 1], w0 * v[0, 0, 0] + (1 - w0) * v[0, 0, 1])


def test_tiled_equals_dense_across_tile_sizes():
    q, k, v= qkv(T=70)
    ref= dense_attention(q, k, v)

    #tiles that divide T, don't divide it, are larger than it, and rectangular
    for bq, bk in [(32, 32), (64, 64), (128, 128), (256, 256), (7, 13), (16, 64), (70, 1)]:
        assert torch.allclose(tiled_attention(q, k, v, bq, bk), ref, atol=1e-12), (bq, bk)


def test_sdpa_equals_dense():
    q, k, v= qkv(T=50)

    assert torch.allclose(sdpa_attention(q, k, v), dense_attention(q, k, v), atol=1e-12)


def test_gradients_match_dense():
    q, k, v= qkv(T=45)
    g= torch.randn(q.shape, generator=torch.Generator().manual_seed(9), dtype=torch.float64)
    grads= []

    for fn in [lambda a, b, c: dense_attention(a, b, c), lambda a, b, c: tiled_attention(a, b, c, 16, 8), sdpa_attention]:
        a, b, c= (x.clone().requires_grad_(True) for x in (q, k, v))
        (fn(a, b, c) * g).sum().backward()
        grads.append([a.grad, b.grad, c.grad])

    for other in grads[1:]:
        assert all(torch.allclose(x, y, atol=1e-10) for x, y in zip(grads[0], other))


def test_causality_future_perturbation():
    q, k, v= qkv(T=64)
    k2, v2= k.clone(), v.clone()
    k2[:, :, 40:]+= 5.0
    v2[:, :, 40:]-= 3.0

    for fn in [dense_attention, lambda a, b, c: tiled_attention(a, b, c, 16, 16), sdpa_attention]:
        a= fn(q, k, v)
        b= fn(q, k2, v2)

        assert torch.allclose(a[:, :, :40], b[:, :, :40], atol=1e-12)
        assert not torch.allclose(a[:, :, 40:], b[:, :, 40:])


def test_extreme_scores_stay_finite():
    #scores ~1e4: a naive exp would overflow; the max-subtracted forms must not
    for dtype in [torch.float32, torch.float16]:
        q, k, v= qkv(T=40, d=16, dtype=dtype, scale=1.0)
        q= q * 300

        ref= dense_attention(q.double(), k.double(), v.double())
        dense= dense_attention(q, k, v)
        tiled= tiled_attention(q, k, v, 8, 8)

        assert torch.isfinite(dense).all() and torch.isfinite(tiled).all()
        #dense and tiled see the same input-dtype scores, so they agree with each other up to the rounding of
        #s - m at |s| ~ 3e3 (fp32 spacing there is ~2.4e-4; the tiled form subtracts running, not final, maxima)
        assert (dense.double() - tiled.double()).abs().max() < (1e-2 if dtype == torch.float16 else 2e-4)

        #fp32 is also close to the fp64 reference; fp16 scores ~1e3 carry rounding error ~1, so its error is
        #measured in the benchmark, not asserted here
        if dtype == torch.float32:
            assert (dense.double() - ref).abs().max() < 1e-3


def test_all_masked_row_convention():
    #with a custom mask a query may see no key: output 0 and finite gradients, instead of nan
    q, k, v= qkv(B=1, H=1, T=4, d=2)
    mask= torch.tril(torch.ones(4, 4, dtype=torch.bool))
    mask[2]= False

    q.requires_grad_(True)
    out= dense_attention(q, k, v, mask=mask)
    out.sum().backward()

    assert torch.equal(out[0, 0, 2], torch.zeros(2, dtype=out.dtype))
    assert torch.isfinite(q.grad).all()


def test_tiled_never_builds_full_score_matrix():
    q, k, v= qkv(B=1, H=2, T=100, d=4)
    _, stats= tiled_attention(q, k, v, 32, 16, return_stats=True)

    assert stats["max_score_tile_elements"] == 1 * 2 * 32 * 16
    #causal skipping: only tiles with some key <= some query are computed
    assert stats["tiles_computed"] == tile_count(100, 32, 16) < math.ceil(100 / 32) * math.ceil(100 / 16)


def test_tile_count_hand():
    #T=4, 2x2 tiles: (q0-1,k0-1), (q2-3,k0-1), (q2-3,k2-3); (q0-1,k2-3) is above the diagonal
    assert tile_count(4, 2, 2) == 3
