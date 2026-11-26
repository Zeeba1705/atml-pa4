#task 2 part C: byte formula, packing, grouping, trailing fp16 rule, budgets, eviction safeguards, rope positions after
#eviction, question scores vs real attention weights, prompt split and evidence mapping
import gzip
import json
import os

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from task2.kv_quant import mb_bytes, fp16_bytes, quantize_last, dequantize_last, to_groups, fake_quant, fake_quant_span, encode, decode, encoded_nbytes, pack, unpack
from task2.kv_cache import prefill, forward, gather_cache, clone_cache, greedy_decode, encode_case
from task2.kv_policies import evict_counts, evict_bytes, hybrid_extra, hybrid_bytes, hybrid_layout, question_index, question_scores, sink_recent_index, Policy

CFG= {"eviction": {"sink_tokens": 8, "recent_tokens": 64}, "quant": {"budget_bits": 4, "hybrid_low_bits": 2}}


def test_byte_formula_hand_count_and_serialized_bytes():
    #L = 130: 2 complete groups + 2 trailing fp16 tokens per channel; one layer, one head, dh = 1
    assert mb_bytes(130, 4, 1, 1, 1) == 2 * (2 * (32 + 4) + 2 * 2)
    assert mb_bytes(130, 2, 1, 1, 1) == 2 * (2 * (16 + 4) + 2 * 2)
    assert fp16_bytes(130, 1, 1, 1) == 4 * 130

    torch.manual_seed(0)
    k, v= torch.randn(1, 2, 130, 128).half(), torch.randn(1, 2, 130, 128).half()
    real= encoded_nbytes(encode(k, 4, "channel")) + encoded_nbytes(encode(v, 4, "token"))
    assert real == mb_bytes(130, 4, 1, 2, 128)
    #token-wise keys give the same byte count (same group size, metadata and trailing rule)
    assert encoded_nbytes(encode(k, 4, "token")) + encoded_nbytes(encode(v, 4, "token")) == real


@pytest.mark.parametrize("bits", [2, 4])
def test_pack_round_trip_and_decode_matches_fake_quant(bits):
    torch.manual_seed(1)
    codes= torch.randint(0, 2 ** bits, (3, 5, 64), dtype=torch.uint8)
    assert torch.equal(unpack(pack(codes, bits), bits), codes)
    assert pack(codes, bits).shape[-1] == 64 * bits // 8

    t= torch.randn(1, 2, 150, 128).half()
    for layout in ["token", "channel"]:
        out= decode(encode(t, bits, layout))
        assert torch.equal(out[:, :, :128], fake_quant(t[:, :, :128], bits, layout))
        assert torch.equal(out[:, :, 128:], t[:, :, 128:])          #trailing 22 tokens untouched fp16


def test_hand_group_quantization():
    x= torch.arange(64, dtype=torch.float16)
    codes, scale, offset= quantize_last(x, 4)
    assert float(offset) == 0.0 and abs(float(scale) - 63 / 15) < 1e-2
    assert codes.max() == 15 and codes.min() == 0
    err= (dequantize_last(codes, scale, offset).float() - x.float()).abs().max()
    assert err <= float(scale) / 2 + 1e-3
    #constant group: scale guard, exact reconstruction
    c, s, o= quantize_last(torch.full((64,), 3.5).half(), 2)
    assert torch.equal(dequantize_last(c, s, o), torch.full((64,), 3.5).half())


def test_channel_vs_token_grouping():
    t= torch.randn(1, 2, 128, 128)
    g= to_groups(t, "channel")
    assert g.shape == (1, 2, 2, 128, 64)
    assert torch.equal(g[0, 1, 1, 7], t[0, 1, 64:128, 7])         #one channel, 64 consecutive tokens
    g= to_groups(t, "token")
    assert g.shape == (1, 2, 128, 2, 64)
    assert torch.equal(g[0, 1, 5, 1], t[0, 1, 5, 64:128])          #one token, 64 consecutive channels


def tiny(eager=False):
    torch.manual_seed(0)
    config= Qwen2Config(vocab_size=101, hidden_size=256, intermediate_size=256, num_hidden_layers=2, num_attention_heads=4,
                        num_key_value_heads=2, attn_implementation="eager" if eager else "sdpa")

    return Qwen2ForCausalLM(config).float().eval()


def test_trailing_rule_and_appended_groups():
    model= tiny()
    ids= torch.randint(0, 101, (150,)).tolist()
    cache, _= prefill(model, ids)
    orig= clone_cache(cache)
    p= Policy("int4_channel_k", 150, mb_bytes(150, 4, 2, 2, 64), model, CFG)
    p.compress(cache, None)
    k, k0= cache.layers[0].keys, orig.layers[0].keys
    assert not torch.equal(k[:, :, :128], k0[:, :, :128]) and torch.equal(k[:, :, 128:], k0[:, :, 128:])

    #appending 41 tokens leaves 22 + 41 = 63 trailing fp16; the 42nd completes the group [128, 192)
    forward(model, cache, list(range(41)), 150)
    p.after_append(cache)
    assert p.qupto == 128
    forward(model, cache, [5], 191)
    p.after_append(cache)
    assert p.qupto == 192 and p.live_bytes(cache) == mb_bytes(192, 4, 2, 2, 64)


def test_budgets_are_tight_and_safeguards_kept():
    nl, hkv, dh= 28, 2, 128
    for L in [1500, 2077, 4100]:
        B= mb_bytes(L, 4, nl, hkv, dh)
        n= evict_counts(L, B, nl, hkv, dh, 8, 64)
        for m in n:
            assert evict_bytes(m, n[m], nl, hkv, dh) <= B < evict_bytes(m, n[m] + 1, nl, hkv, dh)
        e= hybrid_extra(L, B, 2, nl, hkv, dh, 8, 64)
        assert hybrid_bytes(L, e, 2, nl, hkv, dh, 8, 64) <= B
        assert e == L - 72 or hybrid_bytes(L, e + 1, 2, nl, hkv, dh, 8, 64) > B
        fp16, groups= hybrid_layout(L, e, 8, 64)
        assert fp16 + 64 * groups == L and (L - 72 - e) - 64 * groups < 64

    scores= torch.rand(3, 2, 500)
    idx= question_index(scores, 100, 8, 64)
    assert idx.shape == (3, 2, 100) and (idx.diff(dim=-1) > 0).all()
    for must in list(range(8)) + list(range(436, 500)):
        assert (idx == must).any(-1).all()
    sr= sink_recent_index(500, 100, 8, 3, 2, "cpu")
    assert sr[0, 0].tolist() == list(range(8)) + list(range(408, 500))


def test_evicted_cache_keeps_true_rope_positions():
    #gathering a cache and appending with true positions == a full forward that masks the evicted positions
    model= tiny()
    torch.manual_seed(3)
    ids= torch.randint(0, 101, (90,)).tolist()
    L, suffix= 80, ids[80:]
    keep= sorted(set(range(8)) | set(range(40, 80)) | {13, 21})
    mask= torch.ones(90, 90).tril().bool()
    for i in range(L, 90):
        mask[i, [j for j in range(L) if j not in keep]]= False

    with torch.no_grad():
        ref= model(torch.tensor([ids]), attention_mask=mask[None, None]).logits[0, -1]

    cache, _= prefill(model, ids[:L])
    idx= torch.tensor(keep)[None, None].expand(2, 2, -1)
    got= forward(model, gather_cache(cache, idx), suffix, L)[-1]
    assert torch.allclose(got, ref, atol=1e-4)

    #100% retention reproduces ordinary generation
    full, first= prefill(model, ids)
    a= greedy_decode(model, full, first[-1], 90, 6, [])
    cache, _= prefill(model, ids[:L])
    c2= gather_cache(cache, torch.arange(L)[None, None].expand(2, 2, -1))
    b= greedy_decode(model, c2, forward(model, c2, suffix, L)[-1], 90, 6, [])
    assert a == b


def test_question_scores_equal_eager_attention_mass():
    sdpa, eager= tiny(), tiny(eager=True)
    ids= torch.randint(0, 101, (70,)).tolist()
    L= 60
    cache, _= prefill(sdpa, ids[:L])
    scores= question_scores(sdpa, cache, ids[L:], L)
    assert cache.get_seq_length() == L

    with torch.no_grad():
        att= eager(torch.tensor([ids]), output_attentions=True).attentions           #per layer [1, 4, 70, 70]
    for i, a in enumerate(att):
        ref= a[0, :, L:, :L].sum(1).view(2, 2, L).sum(1)
        assert torch.allclose(scores[i], ref, atol=1e-4)


def test_prompt_split_and_evidence_tokens():
    from transformers import AutoTokenizer
    from common.io_utils import load_config, model_revision, resolve_data_root

    cfg= load_config("configs/task2/kv.yaml")
    try:
        tok= AutoTokenizer.from_pretrained(cfg["models"]["target"], revision=model_revision(cfg, "target"))
    except OSError:
        pytest.skip("tokenizer not available offline")

    root= resolve_data_root(cfg["data_repo"])
    case= json.loads(gzip.open(os.path.join(root, cfg["data"])).readline())
    enc= encode_case(tok, case, cfg["question_marker"])
    assert tok.decode(enc["ids"][enc["L"]:]).lstrip("\n").startswith("QUESTION_BEGIN")
    assert case["code"] in tok.decode([enc["ids"][i] for i in enc["evidence"]])
    assert enc["L"] > max(enc["evidence"])
