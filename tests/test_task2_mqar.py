#task 2b: masks, linear recurrence, query-position loss, receptive field, schedule, data and metrics
import math

import torch
import torch.nn.functional as F

from common.io_utils import load_config, resolve_data_root
from task2.mqar_model import (
    window_mask, dilated_mask, head_masks, linear_attention, linear_attention_recurrent,
    MQARModel, attended_pairs, state_bytes
)
from task2.mqar_data import load_generator, make_batch, build_stream, stream_examples
from task2.train_mqar import lr_at, query_loss
from task2.eval_mqar import evaluate_model, summarize_queries

CFG= load_config("configs/task2/mqar.yaml")
ACFG= CFG["attention"]
GEN= load_generator(resolve_data_root(CFG["data_repo"]))


def tiny_model(seed=0):
    torch.manual_seed(seed)

    return MQARModel(CFG["model"], ACFG, CFG["seq_len"], CFG["num_classes"]).eval()


#--- masks ---

def test_window_mask_allows_exactly_last_w_positions():
    m= window_mask(256, 64)

    assert m[100].sum() == 64 and m[100, 37:101].all() and not m[100, 36] and not m[100, 101]
    assert m[10].sum() == 11   #near the start only the 11 causal positions exist
    assert not torch.triu(m, diagonal=1).any()


def test_dilated_mask_offsets():
    m= dilated_mask(256, 4, 252)
    allowed= set(torch.nonzero(m[255]).flatten().tolist())

    assert allowed == {255 - 4 * j for j in range(64)}   #offsets 0,4,...,252
    assert set(torch.nonzero(m[3]).flatten().tolist()) == {3}
    assert set(torch.nonzero(m[9]).flatten().tolist()) == {1, 5, 9}   #offsets not divisible by 4 are never allowed
    assert not torch.triu(m, diagonal=1).any()


def test_local_dilated_heads():
    h= head_masks("local_dilated", 256, ACFG, 4)

    assert h.shape == (4, 256, 256)
    assert torch.equal(h[0], window_mask(256, 64)) and torch.equal(h[1], window_mask(256, 64))
    assert torch.equal(h[2], dilated_mask(256, 4, 252)) and torch.equal(h[3], dilated_mask(256, 4, 252))
    assert head_masks("dense", 256, ACFG, 4) is None and head_masks("linear", 256, ACFG, 4) is None


#--- linear attention ---

def test_linear_recurrence_equals_parallel_causal_form():
    g= torch.Generator().manual_seed(0)
    q, k, v= (torch.randn(2, 3, 9, 4, generator=g, dtype=torch.float64) for _ in range(3))

    par= linear_attention(q, k, v, 1e-6)
    rec, (s, z)= linear_attention_recurrent(q, k, v, 1e-6)

    assert torch.allclose(par, rec, atol=1e-10)
    #state is d x d (+ d normalizer) per head, independent of T
    assert s.shape == (2, 3, 4, 4) and z.shape == (2, 3, 4)

    #hand check of the last position: phi(q)^T S / (phi(q)^T z + eps), phi = elu + 1
    phi= lambda x: F.elu(x) + 1
    s_hand= sum(torch.outer(phi(k[0, 0, j]), v[0, 0, j]) for j in range(9))
    z_hand= sum(phi(k[0, 0, j]) for j in range(9))
    a= phi(q[0, 0, 8]) @ s_hand / (phi(q[0, 0, 8]) @ z_hand + 1e-6)

    assert torch.allclose(par[0, 0, 8], a, atol=1e-10)


#--- model ---

def test_every_attention_is_causal():
    model= tiny_model()
    g= torch.Generator().manual_seed(1)
    x= torch.randint(2, 256, (2, 256), generator=g)
    y= x.clone()
    y[:, 150:]= torch.randint(2, 256, (2, 106), generator=g)

    for attention in ["dense", "window", "local_dilated", "linear"]:
        with torch.no_grad():
            a= model(x, attention=attention)
            b= model(y, attention=attention)

        assert torch.allclose(a[:, :150], b[:, :150], atol=1e-5), attention
        assert not torch.allclose(a[:, 150:], b[:, 150:], atol=1e-5), attention


def test_two_layer_window_receptive_field_is_126():
    #no path from token j to the output at i when i - j > 2(W-1) = 126
    model= tiny_model()
    x= torch.randint(2, 256, (1, 256), generator=torch.Generator().manual_seed(2))
    h= model.embed(x).detach().requires_grad_(True)

    model.forward_embeds(h, attention="window")[0, 200].sum().backward()
    reach= h.grad[0].abs().sum(dim=1)

    assert reach[200 - 126] > 0
    assert (reach[:200 - 126] == 0).all()
    assert ACFG["receptive_field_limit"] == 2 * (ACFG["window"] - 1)


def test_same_parameters_for_every_attention():
    #only the operator changes: identical parameter count and names
    model= tiny_model()
    n= sum(p.numel() for p in model.parameters())

    assert n == sum(p.numel() for p in tiny_model(1).parameters())
    assert model.head.out_features == 64


def test_theoretical_pairs_and_state_bytes():
    pairs= attended_pairs("window", 256, ACFG, 4)

    assert attended_pairs("dense", 256, ACFG, 4)["per_head"] == [256 * 257 // 2] * 4
    assert pairs["per_head"] == [63 * 64 // 2 + 193 * 64] * 4
    dil= sum(min(i, 252) // 4 + 1 for i in range(256))
    assert attended_pairs("local_dilated", 256, ACFG, 4)["per_head"] == [pairs["per_head"][0]] * 2 + [dil] * 2

    #linear: per layer, H heads of (d x d + d) fp32 values; softmax kinds: 2 * T * H * d fp32 cached k/v
    assert state_bytes("linear", 256, CFG["model"], ACFG, 4)["per_layer"] == 4 * (32 * 32 + 32) * 4
    assert state_bytes("dense", 256, CFG["model"], ACFG, 4)["per_layer"] == 2 * 256 * 4 * 32 * 4
    assert state_bytes("window", 256, CFG["model"], ACFG, 4)["per_layer"] == 2 * 64 * 4 * 32 * 4
    #dilated heads must keep the last 253 positions (future queries need other residues mod 4)
    assert state_bytes("local_dilated", 256, CFG["model"], ACFG, 4)["per_layer"] == 2 * (2 * 64 + 2 * 253) * 32 * 4


#--- loss and schedule ---

def test_query_loss_only_counts_query_positions():
    g= torch.Generator().manual_seed(3)
    logits= torch.randn(2, 5, 64, generator=g)
    labels= torch.full((2, 5), -100)
    labels[0, 2]= 7
    labels[1, 4]= 30

    hand= -(F.log_softmax(logits[0, 2], -1)[7] + F.log_softmax(logits[1, 4], -1)[30]) / 2
    assert torch.allclose(query_loss(logits, labels), hand)

    #changing logits anywhere else leaves the loss unchanged
    other= logits.clone()
    other[0, 0]+= 100.0
    other[1, 3]-= 50.0
    assert torch.allclose(query_loss(other, labels), hand)


def test_lr_schedule():
    peak= CFG["optimizer"]["peak_lr"]

    assert math.isclose(lr_at(1, CFG), peak / 200)
    assert math.isclose(lr_at(200, CFG), peak)
    assert math.isclose(lr_at(4500, CFG), 0.15 * peak)
    lrs= [lr_at(u, CFG) for u in range(200, 4501)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))


#--- data ---

def test_make_batch_labels_at_query_positions():
    ex= GEN.make_example("train", 4, "mixed", 0)
    tokens, labels, dist= make_batch([ex])

    assert tokens[0].tolist() == ex["tokens"]
    q= ex["query_positions"]
    assert labels[0, q].tolist() == [a - 128 for a in ex["answer_ids"]]
    assert (labels[0] != -100).sum() == 4
    assert dist[0, q].tolist() == ex["distances"]


def test_stream_is_the_supplied_generator():
    stream= build_stream(GEN, 8, "near", 6, workers=1)

    #update u (0-based), batch row b -> index u * batch_size + b
    exs= stream_examples(stream, update=1, batch_size=3)
    assert [e["tokens"] for e in exs] == [GEN.make_example("train", 8, "near", i)["tokens"] for i in [3, 4, 5]]
    assert exs[0]["query_positions"] == GEN.make_example("train", 8, "near", 3)["query_positions"]


#--- metrics ---

def test_summarize_queries_by_cell_and_bin():
    rows= [
        {"K": 4, "regime": "near", "distance": 5, "correct": 1, "loss": 0.1},
        {"K": 4, "regime": "near", "distance": 40, "correct": 0, "loss": 2.0},
        {"K": 4, "regime": "near", "distance": 63, "correct": 1, "loss": 0.3},
        {"K": 16, "regime": "far", "distance": 200, "correct": 0, "loss": 4.0}
    ]
    s= summarize_queries(rows, CFG["distance_bins"])

    assert s["cells"]["K4_near"]["accuracy"] == 2 / 3 and s["cells"]["K4_near"]["n_queries"] == 3
    assert s["cells"]["K4_near"]["chance_1_over_k"] == 0.25 and s["cells"]["K4_near"]["chance_label_space"] == 1 / 64
    assert s["bins"]["K4_[32,64)"]["accuracy"] == 0.5
    assert s["bins"]["K16_[128,256)"]["n_queries"] == 1


def test_evaluate_model_rows():
    model= tiny_model()
    exs= [GEN.make_example("development", 4, "near", i) for i in range(3)]

    rows= evaluate_model(model, exs, torch.device("cpu"), "dense", batch_size=2)

    assert len(rows) == 12
    assert all(0 <= r["pred"] < 64 and r["distance"] == r["query_position"] - r["source_position"] for r in rows)
