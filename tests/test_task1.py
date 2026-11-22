import math

import torch
import torch.nn as nn
from transformers import Qwen2Config, Qwen2ForCausalLM

from task1.load import eligible_linears, eligible_weight_count, unique_param_count
from task1.prune import input_norms, magnitude_scores, wanda_scores, row_prune_mask, two_four_mask
from task1.quantize import quantize_groups, dequantize_groups, packed_bytes
from task1.awq import awq_scales, fold_into_norm, fold_into_linear, InputScaledLinear
from task1.allocate import rescale_sparsities, integer_zero_counts, row_counts
from task1.evaluate_ppl import window_nll, pooled_ppl
from task1.evaluate_hellaswag import preprocess, build_query_and_choices, pick_answers

ELIGIBLE= ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def tiny_qwen2():
    torch.manual_seed(6304)

    config= Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=True
    )

    model= Qwen2ForCausalLM(config).float().eval()

    return model


#--- eligible matrices and tied weights ---

def test_eligible_linears_order_and_count():
    model= tiny_qwen2()

    linears= eligible_linears(model)

    names= [name for name, _ in linears]
    assert len(names) == 2 * 7
    assert names[0] == "model.layers.0.self_attn.q_proj"
    assert names[-1] == "model.layers.1.mlp.down_proj"
    assert [n.split(".")[-1] for n in names[:7]] == ELIGIBLE

    #per layer: q 32x32, k 16x32, v 16x32, o 32x32, gate/up 64x32, down 32x64
    per_layer= 1024 + 512 + 512 + 1024 + 2048 + 2048 + 2048
    assert eligible_weight_count(model) == 2 * per_layer


def test_tied_embedding_counted_once():
    model= tiny_qwen2()

    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()

    #embed 64x32 (shared with lm_head), per layer weights+biases+2 norms, final norm
    per_layer= (1024 + 32) + (512 + 16) + (512 + 16) + 1024 + 3 * 2048 + 2 * 32
    expected= 64 * 32 + 2 * per_layer + 32
    assert unique_param_count(model) == expected

    #the state dict lists lm_head separately, so naive counting double counts it
    naive= sum(t.numel() for t in model.state_dict().values())
    assert naive == expected + 64 * 32


#--- pruning ---

def test_input_norms():
    x= torch.tensor([[3.0, 0.0], [4.0, 1.0]])

    assert torch.allclose(input_norms(x), torch.tensor([5.0, 1.0]))


def test_wanda_score_is_weight_times_input_norm():
    w= torch.tensor([[1.0, -2.0, 3.0, -4.0, 5.0, -6.0],
                     [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]])

    #one calibration token, so the column norms are just the absolute values
    x= torch.tensor([[6.0, 3.0, 2.0, 1.0, 1.0, -0.5]])
    x_norm= input_norms(x)

    expected= torch.tensor([[6.0, 6.0, 6.0, 4.0, 5.0, 3.0],
                            [36.0, 15.0, 8.0, 3.0, 2.0, 0.5]])
    assert torch.allclose(wanda_scores(w, x_norm), expected)
    assert torch.allclose(magnitude_scores(w), w.abs())

    #prune 3 per row: row 0 differs between the two scores, row 1 agrees
    keep_wanda= row_prune_mask(wanda_scores(w, x_norm), 3)
    keep_mag= row_prune_mask(magnitude_scores(w), 3)

    assert keep_wanda.tolist() == [[True, True, True, False, False, False],
                                   [True, True, True, False, False, False]]
    assert keep_mag.tolist() == [[False, False, False, True, True, True],
                                 [True, True, True, False, False, False]]


def test_row_prune_exact_count():
    torch.manual_seed(0)
    scores= torch.rand(16, 128)

    for s in [0.25, 0.5, 0.75]:
        n_prune= int(s * 128)
        keep= row_prune_mask(scores, n_prune)

        assert keep.shape == scores.shape
        assert ((~keep).sum(dim=1) == n_prune).all()

        #pruned weights are the lowest scores of their row
        kept_min= scores.masked_fill(~keep, float("inf")).min(dim=1).values
        pruned_max= scores.masked_fill(keep, -float("inf")).max(dim=1).values
        assert (pruned_max < kept_min).all()

    #per-row counts (used by nonuniform allocation)
    keep= row_prune_mask(scores[:2], torch.tensor([1, 100]))
    assert (~keep).sum(dim=1).tolist() == [1, 100]


def test_two_four_mask_keeps_top_two_of_each_four():
    scores= torch.tensor([[0.1, 0.9, 0.5, 0.3, 4.0, 3.0, 2.0, 1.0]])

    keep= two_four_mask(scores)
    assert keep.tolist() == [[False, True, True, False, True, True, False, False]]

    torch.manual_seed(1)
    scores= torch.rand(8, 64)
    keep= two_four_mask(scores)

    groups= keep.view(8, 16, 4)
    assert (groups.sum(dim=2) == 2).all()
    assert keep.float().mean().item() == 0.5

    #within each group the kept scores beat the dropped ones
    g_scores= scores.view(8, 16, 4)
    kept_min= g_scores.masked_fill(~groups, float("inf")).min(dim=2).values
    dropped_max= g_scores.masked_fill(groups, -float("inf")).max(dim=2).values
    assert (dropped_max < kept_min).all()


#--- quantization ---

def test_quantize_hand_group_with_padding():
    #din=6 with group 4 -> second group is padded to 4
    w= torch.tensor([[-0.5, 0.0, 0.25, 1.0, 0.3, 0.9]])

    q, scale, zero= quantize_groups(w, bits=2, group_size=4)

    #group 1: s=(1.0-(-0.5))/3=0.5, z=round(0.5/0.5)=1, round(0.25/0.5)=round(0.5)=0 (half to even)
    #group 2: range is widened to include 0 -> s=0.9/3=0.3, z=0
    assert q.shape == (1, 8)
    assert torch.allclose(scale, torch.tensor([[0.5, 0.3]]))
    assert zero.tolist() == [[1, 0]]
    assert q[:, :6].tolist() == [[0, 1, 1, 3, 1, 3]]

    w_hat= dequantize_groups(q, scale, zero, din=6)
    assert w_hat.shape == (1, 6)
    assert torch.allclose(w_hat, torch.tensor([[-0.5, 0.0, 0.0, 1.0, 0.3, 0.9]]), atol=1e-6)


def test_quantize_round_trip_error_bound():
    torch.manual_seed(2)
    w= torch.randn(8, 300)

    for bits in [2, 3, 4, 8]:
        q, scale, zero= quantize_groups(w, bits=bits, group_size=128)

        assert q.min() >= 0 and q.max() <= 2 ** bits - 1
        assert zero.min() >= 0 and zero.max() <= 2 ** bits - 1

        w_hat= dequantize_groups(q, scale, zero, din=300)

        #every element is within half a step of its group's grid
        step= scale.repeat_interleave(128, dim=1)[:, :300]
        assert ((w - w_hat).abs() <= step / 2 + 1e-6).all()


def test_quantize_fp16_scale_is_what_gets_used():
    torch.manual_seed(5)
    w= torch.randn(8, 256).half()

    for bits in [2, 3, 4, 8]:
        q, scale, zero= quantize_groups(w, bits=bits, group_size=128, scale_dtype=torch.float16)

        assert scale.dtype == torch.float16
        assert q.min() >= 0 and q.max() <= 2 ** bits - 1

        #fp16 rounding of the scale can push the group max past the top code by a tiny fraction of a step
        w_hat= dequantize_groups(q, scale, zero, din=256)
        step= scale.float().repeat_interleave(128, dim=1)
        slack= (2 ** bits - 1) * step * 2 ** -10
        assert ((w.float() - w_hat).abs() <= step / 2 + slack + 1e-6).all()


def test_packed_bytes_hand_count():
    #3x200, group 128 -> padded to 256, 2 groups per row, 3 bits
    #payload 3*256*3 bits = 288 B, zeros 3*2*3 = 18 bits -> 3 B, fp16 scales 3*2*2 = 12 B
    assert packed_bytes(dout=3, din=200, bits=3, group_size=128) == 288 + 3 + 12

    #2x128 at 4 bits: payload 128 B, zeros 8 bits = 1 B, scales 4 B
    assert packed_bytes(dout=2, din=128, bits=4, group_size=128) == 128 + 1 + 4


#--- awq rescaling ---

def test_awq_scales():
    x_mean_abs= torch.tensor([1.0, 4.0])

    assert torch.allclose(awq_scales(x_mean_abs, alpha=0.0), torch.ones(2))

    #s = x^alpha normalised so that sqrt(max*min) = 1
    assert torch.allclose(awq_scales(x_mean_abs, alpha=1.0), torch.tensor([0.5, 2.0]))


def test_awq_folding_preserves_outputs_before_rounding():
    model= tiny_qwen2()

    ids= torch.randint(0, 64, (2, 12))
    with torch.no_grad():
        before= model(ids).logits

    torch.manual_seed(3)
    for layer in model.model.layers:
        attn= layer.self_attn
        mlp= layer.mlp

        s_qkv= torch.rand(32) * 1.5 + 0.5
        s_gate_up= torch.rand(32) * 1.5 + 0.5
        s_down= torch.rand(64) * 1.5 + 0.5
        s_o= torch.rand(32) * 1.5 + 0.5

        fold_into_norm(layer.input_layernorm, [attn.q_proj, attn.k_proj, attn.v_proj], s_qkv)
        fold_into_norm(layer.post_attention_layernorm, [mlp.gate_proj, mlp.up_proj], s_gate_up)
        fold_into_linear(mlp.up_proj, mlp.down_proj, s_down)

        #o_proj cannot be folded into v_proj under gqa, so it keeps an fp16 input scale
        attn.o_proj= InputScaledLinear(attn.o_proj, s_o)

    with torch.no_grad():
        after= model(ids).logits

    assert torch.allclose(before, after, atol=1e-5)


def test_fold_into_linear_scales_bias():
    torch.manual_seed(4)
    prev= nn.Linear(4, 3)
    nxt= nn.Linear(3, 2)
    x= torch.randn(5, 4)

    with torch.no_grad():
        before= nxt(prev(x))
        fold_into_linear(prev, nxt, torch.tensor([2.0, 0.5, 4.0]))
        after= nxt(prev(x))

    assert torch.allclose(before, after, atol=1e-6)


#--- nonuniform allocation ---

def test_rescale_sparsities_hits_target_within_bounds():
    sizes= [100, 200, 300, 400]

    for target in [0.5, 0.75]:
        s= rescale_sparsities([0.2, 0.5, 0.9, 0.4], sizes, target, 0.10, 0.85)

        mean= sum(si * n for si, n in zip(s, sizes)) / sum(sizes)
        assert abs(mean - target) < 1e-6
        assert all(0.10 - 1e-9 <= si <= 0.85 + 1e-9 for si in s)

        #a shift keeps the ranking of the raw values
        assert s[0] <= s[3] <= s[1] <= s[2]

    s= rescale_sparsities([0.3, 0.3, 0.3], [10, 20, 30], 0.5, 0.10, 0.85)
    assert all(abs(si - 0.5) < 1e-6 for si in s)


def test_integer_zero_counts_exact_total():
    #sizes like real projections: q/o 1536x1536, k/v 256x1536, mlp 8960x1536
    sizes= [1536 * 1536, 256 * 1536, 256 * 1536, 1536 * 1536, 8960 * 1536, 8960 * 1536, 1536 * 8960]
    n_total= sum(sizes)

    for target in [0.5, 0.75]:
        raw= [0.3, 0.9, 0.8, 0.5, 0.6, 0.6, 0.2]
        s= rescale_sparsities(raw, sizes, target, 0.10, 0.85)
        target_zeros= int(target * n_total)

        zeros= integer_zero_counts(s, sizes, target_zeros)

        assert sum(zeros) == target_zeros
        assert all(isinstance(z, int) for z in zeros)
        assert all(abs(z - si * n) < 1 for z, si, n in zip(zeros, s, sizes))


def test_row_counts_spread_evenly():
    assert row_counts(10, 4) == [3, 3, 2, 2]
    assert row_counts(8, 4) == [2, 2, 2, 2]

    counts= row_counts(1234567, 1536)
    assert sum(counts) == 1234567
    assert max(counts) - min(counts) <= 1


#--- perplexity scoring ---

def test_window_nll_hand_computed():
    token_ids= torch.tensor([0, 2, 1])

    #position 0 predicts token 2 with uniform probs -> -log(1/3)
    #position 1 predicts token 1 with probs [0.5, 0.25, 0.25] -> -log(0.25)
    logits= torch.tensor([[0.0, 0.0, 0.0],
                          [math.log(2.0), 0.0, 0.0],
                          [5.0, -1.0, 3.0]])

    nll, n= window_nll(logits, token_ids, score_start=1, score_end=3)
    assert n == 2
    assert abs(nll - math.log(12.0)) < 1e-5

    #only the second target
    nll, n= window_nll(logits, token_ids, score_start=2, score_end=3)
    assert n == 1
    assert abs(nll - math.log(4.0)) < 1e-5


def test_pooled_ppl_pools_tokens_not_windows():
    n, mean_nll, ppl= pooled_ppl([math.log(12.0), math.log(4.0)], [2, 1])

    assert n == 3
    assert abs(mean_nll - math.log(48.0) / 3) < 1e-9
    assert abs(ppl - math.exp(math.log(48.0) / 3)) < 1e-9


#--- hellaswag (lm-eval hellaswag v1.0 conventions) ---

def test_hellaswag_preprocess_matches_lm_eval():
    assert preprocess("  Making a cake [title] Mix the flour.  [step] Add  eggs ") == "Making a cake. Mix the flour.  Add eggs"


def test_hellaswag_query_and_choices():
    example= {
        "activity_label": "Baking cookies",
        "ctx_a": "A woman is in a kitchen.",
        "ctx_b": "she pours flour",
        "endings": ["into a bowl.", " [step] onto the floor.", "and sings.", "then leaves."],
        "label": "0"
    }

    query, choices, gold= build_query_and_choices(example)

    assert query == "Baking cookies: A woman is in a kitchen. She pours flour"
    #lm-eval strips before removing brackets, so a leading tag leaves a leading space (kept to match lm-eval)
    assert choices == ["into a bowl.", " onto the floor.", "and sings.", "then leaves."]
    assert gold == 0


def test_acc_norm_hand_example():
    lls= [-10.0, -12.0, -9.0, -20.0]
    choices= ["a" * 10, "b" * 30, "c" * 5, "d" * 100]

    #raw: argmax ll -> 2; per char: [-1.0, -0.4, -1.8, -0.2] -> 3
    pred, pred_norm= pick_answers(lls, choices)
    assert pred == 2
    assert pred_norm == 3


def test_acc_norm_excludes_leading_space():
    #lm-eval scores " " + choice but divides by len(choice)
    #per char without space: [-1.0, -0.95] -> 1; with the space it would be [-0.5, -0.63] -> 0
    pred, pred_norm= pick_answers([-1.0, -1.9], ["a", "bb"])
    assert pred == 0
    assert pred_norm == 1
