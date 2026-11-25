#p3 (taylor-guided nonuniform wanda) and q3 (fisher-guided mixed precision), decision 1.7
import itertools

import torch
import torch.nn.functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM

from task1.load import eligible_linears
from task1.quantize import fake_quantize, make_quantizer, packed_bytes
from task1.calibrate import sequential_compress
from task1.sensitivity import taylor_scores, pruned_set_sensitivity, fisher_quant_error, gradient_pass
from task1.allocate import rank_raw_sparsities, p3_allocation, q3_allocation

FOLDS= {"input_layernorm": ["q_proj", "k_proj", "v_proj"], "post_attention_layernorm": ["gate_proj", "up_proj"], "up_proj": ["down_proj"]}
GRID= [0.0, 0.5, 1.0]


def tiny_model():
    torch.manual_seed(6304)

    config= Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=True,
        attn_implementation="sdpa"
    )

    return Qwen2ForCausalLM(config).float().eval()


def sequences(n=3, length=12):
    g= torch.Generator().manual_seed(1)

    return [{"token_ids": torch.randint(0, 64, (length,), generator=g).tolist()} for _ in range(n)]


#--- taylor sensitivity (p3) ---

def test_taylor_scores_hand_example():
    w= torch.tensor([[1.0, -2.0], [3.0, 0.5]])
    g= torch.tensor([[0.5, 0.1], [-1.0, 2.0]])

    assert torch.allclose(taylor_scores(w, g), torch.tensor([[0.5, 0.2], [3.0, 1.0]]))


def test_pruned_set_sensitivity_uses_wanda_pruned_weights_only():
    w= torch.tensor([[1.0, -2.0], [3.0, 0.5]])
    g= torch.tensor([[0.5, 0.1], [-1.0, 2.0]])

    #equal input norms -> wanda prunes the smaller |w| per row: (0,0) and (1,1) -> mean |g w| = (0.5 + 1.0) / 2
    assert abs(pruned_set_sensitivity(w, g, torch.ones(2), 0.5) - 0.75) < 1e-6

    #a large input norm on column 0 flips row 0's choice to (0,1); row 1 still prunes (1,1)
    assert abs(pruned_set_sensitivity(w, g, torch.tensor([10.0, 1.0]), 0.5) - 0.6) < 1e-6


def test_fisher_quant_error_hand_example():
    w= torch.tensor([[0.1, -0.7, 0.3, 0.9]])
    g= torch.tensor([[1.0, 2.0, -1.0, 0.5]])

    dw= fake_quantize(w, 2, 4, torch.float32) - w
    expected= (g.pow(2) * dw.pow(2)).sum().item()

    assert abs(fisher_quant_error(g, w, 2, 4, torch.float32) - expected) < 1e-9
    #8 bits is far more precise than 2 bits on the same weights
    assert fisher_quant_error(g, w, 8, 4, torch.float32) < fisher_quant_error(g, w, 2, 4, torch.float32)


def test_gradient_pass_matches_manual_autograd():
    model= tiny_model()
    seqs= sequences()
    bits= [2, 4]

    out= gradient_pass(model, seqs, torch.device("cpu"), bits, group_size=16, scale_dtype=torch.float32)

    #manual: per-sequence mean next-token loss, gradients summed over sequences; fisher term per sequence
    ref= tiny_model()
    linears= dict(eligible_linears(ref))
    g_sum= {n: torch.zeros_like(m.weight) for n, m in linears.items()}
    fisher= {n: {b: 0.0 for b in bits} for n in linears}
    sumsq= {n: torch.zeros(m.weight.shape[1]) for n, m in linears.items()}
    def stats_hook(n):
        #a forward hook must return None, otherwise it replaces the module output
        def hook(mod, inp, out):
            sumsq[n].add_(inp[0].detach().reshape(-1, inp[0].shape[-1]).pow(2).sum(0))
        return hook

    hooks= [m.register_forward_hook(stats_hook(n)) for n, m in linears.items()]

    for s in seqs:
        ids= torch.tensor([s["token_ids"]])
        ref.zero_grad()
        F.cross_entropy(ref(ids).logits[0, :-1], ids[0, 1:]).backward()

        for n, m in linears.items():
            g_sum[n]+= m.weight.grad

            for b in bits:
                dw= fake_quantize(m.weight.detach(), b, 16, torch.float32) - m.weight.detach()
                fisher[n][b]+= (m.weight.grad.pow(2) * dw.pow(2)).sum().item()

    for h in hooks:
        h.remove()

    assert out["n_sequences"] == len(seqs)

    for n in linears:
        assert torch.allclose(out["units"][n]["g_sum"], g_sum[n], atol=1e-6)
        assert torch.allclose(out["units"][n]["sumsq"], sumsq[n], rtol=1e-5)

        for b in bits:
            assert abs(out["units"][n]["fisher"][b] - fisher[n][b]) <= 1e-6 * max(1.0, fisher[n][b])

    #the pass must leave the model unchanged and without gradients
    assert all(p.grad is None for p in model.parameters())
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), tiny_model().state_dict().values()))


#--- p3 allocation ---

def test_rank_raw_sparsities_linear_in_rank():
    #least sensitive unit -> target + lam, most sensitive -> target - lam
    raw= rank_raw_sparsities([5.0, 1.0, 3.0], target=0.5, lam=0.2)

    assert [round(r, 6) for r in raw] == [0.3, 0.7, 0.5]
    assert rank_raw_sparsities([5.0, 1.0, 3.0], 0.5, 0.0) == [0.5, 0.5, 0.5]


def test_p3_allocation_lambda_zero_is_p2():
    shapes= [(1536, 1536), (256, 1536), (8960, 1536), (1536, 8960)]
    sens= [4.0, 1.0, 2.0, 3.0]

    alloc= p3_allocation(sens, shapes, target=0.5, lam=0.0, s_min=0.10, s_max=0.85)

    assert alloc["zeros"] == [dout * din // 2 for dout, din in shapes]
    #per-row counts are what wanda's row mask consumes
    assert all(set(rows) == {din // 2} for rows, (_, din) in zip(alloc["row_zeros"], shapes))


def test_p3_allocation_exact_total_bounds_and_order():
    shapes= [(1536, 1536), (256, 1536), (256, 1536), (1536, 1536), (8960, 1536), (8960, 1536), (1536, 8960)] * 2
    sens= [0.3, 0.9, 0.8, 0.5, 0.6, 0.65, 0.2, 0.31, 0.91, 0.81, 0.51, 0.61, 0.66, 0.21]
    sizes= [a * b for a, b in shapes]

    for target in [0.5, 0.75]:
        alloc= p3_allocation(sens, shapes, target, lam=0.15, s_min=0.10, s_max=0.85)

        #matched to p1/p2 by the total number of zeroed eligible weights
        assert sum(alloc["zeros"]) == sum(int(target * n) for n in sizes)
        assert all(0.10 * n - 1 <= z <= 0.85 * n + 1 for z, n in zip(alloc["zeros"], sizes))
        assert all(sum(rows) == z and max(rows) - min(rows) <= 1 for rows, z in zip(alloc["row_zeros"], alloc["zeros"]))

        #a more sensitive unit never gets a higher sparsity
        s= alloc["sparsity"]
        for i, j in itertools.permutations(range(len(sens)), 2):
            if sens[i] > sens[j]:
                assert s[i] <= s[j] + 1e-9


#--- q3 allocation ---

def test_q3_allocation_matches_brute_force():
    g= torch.Generator().manual_seed(3)
    bits= [2, 3, 4, 8]
    n_units= 4

    #sensitivity falls with bits, bytes rise with bits
    sens= [sorted(torch.rand(4, generator=g).tolist(), reverse=True) for _ in range(n_units)]
    sizes= [(64, 128), (32, 128), (64, 256), (16, 128)]
    bytes_= [[packed_bytes(dout, din, b, 128) for b in bits] for dout, din in sizes]

    for budget in [sum(row[0] for row in bytes_), sum(row[1] for row in bytes_) + 300, sum(row[2] for row in bytes_), 10 ** 9]:
        alloc= q3_allocation(sens, bytes_, bits, budget)

        best= min(
            (sum(sens[u][c[u]] for u in range(n_units)), c)
            for c in itertools.product(range(4), repeat=n_units)
            if sum(bytes_[u][c[u]] for u in range(n_units)) <= budget
        )

        assert alloc["bytes"] <= budget
        assert abs(alloc["objective"] - best[0]) < 1e-9

    #budget for all-8-bit -> everything at 8 bits; budget for all-2-bit -> everything at 2 bits
    assert q3_allocation(sens, bytes_, bits, 10 ** 9)["bits"] == [8] * n_units
    assert q3_allocation(sens, bytes_, bits, sum(row[0] for row in bytes_))["bits"] == [2] * n_units


#--- q3 quantizer: same awq quantizer as q2, only the bit widths change ---

def run_awq(bits):
    model= tiny_model()
    fn= make_quantizer("awq", bits, 16, torch.float32, alpha_grid=GRID, folds=FOLDS, unfolded=["o_proj"])
    records, _= sequential_compress(model, sequences(6, 16), torch.device("cpu"), batch_size=4, compress_fn=fn, cache_inputs=True)

    return model, records


def test_mixed_bits_with_uniform_map_equals_q2():
    names= ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    uniform, _= run_awq(4)
    mixed, _= run_awq([{n: 4 for n in names} for _ in range(2)])

    assert all(torch.equal(a, b) for a, b in zip(uniform.state_dict().values(), mixed.state_dict().values()))


def test_mixed_bits_assignment_and_bytes():
    names= ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    bit_map= [{n: b for n, b in zip(names, [2, 3, 4, 8, 2, 3, 4])}, {n: b for n, b in zip(names, [8, 4, 3, 2, 8, 4, 3])}]

    model, records= run_awq(bit_map)

    for r in records:
        b= bit_map[r["layer"]][r["proj"]]
        dout, din= r["shape"]

        assert r["bits"] == b
        assert r["packed_bytes"] == packed_bytes(dout, din, b, 16)

    #a 2-bit projection has at most 4 distinct values per 16-wide group
    w= model.model.layers[0].self_attn.q_proj.weight
    assert all(len(torch.unique(g)) <= 4 for g in w.reshape(-1, 16))


def test_gradient_pass_fits_a_t4():
    #qwen2.5-1.5b at 512 tokens: the design must leave headroom on a 14.56 GiB T4 (the first design kept a 5.2 GB
    #fp32 gradient sum on the gpu and ran out of memory at sequence 17)
    from task1.sensitivity import gradient_pass_gpu_bytes, GPU_FRACTION

    need= gradient_pass_gpu_bytes(1543714304, 1310195712, 512, 151936, 1536, 28)
    limit= GPU_FRACTION * 14.56 * 2 ** 30

    assert need < limit
    assert need + 4 * 1310195712 > limit   #the old layout (fp32 sum on the gpu) would have been rejected


def test_gradient_pass_sum_on_cpu():
    model= tiny_model()
    out= gradient_pass(model, sequences(2, 8), torch.device("cpu"), [4], 16, torch.float32, sum_device="cpu")

    assert all(u["g_sum"].device.type == "cpu" and u["g_sum"].dtype == torch.float32 for u in out["units"].values())
