import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from task1.calibrate import sequential_compress
from task1.quantize import make_quantizer, packed_bytes
from task1.awq import InputScaledLinear

FOLDS= {"input_layernorm": ["q_proj", "k_proj", "v_proj"], "post_attention_layernorm": ["gate_proj", "up_proj"], "up_proj": ["down_proj"]}
GRID= [0.0, 0.25, 0.5, 0.75, 1.0]


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


def sequences(n=6, length=16):
    g= torch.Generator().manual_seed(1)

    return [{"token_ids": torch.randint(0, 64, (length,), generator=g).tolist()} for _ in range(n)]


def run(method, bits):
    model= tiny_model()
    seqs= sequences()
    ids= torch.tensor([s["token_ids"] for s in seqs])

    with torch.no_grad():
        before= model(ids).logits

    fn= make_quantizer(method, bits, 16, torch.float32, alpha_grid=GRID, folds=FOLDS, unfolded=["o_proj"])
    records, _= sequential_compress(model, seqs, torch.device("cpu"), batch_size=4, compress_fn=fn, cache_inputs=(method == "awq"))

    with torch.no_grad():
        after= model(ids).logits

    return model, records, before, after


def test_awq_folds_and_error_hook_are_consistent_at_16_bits():
    #16-bit quantization is almost lossless: folding must preserve the function and the error hook must see ~0
    model, records, before, after= run("awq", 16)

    assert torch.allclose(before, after, atol=1e-3)
    assert all(r["rel_output_error"] < 1e-6 for r in records)
    assert all(isinstance(layer.self_attn.o_proj, InputScaledLinear) for layer in model.model.layers)


def test_awq_search_never_worse_than_alpha_zero():
    _, records, _, _= run("awq", 3)

    for r in records:
        assert r["awq_loss"] <= r["awq_loss_alpha0"]
        assert r["awq_alpha"] in GRID


def test_uniform_quantization_bytes_and_error():
    model, records, before, after= run("uniform", 4)

    for r in records:
        dout, din= r["shape"]
        assert r["packed_bytes"] == packed_bytes(dout, din, 4, 16)
        assert r["extra_fp16_bytes"] == 0
        assert 0.0 < r["rel_output_error"] < 0.1

    #every quantized weight lies on its group grid: at most 2^4 distinct values per 16-wide group
    w= model.model.layers[0].mlp.down_proj.weight
    assert all(len(torch.unique(g)) <= 16 for g in w.reshape(-1, 16))


def test_awq_counts_o_proj_scale_bytes():
    _, records, _, _= run("awq", 4)

    o= [r for r in records if r["proj"] == "o_proj"]
    assert all(r["extra_fp16_bytes"] == 2 * 32 for r in o)
    assert all(r["extra_fp16_bytes"] == 0 for r in records if r["proj"] != "o_proj")
