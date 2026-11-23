import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from task1.calibrate import sequential_compress
from task1.prune import make_pruner


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


def test_block_by_block_matches_full_forward_without_compression():
    model= tiny_model()
    seqs= sequences()
    ids= torch.tensor([s["token_ids"] for s in seqs])

    with torch.no_grad():
        ref= model(ids, output_hidden_states=True).hidden_states[-1]

    captured= {}

    def identity(idx, linears, stats):
        captured[idx]= {name: s["n"] for name, s in stats.items()}
        return {}

    records, _= sequential_compress(model, seqs, torch.device("cpu"), batch_size=4, compress_fn=identity)

    #every projection saw every calibration token, and nothing changed
    assert all(n == 6 * 16 for layer in captured.values() for n in layer.values())
    assert all(r["rel_output_error"] == 0.0 for r in records)
    assert len(records) == 2 * 7

    with torch.no_grad():
        after= model(ids, output_hidden_states=True).hidden_states[-1]

    assert torch.equal(ref, after)


def test_sequential_block_outputs_feed_next_block():
    #the inputs seen by block 1 must be block 0's outputs (here: a full forward's hidden state 1)
    model= tiny_model()
    seqs= sequences()
    ids= torch.tensor([s["token_ids"] for s in seqs])

    with torch.no_grad():
        h1= model(ids, output_hidden_states=True).hidden_states[1]

    seen= {}

    def record(idx, linears, stats):
        if idx == 1:
            seen["sumsq"]= stats["q_proj"]["sumsq"]
        return {}

    sequential_compress(model, seqs, torch.device("cpu"), batch_size=6, compress_fn=record)

    #q_proj input = input_layernorm(h1)
    x= model.model.layers[1].input_layernorm(h1).reshape(-1, 32)
    assert torch.allclose(seen["sumsq"], x.pow(2).sum(dim=0), rtol=1e-4)


def test_wanda_pruning_hits_exact_sparsity_and_reports_error():
    model= tiny_model()

    records, _= sequential_compress(model, sequences(), torch.device("cpu"), batch_size=4, compress_fn=make_pruner("wanda", 0.5))

    for layer in model.model.layers:
        for linear in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.mlp.down_proj]:
            assert ((linear.weight == 0).sum(dim=1) == linear.weight.shape[1] // 2).all()

    assert all(0.0 < r["rel_output_error"] < 1.0 for r in records)
    assert all(abs(r["nonzero_fraction"] - 0.5) < 1e-6 for r in records)
    assert all("score_quantiles" in r for r in records)


def test_two_four_pruning_pattern():
    model= tiny_model()

    sequential_compress(model, sequences(), torch.device("cpu"), batch_size=4, compress_fn=make_pruner("wanda", 0.5, two_four=True))

    w= model.model.layers[0].mlp.up_proj.weight
    assert ((w.reshape(w.shape[0], -1, 4) != 0).sum(dim=2) == 2).all()
