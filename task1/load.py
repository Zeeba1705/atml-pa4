import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM

from common.io_utils import read_jsonl, model_revision

ELIGIBLE_ATTN= ["q_proj", "k_proj", "v_proj", "o_proj"]
ELIGIBLE_MLP= ["gate_proj", "up_proj", "down_proj"]
QWEN_VOCAB= 151936   #embedding rows of qwen2.5 (tokenizer has fewer, rest is padding)


def eligible_linears(model):
    linears= []

    for i, layer in enumerate(model.model.layers):
        for name in ELIGIBLE_ATTN:
            linears.append((f"model.layers.{i}.self_attn.{name}", getattr(layer.self_attn, name)))

        for name in ELIGIBLE_MLP:
            linears.append((f"model.layers.{i}.mlp.{name}", getattr(layer.mlp, name)))

    return linears


def eligible_weight_count(model):
    return sum(linear.weight.numel() for _, linear in eligible_linears(model))


def unique_param_count(model):
    #tied lm_head/embedding share storage, so count each storage once
    seen= set()
    total= 0

    for _, p in model.named_parameters(remove_duplicate=False):
        key= (p.data_ptr(), p.numel())

        if key in seen:
            continue

        seen.add(key)
        total+= p.numel()

    return total


def load_tokenizer(cfg):
    #smoke runs use the real tokenizer too
    return AutoTokenizer.from_pretrained(cfg["model"], revision=model_revision(cfg, "task1_base"))


def load_model(cfg, device, smoke):
    if smoke:
        #tiny random-init qwen2 with the real vocab, fp32 on cpu
        torch.manual_seed(cfg["seed"])

        config= Qwen2Config(
            vocab_size=QWEN_VOCAB,
            hidden_size=cfg["smoke"]["hidden_size"],
            intermediate_size=cfg["smoke"]["intermediate_size"],
            num_hidden_layers=cfg["smoke"]["num_hidden_layers"],
            num_attention_heads=cfg["smoke"]["num_attention_heads"],
            num_key_value_heads=cfg["smoke"]["num_key_value_heads"],
            tie_word_embeddings=True,
            attn_implementation=cfg["attn_implementation"]
        )

        model= Qwen2ForCausalLM(config).float()
    else:
        model= AutoModelForCausalLM.from_pretrained(
            cfg["model"],
            revision=model_revision(cfg, "task1_base"),
            dtype=torch.float16,
            attn_implementation=cfg["attn_implementation"]
        )

    model= model.to(device).eval()

    #eligible = exactly 7 projections per block (p.3)
    assert len(eligible_linears(model)) == 7 * model.config.num_hidden_layers

    return model


def data_path(cfg, data_root, key):
    path= os.path.join(data_root, cfg["data"][key])
    assert os.path.exists(path), f"missing {path}, run setup_data.py --public (README)"

    return path


def load_windows(cfg, data_root, suite):
    windows= read_jsonl(data_path(cfg, data_root, suite))

    #exactly as materialized: the scored-token total is fixed by the handout
    assert sum(w["scored_tokens"] for w in windows) == cfg["ppl_scored_tokens"]

    for w in windows:
        start, end= w["score_target_offsets"]
        assert end - start == w["scored_tokens"] and start >= 1 and end <= len(w["token_ids"])

    return windows


def load_sequences(cfg, data_root, key, n_expected):
    seqs= read_jsonl(data_path(cfg, data_root, key))

    assert len(seqs) == n_expected
    assert all(len(s["token_ids"]) == cfg["sequence_length"] for s in seqs)

    return seqs


def load_hellaswag(cfg, data_root):
    rows= read_jsonl(data_path(cfg, data_root, "hellaswag"))
    rows= sorted(rows, key=lambda r: r["order"])

    #must match the course manifest (same ids, same order)
    manifest= read_jsonl(os.path.join(data_root, "data", "public_manifests", "hellaswag_val_ids.jsonl"))
    assert [r["source_row_index"] for r in rows] == [m["row_index"] for m in manifest]
    assert len(rows) == cfg["hellaswag"]["n_full"]

    return rows
