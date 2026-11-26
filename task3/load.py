#task 3 models, tokenizer checks and prompt sets (handout p.12)
#prompt format = setup_data.py's banks: the raw question/instruction as the single user message, default chat
#template, add_generation_prompt (decision 3.2)
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM

from common.io_utils import read_jsonl, model_revision

QWEN_VOCAB= 151936
CHECK_STRINGS= ["Example 123", "Hello there", "A complicated example with 𝜋", "Janet has 3 apples and buys 2 more."]


def load_tokenizer(cfg, key):
    return AutoTokenizer.from_pretrained(cfg["models"][key], revision=model_revision(cfg, key))


def load_model(cfg, key, device, smoke, seed=None):
    if smoke:
        #tiny random-init qwen2 with the real vocab; target and draft differ by seed
        torch.manual_seed(seed if seed is not None else (1 if key == "target" else 2))
        s= cfg["smoke"]
        config= Qwen2Config(
            vocab_size=QWEN_VOCAB,
            hidden_size=s["hidden_size"],
            intermediate_size=s["intermediate_size"],
            num_hidden_layers=s["num_hidden_layers"],
            num_attention_heads=s["num_attention_heads"],
            num_key_value_heads=s["num_key_value_heads"],
            tie_word_embeddings=True,
            attn_implementation="sdpa"
        )

        return Qwen2ForCausalLM(config).float().to(device).eval()

    model= AutoModelForCausalLM.from_pretrained(
        cfg["models"][key],
        revision=model_revision(cfg, key),
        torch_dtype=torch.float16,
        attn_implementation=cfg["attn_implementation"]
    )

    return model.to(device).eval()


def prompt_ids(tok, text):
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True)


def check_agreement(tok_t, tok_d, target, draft, prompts):
    #same token ids, same chat-template output on a fixed prompt set, same logit vocabulary (p.12)
    for s in CHECK_STRINGS:
        assert tok_t.encode(s) == tok_d.encode(s), s

    for p in prompts:
        assert prompt_ids(tok_t, p["text"]) == prompt_ids(tok_d, p["text"]), p["id"]

    vt= target.get_output_embeddings().weight.shape[0]
    vd= draft.get_output_embeddings().weight.shape[0]
    assert vt == vd == QWEN_VOCAB, (vt, vd)

    return {"strings_checked": len(CHECK_STRINGS), "prompts_checked": len(prompts), "logit_vocab": vt,
            "eos_token_ids": sorted({tok_t.convert_tokens_to_ids("<|im_end|>"), tok_t.convert_tokens_to_ids("<|endoftext|>")})}


def data_path(cfg, data_root, key):
    return os.path.join(data_root, cfg["data"][key])


def eval_prompts(cfg, data_root):
    #120 held-out prompts: 60 gsm8k test (ids as in the supplied banks) + 30 familiar + 30 shifted short-form
    out= []

    for r in read_jsonl(data_path(cfg, data_root, "gsm8k_test")):
        out.append({"id": f"GSM8K-{r['order']:04d}", "text": r["example"]["question"], "domain": "math", "split": "math",
                    "answer": r["example"]["answer"].split("####")[-1].strip()})

    for r in read_jsonl(data_path(cfg, data_root, "short_test")):
        out.append({"id": r["id"], "text": r["input"], "domain": "short", "split": r["split"], "family": r["family"], "answer": r["answer"]})

    assert len(out) == cfg["num_eval_prompts"]
    assert sum(p["split"] == "math" for p in out) == 60 and sum(p["split"] == "familiar" for p in out) == 30

    return out


def train_prompts(cfg, data_root):
    #800 training prompts: 500 gsm8k train + 300 short-form train (families 0-2); never test prompts
    out= []

    for r in read_jsonl(data_path(cfg, data_root, "gsm8k_train")):
        out.append({"id": f"GSM8K-train-{r['order']:04d}", "text": r["example"]["question"], "category": "math"})

    for r in read_jsonl(data_path(cfg, data_root, "short_train")):
        assert r["split"] == "train"
        out.append({"id": r["id"], "text": r["input"], "category": "short"})

    assert len(out) == 800
    test_texts= {p["text"] for p in eval_prompts(cfg, data_root)}
    assert not any(p["text"] in test_texts for p in out), "a held-out prompt is in the training pool"

    return out
