import re

import torch
import torch.nn.functional as F


def preprocess(text):
    #copied from lm-evaluation-harness 0.4.13, tasks/hellaswag/utils.py
    text= text.strip()
    text= text.replace(" [title]", ". ")
    text= re.sub("\\[.*?\\]", "", text)
    text= text.replace("  ", " ")

    return text


def build_query_and_choices(example):
    #lm-eval hellaswag v1.0 process_docs
    ctx= example["ctx_a"] + " " + example["ctx_b"].capitalize()

    query= preprocess(example["activity_label"] + ": " + ctx)
    choices= [preprocess(ending) for ending in example["endings"]]
    gold= int(example["label"])

    return query, choices, gold


def pick_answers(lls, choices):
    #acc uses the summed log-likelihood; acc_norm divides by the character length of the choice
    #(lm-eval scores " " + choice but normalises by len(choice))
    norm= [ll / len(choice) for ll, choice in zip(lls, choices)]

    pred= max(range(len(lls)), key=lambda i: lls[i])
    pred_norm= max(range(len(norm)), key=lambda i: norm[i])

    return pred, pred_norm


def encode_pair(tok, context, continuation):
    #lm-eval _encode_pair (causal): trailing context spaces move to the continuation,
    #then the whole string is encoded and split at the context length
    n_spaces= len(context) - len(context.rstrip())

    if n_spaces > 0:
        continuation= context[-n_spaces:] + continuation
        context= context[:-n_spaces]

    whole= tok.encode(context + continuation, add_special_tokens=False)
    context_enc= tok.encode(context, add_special_tokens=False)

    return context_enc, whole[len(context_enc):]


@torch.no_grad()
def score_choices(model, tok, query, choices, device):
    #sum log p(continuation | context) for each " " + choice, the 4 endings batched with right padding
    pairs= [encode_pair(tok, query, " " + choice) for choice in choices]
    seqs= [ctx + cont for ctx, cont in pairs]
    max_len= max(len(s) for s in seqs) - 1

    input_ids= torch.full((len(seqs), max_len), tok.pad_token_id, dtype=torch.long)
    attention_mask= torch.zeros((len(seqs), max_len), dtype=torch.long)

    for i, s in enumerate(seqs):
        input_ids[i, :len(s) - 1]= torch.tensor(s[:-1])
        attention_mask[i, :len(s) - 1]= 1

    logits= model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device)).logits
    assert torch.isfinite(logits).all()
    logp= F.log_softmax(logits.float(), dim=-1)

    lls= []
    n_tokens= []

    for i, (ctx, cont) in enumerate(pairs):
        #continuation token j sits at position len(ctx)+j and is predicted from position len(ctx)+j-1
        positions= torch.arange(len(ctx) - 1, len(ctx) - 1 + len(cont), device=device)
        targets= torch.tensor(cont, device=device)

        lls.append(logp[i, positions, targets].sum().item())
        n_tokens.append(len(cont))

    return lls, n_tokens


def evaluate_hellaswag(model, tok, rows, n_examples, device):
    #first n_examples by the locked order (2,000 full or 512 diagnostic, decision 1.1)
    assert tok.encode("a", add_special_tokens=True) == tok.encode("a", add_special_tokens=False)

    out= []

    for k, row in enumerate(rows[:n_examples]):
        query, choices, gold= build_query_and_choices(row["example"])
        lls, n_tokens= score_choices(model, tok, query, choices, device)
        pred, pred_norm= pick_answers(lls, choices)

        out.append({
            "order": row["order"],
            "source_row_index": row["source_row_index"],
            "gold": gold,
            "lls": lls,
            "choice_chars": [len(c) for c in choices],
            "choice_tokens": n_tokens,
            "pred": pred,
            "pred_norm": pred_norm
        })

        if (k + 1) % 250 == 0:
            print(f"hellaswag {k + 1}/{n_examples}")

    acc= sum(r["pred"] == r["gold"] for r in out) / len(out)
    acc_norm= sum(r["pred_norm"] == r["gold"] for r in out) / len(out)

    return out, {"n_examples": len(out), "acc": acc, "acc_norm": acc_norm}
