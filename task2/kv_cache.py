#task 2 part C cache mechanics (handout p.9 C1; decisions 2.6-2.8). batch 1, no padding.
#prompt = user_text as the single user message, default chat template. split at the token where QUESTION_BEGIN
#starts: context = ids[:L] (compressed), suffix = ids[L:] (always fp16, shared). after editing, cached keys keep their
#post-rope rotation; new tokens get their true absolute position_ids and the physical cache_position
import torch
from transformers import DynamicCache


def encode_case(tok, case, marker):
    #returns ids, L (first suffix token) and the evidence token positions (evaluation only, never given to a policy)
    text= tok.apply_chat_template([{"role": "user", "content": case["user_text"]}], tokenize=False, add_generation_prompt=True)
    enc= tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets= enc["input_ids"], enc["offset_mapping"]
    assert text.count(marker) == 1 and case["user_text"].count(marker) == 1, case["id"]

    m= text.index(marker)
    L= next(i for i, (s, e) in enumerate(offsets) if e > m)
    base= text.index(case["user_text"])
    es, ee= base + case["evidence_char_start"], base + case["evidence_char_end"]
    evidence= [i for i, (s, e) in enumerate(offsets) if s < ee and e > es]
    assert evidence and max(evidence) < L, case["id"]

    return {"ids": ids, "L": L, "evidence": evidence, "text": text}


def forward(model, cache, tokens, abs_start, keep=1, mask=None):
    #append tokens at true absolute positions abs_start.. and physical positions cache_len..; returns last `keep` logit rows
    device= model.device
    n= len(tokens)
    phys= cache.get_seq_length()

    with torch.no_grad():
        out= model(
            input_ids=torch.tensor([tokens], device=device),
            position_ids=torch.arange(abs_start, abs_start + n, device=device).unsqueeze(0),
            cache_position=torch.arange(phys, phys + n, device=device),
            past_key_values=cache, use_cache=True, logits_to_keep=keep, attention_mask=mask
        )

    return out.logits[0].float()


def prefill(model, ids):
    cache= DynamicCache()
    logits= forward(model, cache, ids, 0)

    return cache, logits


def clone_cache(cache):
    new= DynamicCache()

    for i, layer in enumerate(cache.layers):
        new.update(layer.keys.clone(), layer.values.clone(), i)

    return new


def gather_cache(cache, idx):
    #idx [nl, hkv, n] ascending positions per (layer, kv head); same count n everywhere
    new= DynamicCache()

    for i, layer in enumerate(cache.layers):
        ix= idx[i].to(layer.keys.device)[None, :, :, None].expand(1, -1, -1, layer.keys.shape[-1])
        new.update(layer.keys.gather(2, ix), layer.values.gather(2, ix), i)

    return new


def greedy_decode(model, cache, first_logits, abs_next, max_new, eos_ids, after_append=None):
    #first_logits = row predicting the first new token; abs_next = its true absolute position
    out= []
    logits= first_logits

    for t in range(max_new):
        y= int(logits.argmax())
        out.append(y)

        if y in eos_ids or t == max_new - 1:
            break

        logits= forward(model, cache, [y], abs_next + t)[-1]

        if after_append is not None:
            after_append(cache)

    return out
