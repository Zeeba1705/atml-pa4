#task 2 part C retention and precision policies at the case-specific int4 budget (handout pp.8-10; decisions 2.6, 2.7,
#2.9, 2.10, 2.21). the budget is B = M_4(L) at the pre-question cache. no policy ever sees the answer code or the
#evidence span: question-aware scores come only from the question-suffix queries
import torch
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from task2.kv_quant import GROUP, META_BYTES, mb_bytes, fp16_bytes, fake_quant, fake_quant_span
from task2.kv_cache import forward, gather_cache

METHODS= ["fp16", "evict_sink_recent", "evict_question", "int4_token", "int4_channel_k", "hybrid_int2"]
INDEX_BYTES= 2          #int16 position per retained (layer, kv head) entry
RANGE_BYTES= 4          #sink+recent is one shared rule: two int16 counts describe it


def shape_of(model):
    c= model.config

    return c.num_hidden_layers, c.num_key_value_heads, getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads


def evict_counts(L, budget, nl, hkv, dh, sink, recent):
    #largest fp16 position counts whose actual bytes fit the budget
    per_pos= 2 * 2 * dh * nl * hkv                  #k and v, fp16, every layer and head
    n_shared= min(L, (budget - RANGE_BYTES) // per_pos)
    n_head= min(L, budget // (per_pos + INDEX_BYTES * nl * hkv))
    assert n_shared >= sink + recent and n_head >= sink + recent, (L, budget)

    return {"evict_sink_recent": n_shared, "evict_question": n_head}


def evict_bytes(method, n, nl, hkv, dh):
    per_pos= 2 * 2 * dh * nl * hkv

    if method == "evict_sink_recent":
        return n * per_pos + RANGE_BYTES

    return n * per_pos + n * INDEX_BYTES * nl * hkv


def hybrid_layout(L, extra, sink, recent):
    #fp16 = sink + recent + `extra` question-selected positions + the incomplete low-bit residual; low = 64-token groups
    low= L - sink - recent - extra
    groups= low // GROUP
    fp16= L - groups * GROUP

    return fp16, groups


def hybrid_bytes(L, extra, bits, nl, hkv, dh, sink, recent):
    fp16, groups= hybrid_layout(L, extra, sink, recent)
    per_head= 2 * (fp16 * dh * 2) + 2 * (groups * dh * (GROUP * bits // 8 + META_BYTES)) + fp16 * INDEX_BYTES

    return nl * hkv * per_head


def hybrid_extra(L, budget, bits, nl, hkv, dh, sink, recent):
    #largest number of question-selected fp16 positions within the budget
    for extra in range(L - sink - recent, -1, -1):
        if hybrid_bytes(L, extra, bits, nl, hkv, dh, sink, recent) <= budget:
            return extra

    raise ValueError(f"hybrid does not fit: L={L}, budget={budget}")


def sink_recent_index(L, n, sink, nl, hkv, device):
    #first `sink` positions + the most recent n - sink positions, shared by every layer and head
    idx= torch.cat([torch.arange(sink), torch.arange(L - (n - sink), L)]).to(device)

    return idx[None, None].expand(nl, hkv, -1).contiguous()


def question_scores(model, cache, suffix, L):
    #summed fp32 softmax mass from every question-suffix query (6 grouped q heads per kv head) on each context position.
    #suffix pass over the uncompressed cache (sdpa model, unchanged); q projections captured with hooks and rotated with
    #the same rope; the cache is cropped back to L afterwards
    queries= {}
    hooks= []

    def grab(i):
        def hook(module, args, kwargs):
            h= kwargs["hidden_states"]
            cos, sin= kwargs["position_embeddings"]
            q= module.q_proj(h).view(*h.shape[:-1], -1, module.head_dim).transpose(1, 2)
            queries[i]= apply_rotary_pos_emb(q, q, cos, sin)[0]

        return hook

    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.self_attn.register_forward_pre_hook(grab(i), with_kwargs=True))

    try:
        forward(model, cache, suffix, L)
    finally:
        for h in hooks:
            h.remove()

    nl, hkv, dh= shape_of(model)
    q_len= len(suffix)
    groups= model.config.num_attention_heads // hkv
    visible= torch.arange(L + q_len, device=model.device)[None, :] <= (L + torch.arange(q_len, device=model.device))[:, None]
    scores= []

    for i, layer in enumerate(cache.layers):
        k= layer.keys[0].float().repeat_interleave(groups, dim=0)            #[12, L + q, dh]
        s= queries[i][0].float() @ k.transpose(-1, -2) * dh ** -0.5        #[12, q, L + q]
        p= torch.softmax(s.masked_fill(~visible, float("-inf")), dim=-1)
        scores.append(p[:, :, :L].sum(1).view(hkv, groups, L).sum(1))

    cache.crop(L)

    return torch.stack(scores)                                                #[nl, hkv, L]


def question_index(scores, n, sink, recent):
    #sinks + recent always kept; the rest of the n slots by score, independently per (layer, kv head); ascending order
    nl, hkv, L= scores.shape
    s= scores.clone()
    s[..., :sink]= float("inf")
    s[..., L - recent:]= float("inf")

    return s.topk(n, dim=-1).indices.sort(dim=-1).values


class Policy:
    #one cache configuration; compress() edits the pre-question cache, after_append() handles appended tokens
    def __init__(self, method, L, budget, model, cfg):
        self.method= method
        self.L= L
        self.nl, self.hkv, self.dh= shape_of(model)
        self.sink= cfg["eviction"]["sink_tokens"]
        self.recent= cfg["eviction"]["recent_tokens"]
        self.bits= cfg["quant"]["budget_bits"]
        self.low_bits= cfg["quant"]["hybrid_low_bits"]
        self.budget= budget
        self.qupto= None          #quantized prefix length for the int4 layouts
        self.info= {}

    def compress(self, cache, scores):
        m= self.method
        n16= fp16_bytes(self.L, self.nl, self.hkv, self.dh)

        if m == "fp16":
            self.info= {"bytes": n16, "fp16_positions": self.L}
            return cache

        if m.startswith("evict"):
            n= evict_counts(self.L, self.budget, self.nl, self.hkv, self.dh, self.sink, self.recent)[m]
            device= cache.layers[0].keys.device

            if m == "evict_sink_recent":
                idx= sink_recent_index(self.L, n, self.sink, self.nl, self.hkv, device)
            else:
                idx= question_index(scores, n, self.sink, self.recent)

            self.index= idx
            self.info= {"bytes": evict_bytes(m, n, self.nl, self.hkv, self.dh), "fp16_positions": n,
                        "retention_granularity": "shared" if m == "evict_sink_recent" else "per_layer_kv_head"}
            return gather_cache(cache, idx)

        if m.startswith("int4"):
            layout_k= "token" if m == "int4_token" else "channel"
            self.layouts= (layout_k, "token")
            self.qupto= (self.L // GROUP) * GROUP

            for layer in cache.layers:
                fake_quant_span(layer.keys, 0, self.qupto, self.bits, layout_k)
                fake_quant_span(layer.values, 0, self.qupto, self.bits, "token")

            self.info= {"bytes": mb_bytes(self.L, self.bits, self.nl, self.hkv, self.dh), "fp16_positions": self.L - self.qupto,
                        "layout": f"{layout_k}-wise K / token-wise V"}
            return cache

        assert m == "hybrid_int2", m
        extra= hybrid_extra(self.L, self.budget, self.low_bits, self.nl, self.hkv, self.dh, self.sink, self.recent)
        fp16, groups= hybrid_layout(self.L, extra, self.sink, self.recent)
        keep= question_index(scores, self.sink + self.recent + extra, self.sink, self.recent)      #[nl, hkv, n]
        is16= torch.zeros(scores.shape, dtype=torch.bool, device=scores.device).scatter_(-1, keep, True)
        low= (~is16).float().argsort(dim=-1, descending=True, stable=True)[..., :self.L - keep.shape[-1]].sort(-1).values
        grouped= low[..., :groups * GROUP]                       #ascending; the last (low mod 64) stay fp16 residual
        is16.scatter_(-1, low[..., groups * GROUP:], True)

        for i, layer in enumerate(cache.layers):
            ix= grouped[i][None, :, :, None].expand(1, -1, -1, self.dh)
            for t, layout in [(layer.keys, "channel"), (layer.values, "token")]:
                t.scatter_(2, ix, fake_quant(t.gather(2, ix).contiguous(), self.low_bits, layout))

        self.fp16_mask= is16
        self.info= {"bytes": hybrid_bytes(self.L, extra, self.low_bits, self.nl, self.hkv, self.dh, self.sink, self.recent),
                    "fp16_positions": fp16, "question_selected_extra": extra, "low_bit_groups_per_channel": groups,
                    "low_bits": self.low_bits, "retention_granularity": "per_layer_kv_head"}
        return cache

    def after_append(self, cache):
        #int4 layouts: appended tokens stay fp16 until the trailing block completes a 64-token group (decision 2.10)
        if self.qupto is None:
            return

        while cache.get_seq_length() - self.qupto >= GROUP:
            for layer in cache.layers:
                fake_quant_span(layer.keys, self.qupto, self.qupto + GROUP, self.bits, self.layouts[0])
                fake_quant_span(layer.values, self.qupto, self.qupto + GROUP, self.bits, self.layouts[1])
            self.qupto+= GROUP

    def live_bytes(self, cache):
        #actual live cache bytes now: compressed context + appended tokens (fp16, or int4 groups once complete)
        appended= cache.get_seq_length() - (self.info["fp16_positions"] if self.method.startswith("evict") else self.L)
        shared= fp16_bytes(appended, self.nl, self.hkv, self.dh)

        if self.qupto is not None:
            return mb_bytes(cache.get_seq_length(), self.bits, self.nl, self.hkv, self.dh)

        return self.info["bytes"] + shared

    def evidence_retained(self, evidence):
        #evaluation only: fraction of gold evidence tokens kept in fp16, averaged over (layer, kv head)
        ev= torch.tensor(evidence)

        if self.method.startswith("evict"):
            kept= torch.isin(self.index.cpu(), ev).sum(-1).float() / len(evidence)
        elif self.method == "hybrid_int2":
            kept= self.fp16_mask[..., ev].float().mean(-1).cpu()
        else:
            return None

        return {"mean": float(kept.mean()), "min": float(kept.min()), "all_heads_all_tokens": float((kept == 1).float().mean())}
