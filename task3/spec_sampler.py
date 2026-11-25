#task 3 exact speculative sampling (handout p.11, p.13 step 1; leviathan et al. 2023)
#one loop for every model: a runner has feed(tokens, keep) -> next-token probabilities after each fed token,
#crop(n) -> keep the first n cached positions, length() -> cached positions. HFRunner wraps a hf model + DynamicCache;
#task3/toy_markov.py has a markov runner, so the chi-square test runs exactly this code
import time

import torch
from transformers import DynamicCache


def sample(probs, gen):
    return int(torch.multinomial(probs.to(gen.device), 1, generator=gen))


def accept_prob(p, q, x):
    #a(x|h) = min(1, p(x|h) / q(x|h)); q(x) > 0 because x was drawn from q
    return torch.clamp(p[x] / q[x], max=1.0)


def positive_residual(p, q):
    #r(y) = [p(y) - q(y)]+ / sum_z [p(z) - q(z)]+ ; if p == q no rejection can happen, return p
    r= torch.clamp(p - q, min=0.0)
    s= r.sum()

    if s <= 0:
        return p

    return r / s


def verify_round(p_rows, q_rows, drafts, gen, residual_fn=positive_residual):
    #p_rows: (g+1, V) target probs at each draft position plus the one after; q_rows: (g, V) draft probs
    #accept left to right; first rejection -> one residual token; all accepted -> one bonus token from p_rows[g]
    for i, x in enumerate(drafts):
        u= torch.rand((), generator=gen, device=gen.device)

        if u < accept_prob(p_rows[i], q_rows[i], x).to(gen.device):
            continue

        y= sample(residual_fn(p_rows[i], q_rows[i]), gen)

        return drafts[:i] + [y], {"accepted": i, "examined": i + 1, "bonus": False}

    y= sample(p_rows[len(drafts)], gen)

    return drafts + [y], {"accepted": len(drafts), "examined": len(drafts), "bonus": True}


def clock(timer):
    #synchronized timestamp when timing per phase; no sync (and no cost) when timer is None
    if timer is None:
        return 0.0

    if timer["device"].type == "cuda":
        torch.cuda.synchronize(timer["device"])

    return time.perf_counter()


def add_time(timer, name, start):
    if timer is not None:
        now= clock(timer)
        timer["phases"][name]= timer["phases"].get(name, 0.0) + now - start
        return now

    return 0.0


def cut_at_eos(new, eos_ids):
    for i, t in enumerate(new):
        if t in eos_ids:
            return new[:i + 1], True

    return new, False


def prefill(runners, prompt, timer):
    #everything but the last prompt token goes into the caches; the last one is fed with the first round
    t= clock(timer)

    if len(prompt) > 1:
        for r in runners:
            r.feed(prompt[:-1], keep=1)

    add_time(timer, "prefill", t)


def speculative_generate(target, draft, prompt, gamma, max_new, eos_ids, gen, residual_fn=positive_residual, timer=None):
    tokens= list(prompt)
    out= []
    rounds= []
    stop= "length"
    target_passes= 0
    draft_passes= 0

    prefill([target, draft], prompt, timer)

    while len(out) < max_new:
        #accepted + residual/bonus never exceed the cap; with one token left this is a plain target step
        g= min(gamma, max_new - len(out) - 1)
        t= clock(timer)

        drafts= []
        q_rows= []
        pending= tokens[draft.length():]

        for _ in range(g):
            q= draft.feed(pending, keep=1)[-1]
            x= sample(q, gen)
            drafts.append(x)
            q_rows.append(q)
            pending= [x]
            draft_passes+= 1

        t= add_time(timer, "draft", t)

        #one target pass over the uncached committed tokens plus all proposals
        p_rows= target.feed(tokens[target.length():] + drafts, keep=g + 1)
        target_passes+= 1
        t= add_time(timer, "verify", t)

        if g > 0:
            new, info= verify_round(p_rows, torch.stack(q_rows), drafts, gen, residual_fn)
        else:
            new, info= [sample(p_rows[-1], gen)], {"accepted": 0, "examined": 0, "bonus": False}

        new, hit_eos= cut_at_eos(new, eos_ids)
        t= add_time(timer, "accept", t)

        tokens+= new
        out+= new

        #rollback/advance: cached positions beyond the last committed-and-verified token are stale
        keep= len(tokens) - 1
        target.crop(min(target.length(), keep))
        draft.crop(min(draft.length(), keep))
        add_time(timer, "cache", t)

        rounds.append({"proposed": g, "emitted": len(new), **info})

        if hit_eos:
            stop= "eos"
            break

    return out, {
        "rounds": rounds,
        "stop": stop,
        "n_out": len(out),
        "target_passes": target_passes,
        "draft_passes": draft_passes,
        "proposed": sum(r["proposed"] for r in rounds),
        "accepted": sum(r["accepted"] for r in rounds),
        "examined": sum(r["examined"] for r in rounds)
    }


def target_generate(target, prompt, max_new, eos_ids, gen, timer=None):
    #target-only reference: one target pass per output token, same prefill and sampling
    tokens= list(prompt)
    out= []
    stop= "length"

    prefill([target], prompt, timer)

    while len(out) < max_new:
        t= clock(timer)
        p= target.feed(tokens[target.length():], keep=1)[-1]
        t= add_time(timer, "verify", t)

        y= sample(p, gen)
        tokens.append(y)
        out.append(y)
        add_time(timer, "accept", t)

        if y in eos_ids:
            stop= "eos"
            break

    return out, {"stop": stop, "n_out": len(out), "target_passes": len(out), "draft_passes": 0}


class HFRunner:
    #hf causal lm + DynamicCache; positions come from the cache length (batch 1, no padding)
    def __init__(self, model, device, temperature=1.0):
        self.model= model
        self.device= device
        self.temperature= temperature
        self.cache= DynamicCache()

    def length(self):
        return self.cache.get_seq_length()

    def feed(self, tokens, keep=None):
        #keep = number of final positions whose logits are needed (None = all)
        ids= torch.tensor([tokens], device=self.device)

        with torch.no_grad():
            out= self.model(input_ids=ids, past_key_values=self.cache, use_cache=True, logits_to_keep=keep or 0)

        self.cache= out.past_key_values

        return torch.softmax(out.logits[0].float() / self.temperature, dim=-1)

    def crop(self, n):
        self.cache.crop(n)
