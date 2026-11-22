import math

import torch
import torch.nn.functional as F


def window_nll(logits, token_ids, score_start, score_end):
    #target at offset t is predicted by the logits at position t-1
    logp= F.log_softmax(logits[score_start - 1:score_end - 1].float(), dim=-1)
    targets= token_ids[score_start:score_end]

    nll= -logp.gather(1, targets[:, None]).sum().item()

    return nll, score_end - score_start


def pooled_ppl(nll_sums, counts):
    #pool over tokens (not a mean of per-window ppl)
    n= sum(counts)
    mean_nll= sum(nll_sums) / n

    return n, mean_nll, math.exp(mean_nll)
