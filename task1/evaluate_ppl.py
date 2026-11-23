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


@torch.no_grad()
def evaluate_ppl(model, windows, device):
    #each window is scored independently, batch 1, no padding
    rows= []

    for i, w in enumerate(windows):
        token_ids= torch.tensor(w["token_ids"], device=device)
        start, end= w["score_target_offsets"]

        logits= model(token_ids[None, :end]).logits[0]
        assert torch.isfinite(logits).all(), f"non-finite logits in window {i}"

        nll, n= window_nll(logits, token_ids, start, end)

        rows.append({
            "window": i,
            "source_row": w["source_row"],
            "token_offset": w["token_offset"],
            "nll_sum": nll,
            "n_scored": n
        })

        if (i + 1) % 100 == 0:
            print(f"ppl window {i + 1}/{len(windows)}")

    n, mean_nll, ppl= pooled_ppl([r["nll_sum"] for r in rows], [r["n_scored"] for r in rows])

    return rows, {"n_scored": n, "n_windows": len(rows), "mean_nll": mean_nll, "ppl": ppl}
