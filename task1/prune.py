import torch


def input_norms(x):
    #x is (tokens, din) -> l2 norm of each input feature over calibration tokens
    return x.float().pow(2).sum(dim=0).sqrt()


def magnitude_scores(w):
    return w.float().abs()


def wanda_scores(w, x_norm):
    return w.float().abs() * x_norm.float()[None, :]


def row_prune_mask(scores, n_prune):
    #n_prune is an int (same for every row) or a tensor with one count per row
    dout, din= scores.shape

    order= torch.argsort(scores, dim=1, stable=True)
    ranks= torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(din, device=scores.device).expand(dout, din))

    if not torch.is_tensor(n_prune):
        n_prune= torch.full((dout,), int(n_prune), device=scores.device)

    keep= ranks >= n_prune.to(scores.device)[:, None]

    return keep


def two_four_mask(scores):
    dout, din= scores.shape
    assert din % 4 == 0

    groups= scores.reshape(dout, din // 4, 4)
    top= groups.topk(2, dim=2).indices

    keep= torch.zeros_like(groups, dtype=torch.bool)
    keep.scatter_(2, top, True)

    return keep.reshape(dout, din)


SCORE_QUANTILES= [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]


def score_summary(scores):
    #per-projection score distribution (p.4 step 3); torch.quantile is capped at 2^24 elements
    flat= scores.flatten().float()

    if flat.numel() > 2 ** 24:
        flat= flat[torch.randperm(flat.numel(), generator=torch.Generator().manual_seed(0))[:2 ** 24].to(flat.device)]

    q= torch.quantile(flat, torch.tensor(SCORE_QUANTILES, device=flat.device))

    return {"score_quantiles": dict(zip([str(x) for x in SCORE_QUANTILES], q.tolist())), "score_mean": flat.mean().item()}


def make_pruner(method, sparsity, two_four=False, row_zeros=None):
    #returns a compress_fn for calibrate.sequential_compress
    #row_zeros (p3): one {proj: [zeros per output row]} dict per block; the wanda ranking within each row is unchanged
    def compress(idx, linears, stats, layer):
        infos= {}

        for name, linear in linears:
            w= linear.weight

            if method == "magnitude":
                scores= magnitude_scores(w)
            elif method == "wanda":
                scores= wanda_scores(w, stats[name]["sumsq"].sqrt())
            else:
                raise ValueError(method)

            if two_four:
                assert sparsity == 0.5
                keep= two_four_mask(scores)
            elif row_zeros is not None:
                counts= torch.tensor(row_zeros[idx][name], device=w.device)
                assert counts.numel() == w.shape[0]
                keep= row_prune_mask(scores, counts)
                assert (~keep).sum().item() == counts.sum().item()
            else:
                n_prune= int(round(sparsity * w.shape[1]))
                assert n_prune == sparsity * w.shape[1], "per-row count must be exact"
                keep= row_prune_mask(scores, n_prune)

            kept_score= scores[keep].sum().item()
            w.mul_(keep.to(w.dtype))

            infos[name]= {
                "shape": list(w.shape),
                "nonzero_fraction": (w != 0).float().mean().item(),
                "pruned_score_fraction": 1.0 - kept_score / scores.sum().item(),
                **(score_summary(scores) if method == "wanda" else {})
            }

        return infos

    return compress
