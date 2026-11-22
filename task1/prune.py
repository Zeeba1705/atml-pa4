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
