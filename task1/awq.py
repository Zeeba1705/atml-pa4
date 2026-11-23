import torch
import torch.nn as nn


def awq_scales(x_mean_abs, alpha):
    #as in the llm-awq reference: s = x^alpha, normalised by sqrt(max*min)
    s= x_mean_abs.float().pow(alpha).clamp(min=1e-4)
    s= s / (s.max() * s.min()).sqrt()

    return s


def fold_into_norm(norm, linears, s):
    #norm output is divided by s, the following linears' input columns multiplied by s
    with torch.no_grad():
        norm.weight.div_(s.to(norm.weight.dtype))

        for linear in linears:
            linear.weight.mul_(s.to(linear.weight.dtype)[None, :])


def fold_into_linear(prev, nxt, s):
    #prev's output rows (and bias) divided by s, nxt's input columns multiplied by s
    with torch.no_grad():
        prev.weight.div_(s.to(prev.weight.dtype)[:, None])

        if prev.bias is not None:
            prev.bias.div_(s.to(prev.bias.dtype))

        nxt.weight.mul_(s.to(nxt.weight.dtype)[None, :])


class InputScaledLinear(nn.Module):
    #used where the inverse scale can't be folded (o_proj under gqa); the fp16 scales are stored and counted
    def __init__(self, linear, s):
        super().__init__()

        with torch.no_grad():
            linear.weight.mul_(s.to(linear.weight.dtype)[None, :])

        self.linear= linear
        self.register_buffer("scales", s.to(linear.weight.dtype).clone())

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def forward(self, x):
        return self.linear(x / self.scales)


def effective_weight(w, s, quantize_fn):
    #Q(W*s)/s: the weight seen by an unscaled input when the module gets X/s
    return quantize_fn(w.float() * s[None, :]) / s[None, :]


def awq_group_loss(xs, weights, w_hats, chunk):
    #sum over the group's projections of ||W x - W_hat x||^2 on the cached calibration inputs
    loss= 0.0

    for x in xs:
        for i in range(0, x.shape[0], chunk):
            xc= x[i:i + chunk]

            for w, w_hat in zip(weights, w_hats):
                diff= xc @ (w - w_hat.to(w.dtype)).T
                loss+= diff.float().pow(2).sum().item()

    return loss


def awq_search(xs, x_mean_abs, weights, alpha_grid, quantize_fn, chunk=8192):
    #grid search over alpha (s = x^alpha normalised); alpha=0 is plain uniform quantization
    losses= []

    for alpha in alpha_grid:
        s= awq_scales(x_mean_abs, alpha).to(weights[0].device)
        w_hats= [effective_weight(w, s, quantize_fn) for w in weights]
        losses.append(awq_group_loss(xs, weights, w_hats, chunk))

    best= min(range(len(alpha_grid)), key=lambda i: losses[i])

    return alpha_grid[best], awq_scales(x_mean_abs, alpha_grid[best]).to(weights[0].device), losses
