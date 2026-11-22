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

    def forward(self, x):
        return self.linear(x / self.scales)
