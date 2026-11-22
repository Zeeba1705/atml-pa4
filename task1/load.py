import torch

ELIGIBLE_ATTN= ["q_proj", "k_proj", "v_proj", "o_proj"]
ELIGIBLE_MLP= ["gate_proj", "up_proj", "down_proj"]


def eligible_linears(model):
    linears= []

    for i, layer in enumerate(model.model.layers):
        for name in ELIGIBLE_ATTN:
            linears.append((f"model.layers.{i}.self_attn.{name}", getattr(layer.self_attn, name)))

        for name in ELIGIBLE_MLP:
            linears.append((f"model.layers.{i}.mlp.{name}", getattr(layer.mlp, name)))

    return linears


def eligible_weight_count(model):
    return sum(linear.weight.numel() for _, linear in eligible_linears(model))


def unique_param_count(model):
    #tied lm_head/embedding share storage, so count each storage once
    seen= set()
    total= 0

    for _, p in model.named_parameters(remove_duplicate=False):
        key= (p.data_ptr(), p.numel())

        if key in seen:
            continue

        seen.add(key)
        total+= p.numel()

    return total
