import inspect

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.generation.utils import GenerationMixin

from kaggle.supplied_banks import patch_generate, ORIGINAL_GENERATE


def tiny_model():
    torch.manual_seed(6304)

    config= Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2
    )

    return Qwen2ForCausalLM(config).float().eval()


def sample(model, seed, use_generator):
    ids= torch.tensor([[1, 2, 3]])
    kwargs= dict(max_new_tokens=20, do_sample=True, temperature=1.0, top_p=1.0, top_k=0, pad_token_id=0)

    if use_generator:
        kwargs["generator"]= torch.Generator().manual_seed(seed)
    else:
        torch.manual_seed(seed)

    with torch.inference_mode():
        return model.generate(ids, **kwargs)[0].tolist()


def test_pinned_generate_has_no_generator_kwarg():
    #this is why the supplied script fails; if it ever starts failing the patch may no longer be needed
    assert "generator" not in inspect.signature(ORIGINAL_GENERATE).parameters


def test_patched_generate_matches_seeding_default_rng():
    model= tiny_model()
    patch_generate()

    try:
        a= sample(model, 6304, use_generator=True)
        b= sample(model, 6304, use_generator=True)
        c= sample(model, 6304, use_generator=False)
        d= sample(model, 6305, use_generator=True)

        assert a == b == c
        assert a != d
    finally:
        GenerationMixin.generate= ORIGINAL_GENERATE
