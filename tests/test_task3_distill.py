#task 3 distillation pieces: forward kl, prompt schedule and quotas, exact 256-token updates, lora setup
import torch
import torch.nn.functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM

from task3.distill import forward_kl_sum, category_perms, pick_presentation, update_chunks, add_lora, teacher_probs, student_log_probs


def test_forward_kl_hand_example():
    p= torch.tensor([[0.5, 0.5, 0.0], [0.2, 0.3, 0.5]])
    logits= torch.tensor([[1.0, 0.0, -1.0], [0.0, 0.0, 0.0]])
    logq= F.log_softmax(logits, dim=-1)

    hand= sum(
        sum(p[i, v] * (torch.log(p[i, v]) - logq[i, v]) for v in range(3) if p[i, v] > 0)
        for i in range(2)
    )

    assert torch.allclose(forward_kl_sum(p, logq), hand)
    #zero when the student equals the teacher
    assert forward_kl_sum(p[1:], torch.log(p[1:])).abs() < 1e-7


def test_schedule_alternates_cycles_and_meets_quota_exactly():
    cats= ["math"] * 5 + ["short"] * 3
    perms= category_perms(cats, seed=6304)

    assert sorted(perms["math"]) == [0, 1, 2, 3, 4] and sorted(perms["short"]) == [5, 6, 7]
    assert perms == category_perms(cats, seed=6304) and perms != category_perms(cats, seed=6305)

    quotas= {"math": 100, "short": 24}
    lengths= {i: (30 if c == "math" else 4) for i, c in enumerate(cats)}
    state= {"scored": {"math": 0, "short": 0}, "next": "math", "pos": {"math": 0, "short": 0}}
    seq= []

    while True:
        pick= pick_presentation(state, perms, quotas)

        if pick is None:
            break

        c, idx= pick
        n= min(lengths[idx], quotas[c] - state["scored"][c])   #truncated to the remaining quota
        state["scored"][c]+= n
        seq.append((c, idx, n))

    assert state["scored"] == quotas
    #alternation while both quotas are open, then only the open category
    assert [c for c, _, _ in seq[:6]] == ["math", "short"] * 3
    #prompts are reused after a category is exhausted, in permutation order
    shorts= [idx for c, idx, _ in seq if c == "short"]
    assert shorts[:6] == perms["short"] * 2
    #math: 30 + 30 + 30 + 10, the last one truncated to the remaining quota
    assert [n for c, _, n in seq if c == "math"] == [30, 30, 30, 10]


def test_update_chunks_are_exactly_256_with_short_final():
    #responses of 200, 300, 100 tokens and a 600-token total quota -> updates of 256, 256, 88
    chunks= update_chunks([200, 300, 100], total=600, size=256)

    sizes= {}
    for c in chunks:
        sizes[c["update"]]= sizes.get(c["update"], 0) + c["k"]

    assert sizes == {0: 256, 1: 256, 2: 88}
    #the 300-token response straddles updates 0 and 1 (56 + 244), the 100-token one updates 1 and 2 (12 + 88)
    assert [(c["presentation"], c["offset"], c["k"], c["update"]) for c in chunks] == [
        (0, 0, 200, 0), (1, 0, 56, 0), (1, 56, 244, 1), (2, 0, 12, 1), (2, 12, 88, 2)
    ]
    #every chunk knows its update's normalizer
    assert all(c["update_size"] == sizes[c["update"]] for c in chunks)


def tiny(seed):
    torch.manual_seed(seed)
    config= Qwen2Config(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                        num_attention_heads=4, num_key_value_heads=2, attn_implementation="sdpa")

    return Qwen2ForCausalLM(config).float().eval()


def test_lora_only_qkvo_fp32_and_count():
    model= add_lora(tiny(0), {"r": 8, "alpha": 16, "dropout": 0.0, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]}, seed=6304)
    trainable= {n: p for n, p in model.named_parameters() if p.requires_grad}

    assert trainable and all("lora_" in n for n in trainable)
    assert all(p.dtype == torch.float32 for p in trainable.values())
    #per layer: q 32->32, k 32->16, v 32->16, o 32->32; lora adds r*(in+out)
    per_layer= 8 * ((32 + 32) + (32 + 16) + (32 + 16) + (32 + 32))
    assert sum(p.numel() for p in trainable.values()) == 2 * per_layer
    #b starts at zero, so the adapted model equals the base model at init
    ids= torch.tensor([[3, 5, 7, 9]])
    with torch.no_grad():
        assert torch.allclose(model(ids).logits, tiny(0)(ids).logits, atol=1e-6)


def test_kl_positions_are_the_response_positions():
    #teacher rows i and student rows i both predict response token i given prompt + response[:i]
    teacher, student= tiny(1), tiny(2)
    prompt, resp= [4, 8, 15], [16, 23, 42, 7]

    p= teacher_probs(teacher, prompt, resp, torch.device("cpu"))
    logq= student_log_probs(student, prompt, resp, offset=1, k=2, device=torch.device("cpu"), autocast=False)

    with torch.no_grad():
        full_t= torch.softmax(teacher(torch.tensor([prompt + resp])).logits[0].float(), -1)
        full_s= F.log_softmax(student(torch.tensor([prompt + resp])).logits[0].float(), -1)

    #row for response token i sits at position len(prompt) - 1 + i
    assert torch.allclose(p, full_t[2:6], atol=1e-6)
    assert torch.allclose(logq, full_s[3:5], atol=1e-5)
