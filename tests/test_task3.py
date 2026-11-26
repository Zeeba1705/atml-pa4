#task 3 step 0: exact speculative sampling (handout p.11, p.13 step 1)
import itertools

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from task3.spec_sampler import accept_prob, positive_residual, verify_round, speculative_generate, target_generate, HFRunner
from task3.toy_markov import MarkovRunner, exact_joint, sample_joint_counts, chi_square, TARGET, DRAFT, BOS


def one_hot_chain(next_of, start, V=5):
    #deterministic markov chain: after symbol s always next_of[s]; after BOS always start
    p0= torch.zeros(V)
    p0[start]= 1.0
    P= torch.zeros(V, V)

    for s, t in next_of.items():
        P[s, t]= 1.0

    return {"p0": p0, "P": P}


#--- one-round primitives ---

def test_accept_prob():
    p= torch.tensor([0.5, 0.3, 0.2])
    q= torch.tensor([0.25, 0.6, 0.15])

    assert torch.allclose(accept_prob(p, q, torch.tensor([0, 1, 2])), torch.tensor([1.0, 0.5, 1.0]))


def test_positive_residual_normalized():
    p= torch.tensor([0.5, 0.3, 0.2, 0.0])
    q= torch.tensor([0.2, 0.1, 0.5, 0.2])

    #[p - q]+ = [0.3, 0.2, 0, 0] -> [0.6, 0.4, 0, 0]
    assert torch.allclose(positive_residual(p, q), torch.tensor([0.6, 0.4, 0.0, 0.0]))


def test_verify_round_rejects_impossible_proposal_and_samples_residual():
    #target never emits token 1; the draft proposed it -> rejected at once, residual is one-hot on 0
    p= torch.tensor([[1.0, 0.0, 0.0], [0.2, 0.3, 0.5], [0.2, 0.3, 0.5]])
    q= torch.tensor([[0.0, 1.0, 0.0], [0.2, 0.3, 0.5]])
    g= torch.Generator().manual_seed(0)

    tokens, info= verify_round(p, q, [1, 2], g)

    assert tokens == [0] and info["accepted"] == 0 and info["examined"] == 1 and not info["bonus"]


def test_verify_round_full_accept_adds_bonus():
    #draft equals target -> every proposal accepted, then one bonus token from the last target row
    p= torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    g= torch.Generator().manual_seed(0)

    tokens, info= verify_round(p, p[:2], [1, 2], g)

    assert tokens == [1, 2, 0] and info["accepted"] == 2 and info["examined"] == 2 and info["bonus"]


#--- whole generation on deterministic chains ---

def test_full_accept_round_and_cap():
    chain= one_hot_chain({0: 1, 1: 2, 2: 0, 3: 3, 4: 4}, start=0)
    g= torch.Generator().manual_seed(0)

    out, stats= speculative_generate(MarkovRunner(chain), MarkovRunner(chain), [BOS], gamma=4, max_new=9, eos_ids=[], gen=g)

    #rounds: gamma 4 (+bonus) = 5 tokens, then remaining 4 -> gamma 3 (+bonus) = 4 tokens; never above the cap
    assert out == [0, 1, 2, 0, 1, 2, 0, 1, 2]
    assert [r["proposed"] for r in stats["rounds"]] == [4, 3]
    assert stats["target_passes"] == 2 and all(r["bonus"] for r in stats["rounds"])


def test_gamma_larger_than_remaining_tokens():
    chain= one_hot_chain({0: 1, 1: 2, 2: 0, 3: 3, 4: 4}, start=0)
    g= torch.Generator().manual_seed(0)

    out, stats= speculative_generate(MarkovRunner(chain), MarkovRunner(chain), [BOS], gamma=4, max_new=3, eos_ids=[], gen=g)

    assert out == [0, 1, 2] and [r["proposed"] for r in stats["rounds"]] == [2]

    #one token left -> a plain target step, no draft pass
    out, stats= speculative_generate(MarkovRunner(chain), MarkovRunner(chain), [BOS], gamma=4, max_new=1, eos_ids=[], gen=g)
    assert out == [0] and stats["rounds"][0]["proposed"] == 0 and stats["draft_passes"] == 0


def test_eos_mid_round_stops_after_eos():
    #4 is eos; the round proposes 0, 1, 4, 4 and all are accepted, the output must end at the first eos
    chain= one_hot_chain({0: 1, 1: 4, 4: 4, 2: 2, 3: 3}, start=0)
    g= torch.Generator().manual_seed(0)

    out, stats= speculative_generate(MarkovRunner(chain), MarkovRunner(chain), [BOS], gamma=4, max_new=20, eos_ids=[4], gen=g)

    assert out == [0, 1, 4] and stats["stop"] == "eos"


#--- exactness: 3-symbol markov toy, 100,000 length-3 sequences (p.13) ---

def test_exact_joint_sums_to_one_and_is_markov():
    joint= exact_joint(TARGET, 3)

    assert abs(sum(joint.values()) - 1.0) < 1e-12
    assert len(joint) == 27 and min(joint.values()) * 100000 >= 10   #every expected count >= 10
    x= (0, 2, 1)
    assert abs(joint[x] - TARGET["p0"][0] * TARGET["P"][0, 2] * TARGET["P"][2, 1]) < 1e-12


def test_markov_chi_square_correct_sampler_passes_bug_rejected():
    joint= exact_joint(TARGET, 3)

    for gamma in [2, 4]:
        counts= sample_joint_counts(TARGET, DRAFT, gamma, 100000, seed=6304)
        stat, p= chi_square(counts, joint, 100000)
        assert p >= 0.01, (gamma, stat, p)

    #negative control: residual replaced by the target distribution p
    counts= sample_joint_counts(TARGET, DRAFT, 2, 100000, seed=6304, residual_fn=lambda p, q: p)
    stat, p= chi_square(counts, joint, 100000)
    assert p < 0.01, (stat, p)


#--- real-model cache handling on tiny random qwen2 models ---

def tiny_qwen(seed):
    torch.manual_seed(seed)
    config= Qwen2Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        attn_implementation="sdpa"
    )

    return Qwen2ForCausalLM(config).float().eval()


def test_split_prefill_equals_full_forward():
    model= tiny_qwen(1)
    ids= list(range(3, 30))

    runner= HFRunner(model, torch.device("cpu"))
    runner.feed(ids[:-5])
    rows= runner.feed(ids[-5:])

    with torch.no_grad():
        full= torch.softmax(model(torch.tensor([ids])).logits[0, -5:].float(), dim=-1)

    assert torch.allclose(rows, full, atol=1e-5)


def test_cache_rollback_matches_fresh_forward():
    #after speculative generation (with rejections and full accepts), each cache plus its pending tokens
    #gives the same next-token distribution as a fresh forward of the returned sequence
    target, draft= tiny_qwen(1), tiny_qwen(2)
    prompt= [5, 17, 33, 2, 8]
    seen= {"reject": 0, "full": 0}

    for seed in range(6):
        g= torch.Generator().manual_seed(seed)
        t_run, d_run= HFRunner(target, torch.device("cpu")), HFRunner(draft, torch.device("cpu"))

        out, stats= speculative_generate(t_run, d_run, prompt, gamma=3, max_new=25, eos_ids=[], gen=g)
        seq= prompt + out

        for r in stats["rounds"]:
            seen["reject"]+= int(r["examined"] > r["accepted"])
            seen["full"]+= int(r["bonus"])

        for run, model in [(t_run, target), (d_run, draft)]:
            assert run.length() <= len(seq) - 1
            cached= run.feed(seq[run.length():])[-1]

            with torch.no_grad():
                fresh= torch.softmax(model(torch.tensor([seq])).logits[0, -1].float(), dim=-1)

            assert torch.allclose(cached, fresh, atol=1e-5)

    assert seen["reject"] > 0 and seen["full"] > 0


def test_target_only_reference_matches_chain():
    #target-only generation with the same runner interface (the paired timing reference)
    chain= one_hot_chain({0: 2, 2: 1, 1: 0, 3: 3, 4: 4}, start=1)
    out, stats= target_generate(MarkovRunner(chain), [BOS], max_new=5, eos_ids=[], gen=torch.Generator().manual_seed(0))

    assert out == [1, 0, 2, 1, 0] and stats["target_passes"] == 5


def test_bank_positions_match_supplied_rule():
    #the loop from setup_data.prefix_banks (course script), copied verbatim for comparison
    import random
    from task3.prefix_banks import select_positions

    bins= [((1, 16), 8), ((17, 32), 8), ((33, 64), 16), ((65, 192), 16)]

    for seq_len in [5, 20, 100, 192]:
        rng= random.Random(f"PA4-banks-v1:6304:GSM8K-0003:teacher")
        positions= []
        for lo, hi, n in [(1, 16, 8), (17, 32, 8), (33, 64, 16), (65, 192, 16)]:
            candidates= list(range(lo, min(hi, seq_len) + 1))
            positions.extend(sorted(rng.sample(candidates, min(n, len(candidates)))))

        assert select_positions(seq_len, "GSM8K-0003", "teacher", bins) == positions

    assert len(select_positions(192, "x", "teacher", bins)) == 48 and len(select_positions(20, "x", "teacher", bins)) == 12
