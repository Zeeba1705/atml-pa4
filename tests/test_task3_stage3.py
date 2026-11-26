#task 3 stage 3 pieces: standalone answer scoring (decision 3.16) and bank alignment metrics / prefix indexing
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from task3.standalone_eval import extract_math, score_math, score_short
from task3.align_metrics import pair_metrics, group_prefixes, rows_at, position_bin


def test_math_extraction_order_and_comparison():
    assert extract_math("so 3 + 4 = 7, the answer is \\boxed{7}. Check 12") == (7.0, "boxed")
    assert extract_math("work 5 then #### 1,250") == (1250.0, "hashes")
    assert extract_math("It costs $18.00 in total.") == (18.0, "last_number")
    assert extract_math("no digits here") == (None, "none")
    assert score_math("The answer is 1,250.", "1250")["correct"]
    assert not score_math("The answer is 125.", "1250")["correct"]
    assert not score_math("nothing", "3")["extractable"]


def test_short_compliance_and_correctness():
    s= score_short(" **May**.\n", "May", 0)
    assert s["compliant"] and s["correct"]
    s= score_short("The month is May.", "May", 0)
    assert not s["compliant"] and not s["correct"] and s["correct_lenient"]
    assert score_short("888", "888", 1)["correct"]
    #digits inside a longer number do not count as the answer, even leniently
    assert not score_short("8889", "888", 1)["correct_lenient"]
    assert not score_short("pk", "PK", 2)["compliant"]
    assert score_short("'DE'", "DE", 2)["correct"]


def test_pair_metrics_hand_example():
    lp= torch.log(torch.tensor([[0.5, 0.5, 1e-30], [0.7, 0.2, 0.1]]))
    lq= torch.log(torch.tensor([[0.5, 0.5, 1e-30], [0.1, 0.2, 0.7]]))
    m= pair_metrics(lp, lq)

    assert torch.allclose(m["fkl"][0], torch.tensor(0.0), atol=1e-6) and m["tv"][0].abs() < 1e-6
    assert torch.allclose(m["tv"][1], torch.tensor(0.6), atol=1e-6)
    hand= 0.7 * torch.log(torch.tensor(7.0)) + 0.1 * torch.log(torch.tensor(1 / 7))
    assert torch.allclose(m["fkl"][1], hand, atol=1e-5) and torch.allclose(m["rkl"][1], hand, atol=1e-5)
    assert m["greedy_agree"].tolist() == [1.0, 0.0]


def test_bank_rows_match_full_forward():
    torch.manual_seed(0)
    model= Qwen2ForCausalLM(Qwen2Config(vocab_size=50, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                        num_attention_heads=4, num_key_value_heads=2)).float().eval()
    prompt, seq= [1, 2, 3], [10, 11, 12, 13, 14]
    recs= [{"prompt_id": "x", "position": pos, "response_prefix_token_ids": seq[:pos - 1]} for pos in [1, 3, 5]]
    longest= group_prefixes(recs)
    rows= [len(prompt) + r["position"] - 2 for r in recs]

    got= rows_at(model, prompt + longest, rows, torch.device("cpu"))
    #position pos is predicted from prompt + seq[:pos-1]: the last logit of a fresh forward on exactly that prefix
    for i, r in enumerate(recs):
        with torch.no_grad():
            ref= model(torch.tensor([prompt + r["response_prefix_token_ids"]])).logits[0, -1]
        assert torch.allclose(got[i], ref, atol=1e-5)

    with pytest.raises(AssertionError):
        group_prefixes(recs + [{"prompt_id": "x", "position": 2, "response_prefix_token_ids": [99]}])

    assert [position_bin(p) for p in [1, 16, 17, 64, 65, 192]] == ["1-16", "1-16", "17-32", "33-64", "65-192", "65-192"]
