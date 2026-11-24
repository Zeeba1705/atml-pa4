import math

import torch
import torch.nn.functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM

from common.io_utils import load_config
from task1.load import load_tokenizer, QWEN_VOCAB
from task1.evaluate_ppl import evaluate_ppl
from task1.evaluate_hellaswag import encode_pair, score_choices

CFG= load_config("configs/task1/base.yaml")
TOK= load_tokenizer(CFG)


def tiny_real_vocab_model():
    torch.manual_seed(6304)

    config= Qwen2Config(
        vocab_size=QWEN_VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=True,
        attn_implementation="sdpa"
    )

    return Qwen2ForCausalLM(config).float().eval()


def sequence_logprob(model, ids, n_context):
    #reference: one unpadded sequence, sum log p of tokens after the context
    with torch.no_grad():
        logits= model(torch.tensor([ids])).logits[0]

    logp= F.log_softmax(logits.float(), dim=-1)

    return sum(logp[t - 1, ids[t]].item() for t in range(n_context, len(ids)))


def test_encode_pair_splits_whole_string():
    ctx, cont= encode_pair(TOK, "The cat", " sat on the mat.")

    assert ctx + cont == TOK.encode("The cat sat on the mat.", add_special_tokens=False)
    assert TOK.decode(cont) == " sat on the mat."


def test_encode_pair_moves_trailing_context_space():
    ctx, cont= encode_pair(TOK, "The cat ", "sat")

    #same tokens as if the space had been part of the continuation
    assert (ctx, cont) == encode_pair(TOK, "The cat", " sat")


def test_batched_choice_scores_match_unpadded_reference():
    model= tiny_real_vocab_model()
    query= "Baking cookies: A woman is in a kitchen. She pours flour"
    choices= ["into a bowl.", " onto the floor and then sweeps it all up again.", "and sings.", "x"]

    lls, n_tokens= score_choices(model, TOK, query, choices, torch.device("cpu"))

    for choice, ll, n in zip(choices, lls, n_tokens):
        ctx, cont= encode_pair(TOK, query, " " + choice)

        assert n == len(cont)
        assert abs(ll - sequence_logprob(model, ctx + cont, len(ctx))) < 1e-4


def test_evaluate_ppl_matches_reference_and_skips_unscored():
    model= tiny_real_vocab_model()
    ids= TOK.encode("One two three four five six seven eight nine ten.", add_special_tokens=False)

    windows= [
        {"token_ids": ids, "score_target_offsets": [1, len(ids)], "scored_tokens": len(ids) - 1, "source_row": 0, "token_offset": 0},
        {"token_ids": ids, "score_target_offsets": [1, 4], "scored_tokens": 3, "source_row": 1, "token_offset": 0}
    ]

    rows, summary= evaluate_ppl(model, windows, torch.device("cpu"))

    ref_full= -sequence_logprob(model, ids, 1)
    ref_part= -sequence_logprob(model, ids[:4], 1)

    assert abs(rows[0]["nll_sum"] - ref_full) < 1e-3
    assert abs(rows[1]["nll_sum"] - ref_part) < 1e-3
    assert summary["n_scored"] == len(ids) - 1 + 3
    #pooled over tokens; ppl of the random tiny model is ~1.6e5, so compare it relatively (exp amplifies fp32 noise)
    ref_mean= (ref_full + ref_part) / summary["n_scored"]
    assert abs(summary["mean_nll"] - ref_mean) < 1e-4
    assert math.isclose(summary["ppl"], math.exp(ref_mean), rel_tol=1e-4)
