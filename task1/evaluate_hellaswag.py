import re


def preprocess(text):
    #copied from lm-evaluation-harness 0.4.13, tasks/hellaswag/utils.py
    text= text.strip()
    text= text.replace(" [title]", ". ")
    text= re.sub("\\[.*?\\]", "", text)
    text= text.replace("  ", " ")

    return text


def build_query_and_choices(example):
    #lm-eval hellaswag v1.0 process_docs
    ctx= example["ctx_a"] + " " + example["ctx_b"].capitalize()

    query= preprocess(example["activity_label"] + ": " + ctx)
    choices= [preprocess(ending) for ending in example["endings"]]
    gold= int(example["label"])

    return query, choices, gold


def pick_answers(lls, choices):
    #acc uses the summed log-likelihood; acc_norm divides by the character length of the choice
    #(lm-eval scores " " + choice but normalises by len(choice))
    norm= [ll / len(choice) for ll, choice in zip(lls, choices)]

    pred= max(range(len(lls)), key=lambda i: lls[i])
    pred_norm= max(range(len(norm)), key=lambda i: norm[i])

    return pred, pred_norm
