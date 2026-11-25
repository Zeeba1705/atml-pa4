#task 3 step 1 exactness check (handout p.13): 3-symbol first-order markov target and draft, length-3 sequences,
#100,000 fixed-seed speculative samples vs the exact joint, pearson chi-square; negative control = residual replaced by p
import os
import json
import time
import argparse
import itertools
from collections import Counter

import torch
from scipy.stats import chisquare

from common.io_utils import load_config, write_json, git_commit
from task3.spec_sampler import speculative_generate, positive_residual

BOS= 3   #start symbol, never emitted (probability 0 under both chains)

#rows 0..2 = next-symbol distribution after that symbol; p0 = after BOS. the draft differs clearly (tv ~ 0.3)
TARGET= {
    "p0": torch.tensor([0.5, 0.3, 0.2, 0.0], dtype=torch.float64),
    "P": torch.tensor([
        [0.6, 0.3, 0.1, 0.0],
        [0.2, 0.5, 0.3, 0.0],
        [0.3, 0.2, 0.5, 0.0],
        [0.0, 0.0, 0.0, 0.0]
    ], dtype=torch.float64)
}
DRAFT= {
    "p0": torch.tensor([0.2, 0.3, 0.5, 0.0], dtype=torch.float64),
    "P": torch.tensor([
        [0.3, 0.3, 0.4, 0.0],
        [0.5, 0.2, 0.3, 0.0],
        [0.1, 0.6, 0.3, 0.0],
        [0.0, 0.0, 0.0, 0.0]
    ], dtype=torch.float64)
}


class MarkovRunner:
    #same interface as spec_sampler.HFRunner: the "cache" is the list of fed symbols
    def __init__(self, chain):
        self.chain= chain
        self.tokens= []

    def length(self):
        return len(self.tokens)

    def feed(self, tokens, keep=None):
        rows= []

        for t in tokens:
            self.tokens.append(t)
            rows.append(self.chain["p0"] if t == BOS else self.chain["P"][t])

        rows= torch.stack(rows)

        return rows[-keep:] if keep else rows

    def crop(self, n):
        self.tokens= self.tokens[:n]


def exact_joint(chain, n):
    #p(x1..xn) = p0(x1) * prod P(x_{i-1}, x_i) over the 3 emitted symbols
    joint= {}

    for x in itertools.product(range(3), repeat=n):
        p= chain["p0"][x[0]]

        for a, b in zip(x, x[1:]):
            p= p * chain["P"][a, b]

        joint[x]= float(p)

    return joint


def sample_joint_counts(target, draft, gamma, n, seed, residual_fn=positive_residual, length=3):
    gen= torch.Generator().manual_seed(seed)
    counts= Counter()
    stats= {"rounds": 0, "accepted": 0, "proposed": 0, "bonus": 0}

    for _ in range(n):
        out, s= speculative_generate(MarkovRunner(target), MarkovRunner(draft), [BOS], gamma, length, [], gen, residual_fn)
        counts[tuple(out)]+= 1
        stats["rounds"]+= len(s["rounds"])
        stats["accepted"]+= s["accepted"]
        stats["proposed"]+= s["proposed"]
        stats["bonus"]+= sum(r["bonus"] for r in s["rounds"])

    counts.stats= stats

    return counts


def chi_square(counts, joint, n):
    keys= sorted(joint)
    obs= [counts.get(k, 0) for k in keys]
    exp= [joint[k] * n for k in keys]
    res= chisquare(obs, exp)

    return float(res.statistic), float(res.pvalue)


def main(args):
    cfg= load_config(args.config)["markov_test"]
    n= cfg["num_samples"]
    joint= exact_joint(TARGET, cfg["sequence_length"])
    assert min(joint.values()) * n >= cfg["min_expected_count"]

    out= {"git_commit": git_commit(), "seed": args.seed, "n_samples": n, "target": {k: v.tolist() for k, v in TARGET.items()},
          "draft": {k: v.tolist() for k, v in DRAFT.items()}, "exact_joint": {"".join(map(str, k)): v for k, v in joint.items()},
          "min_expected_count": min(joint.values()) * n, "dof": len(joint) - 1, "runs": []}

    #at length 3 the cap clips gamma >= 2 to 2 in the first round (gamma 2 and 4 take identical paths), so an extra
    #length-4 gamma-3 run exercises several rounds with full-length proposals (min expected count 20)
    joint4= exact_joint(TARGET, 4)
    assert min(joint4.values()) * n >= cfg["min_expected_count"]

    for name, gamma, fn, length in [("correct", 1, positive_residual, 3), ("correct", 2, positive_residual, 3), ("correct", 4, positive_residual, 3),
                                    ("bug_residual_is_p", 2, lambda p, q: p, 3), ("bug_residual_is_p", 4, lambda p, q: p, 3),
                                    ("correct", 3, positive_residual, 4), ("bug_residual_is_p", 3, lambda p, q: p, 4)]:
        start= time.perf_counter()
        ref= joint if length == 3 else joint4
        counts= sample_joint_counts(TARGET, DRAFT, gamma, n, args.seed, fn, length)
        stat, p= chi_square(counts, ref, n)
        reject= p < cfg["p_value_threshold"]

        out["runs"].append({
            "sampler": name, "gamma": gamma, "sequence_length": length, "dof": len(ref) - 1, "chi2": stat, "p_value": p, "rejected_at_0.01": reject,
            "empirical": {"".join(map(str, k)): counts.get(k, 0) / n for k in sorted(ref)},
            "rounds_per_sequence": counts.stats["rounds"] / n,
            "accepted_over_proposed": counts.stats["accepted"] / max(1, counts.stats["proposed"]),
            "bonus_rounds": counts.stats["bonus"],
            "seconds": time.perf_counter() - start
        })
        print(f"{name} gamma={gamma} length={length}: chi2={stat:.2f} p={p:.4g} -> {'REJECT' if reject else 'not rejected'}")

    write_json(os.path.join(args.results_dir, "task3", "markov_chi2.json"), out)


if __name__ == "__main__":
    parser= argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/task3/spec_eval.yaml"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=6304
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results"
    )

    args= parser.parse_args()
    main(args)
