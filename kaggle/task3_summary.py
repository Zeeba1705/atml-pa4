#prints task 3 progress: real-model checks, step costs, speculative eval summaries, generated banks, teacher pool
import os
import glob
import json
import argparse

from common.io_utils import read_jsonl
from task3.spec_eval import summarize


def main(args):
    root= os.path.join(args.results_dir, "task3")

    for d in sorted(glob.glob(os.path.join(root, "spec_eval", "*"))):
        print(f"--- spec eval: {os.path.basename(d)}")
        p= os.path.join(d, "checks.json")

        if os.path.exists(p):
            c= json.load(open(p))
            print(f"    checks: worst prob diff {c['worst']:.2e} (tol {c['tolerance']}), vocab {c['agreement']['logit_vocab']}")

        p= os.path.join(d, "step_costs.json")

        if os.path.exists(p):
            for L, row in json.load(open(p))["costs"].items():
                print(f"    L={L}: target step {row['target_feed1']['median']:.2f} ms, draft step {row['draft_feed1']['median']:.2f} ms, c={row['c']:.3f}, v={ {k: round(v, 3) for k, v in row['v'].items()} }")

        for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
            rows= read_jsonl(f)
            s= summarize(rows)
            print(f"    {os.path.basename(f)}: {len(rows) // 2} prompts, accepted/examined={s['accepted_over_examined']:.3f}, accepted/proposed={s['accepted_over_proposed']:.3f}, target passes/token={s['target_passes_per_token_spec']:.3f}, median tok/s ratio={s['median_throughput_ratio']:.3f}")

    for f in sorted(glob.glob(os.path.join(root, "banks", "*_info.json"))):
        d= json.load(open(f))
        print(f"--- bank {d['bank']}: {d['n_records']} records, mean length {d['mean_response_length']:.1f}, eos {d['eos_fraction']:.2f}, {d['generation_seconds']:.0f}s")

    p= os.path.join(root, "teacher_pool_info.json")

    if os.path.exists(p):
        d= json.load(open(p))
        print(f"--- teacher pool: {d['n_prompts']} responses, tokens {d['tokens']}, eos {d['eos_fraction']}, {d['teacher_generation_seconds']:.0f}s")


if __name__ == "__main__":
    parser= argparse.ArgumentParser()

    parser.add_argument(
        "--results_dir",
        type=str,
        default="results"
    )

    args= parser.parse_args()
    main(args)
