#prints the task 2b sweep status: test accuracy per run (mean over finished seeds) at every K and regime
import os
import json
import argparse

from common.io_utils import load_config


def cell_table(metrics, k):
    return " ".join(f"{reg}={sum(m['cells'][f'K{k}_{reg}']['accuracy'] for m in metrics) / len(metrics):.3f}" for reg in ["near", "far", "mixed"])


def main(args):
    cfg= load_config(args.config)
    root= os.path.join(args.results_dir, "task2", "mqar")
    done= 0
    total= 0

    for run in cfg["runs"]:
        tags= ["test"] + (["test_window_mask"] if run["name"] in cfg["test_time_window_control"] else [])

        for tag in tags:
            metrics= []
            seeds= []

            for seed in cfg["seeds"]:
                total+= 1
                p= os.path.join(root, f"{run['name']}_s{seed}", f"{tag}_metrics.json")

                if os.path.exists(p):
                    metrics.append(json.load(open(p)))
                    seeds.append(seed)
                    done+= 1

            label= run["name"] + ("" if tag == "test" else " +test-time W64")

            if not metrics:
                print(f"{label:34s} not done")
                continue

            ms= sum(m["ms_per_update"] for m in metrics) / len(metrics)
            print(f"{label:34s} seeds {seeds}  {ms:.1f} ms/update")

            for k in [4, 8, 16]:
                mark= "*" if k == run["K"] else " "
                print(f"    {mark}K={k:<2d} {cell_table(metrics, k)}")

    print(f"\n{done}/{total} test results present (* = trained K)")

    failed= os.path.join(args.results_dir, "task2", "logs", "failed.log")

    if os.path.exists(failed):
        print("FAILED:\n" + open(failed).read())


if __name__ == "__main__":
    parser= argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/task2/mqar.yaml"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results"
    )

    args= parser.parse_args()
    main(args)
