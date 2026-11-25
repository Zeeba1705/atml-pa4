#prints the task 2b sweep status: test accuracy per run (mean over finished seeds) at every K and regime
import os
import glob
import json
import argparse

from common.io_utils import load_config


def cell_table(metrics, k):
    return " ".join(f"{reg}={sum(m['cells'][f'K{k}_{reg}']['accuracy'] for m in metrics) / len(metrics):.3f}" for reg in ["near", "far", "mixed"])


def attention_summary(results_dir):
    files= sorted(glob.glob(os.path.join(results_dir, "task2", "attention", "*_L*.json")), key=lambda f: (f.split("_L")[0], int(f.split("_L")[1][:-5])))

    for f in files:
        d= json.load(open(f))
        print(f"--- A {d['dtype']} L={d['T']}  ref={d['reference']}  sdpa default={d['sdpa_default_backend']}")

        for r in d["rows"]:
            if r["status"] != "ok":
                print(f"    {r['impl']:26s} {r['status']}: {r.get('message', '')[:80]}")
                continue

            fb= f"{r['forward_backward']['median']:9.2f} ms" if r["backward_status"] == "ok" else f"{r['backward_status']:>12s}"
            mem= r["forward_memory"]["peak_allocated_bytes"]
            print(
                f"    {r['impl']:26s} fwd {r['forward']['median']:9.2f} ms  fwd+bwd {fb}  "
                f"fwd peak {mem / 2**20 if mem else 0:8.0f} MiB  out err {r['forward_errors_vs_reference']['output']['max_abs']:.2e}"
            )


def main(args):
    cfg= load_config(args.config)
    attention_summary(args.results_dir)
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
