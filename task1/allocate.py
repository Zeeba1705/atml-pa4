import os
import json
import math
import time
import hashlib
import argparse

import torch

from common.seed import set_seed
from common.io_utils import load_config, default_output_dir, write_json, resolve_data_root, git_commit
from task1.load import eligible_linears
from task1.quantize import packed_bytes


def mean_sparsity(raw, sizes, t, s_min, s_max):
    total= sum(sizes)

    return sum(min(max(r + t, s_min), s_max) * n for r, n in zip(raw, sizes)) / total


def rescale_sparsities(raw, sizes, target, s_min, s_max):
    #additive shift t, clipped to [s_min, s_max]; the mean is monotone in t so bisection finds it
    assert s_min <= target <= s_max

    lo= s_min - max(raw)
    hi= s_max - min(raw)

    for _ in range(200):
        mid= (lo + hi) / 2

        if mean_sparsity(raw, sizes, mid, s_min, s_max) < target:
            lo= mid
        else:
            hi= mid

    t= (lo + hi) / 2

    return [min(max(r + t, s_min), s_max) for r in raw]


def integer_zero_counts(s, sizes, target_zeros):
    #floor, then largest-remainder so the total is exact (decision 1.6)
    exact= [si * n for si, n in zip(s, sizes)]
    zeros= [int(e) for e in exact]

    missing= target_zeros - sum(zeros)
    assert 0 <= missing <= len(zeros), missing

    order= sorted(range(len(zeros)), key=lambda i: exact[i] - zeros[i], reverse=True)

    for i in order[:missing]:
        zeros[i]+= 1

    assert sum(zeros) == target_zeros

    return zeros


def row_counts(n_zeros, n_rows):
    base= n_zeros // n_rows
    extra= n_zeros % n_rows

    return [base + 1 if i < extra else base for i in range(n_rows)]


#--- decision 1.7: p3 rank-linear rule and q3 exact ilp ---

def rank_raw_sparsities(sens, target, lam):
    #least sensitive unit -> target + lam, most sensitive -> target - lam, linear in rank (ties keep input order)
    n= len(sens)

    if n == 1 or lam == 0:
        return [target] * n

    order= sorted(range(n), key=lambda i: sens[i])
    rank= [0] * n

    for r, i in enumerate(order):
        rank[i]= r

    return [target + lam * (1 - 2 * rank[i] / (n - 1)) for i in range(n)]


def p3_allocation(sens, shapes, target, lam, s_min, s_max):
    #raw rank-linear sparsities -> shift/clip to the exact mean (decision 1.6) -> integer zeros per unit and per row
    sizes= [dout * din for dout, din in shapes]
    raw= rank_raw_sparsities(sens, target, lam)
    s= rescale_sparsities(raw, sizes, target, s_min, s_max)

    #p2 prunes exactly target * din per row, so its total zero count is target * size per unit
    target_zeros= sum(int(round(target * n)) for n in sizes)
    zeros= integer_zero_counts(s, sizes, target_zeros)
    row_zeros= [row_counts(z, dout) for z, (dout, _) in zip(zeros, shapes)]

    for rows, (_, din) in zip(row_zeros, shapes):
        assert max(rows) <= din

    return {
        "raw": raw,
        "sparsity": s,
        "achieved": [z / n for z, n in zip(zeros, sizes)],
        "zeros": zeros,
        "row_zeros": row_zeros
    }


def q3_allocation(sens, bytes_, bits, budget):
    #choose one bit width per unit minimizing the summed sensitivity with total packed bytes <= budget (exact ilp)
    import numpy as np
    from scipy.optimize import milp, LinearConstraint, Bounds

    n_units= len(sens)
    k= len(bits)

    assert sum(min(row) for row in bytes_) <= budget, "budget below the cheapest assignment"

    c= np.array([v for row in sens for v in row], dtype=np.float64)
    c= c / max(c.max(), 1e-300)
    b= np.array([v for row in bytes_ for v in row], dtype=np.float64)

    one_each= np.zeros((n_units, n_units * k))

    for u in range(n_units):
        one_each[u, u * k:(u + 1) * k]= 1

    limit= budget

    #the solver's feasibility tolerance is relative: re-solve with a tighter limit until the integer bytes fit exactly
    for _ in range(50):
        res= milp(
            c,
            constraints=[LinearConstraint(one_each, 1, 1), LinearConstraint(b[None, :], -np.inf, limit)],
            integrality=np.ones_like(c),
            bounds=Bounds(0, 1),
            options={"mip_rel_gap": 0.0}
        )
        assert res.success, res.message

        choice= [int(np.argmax(res.x[u * k:(u + 1) * k])) for u in range(n_units)]
        total= sum(bytes_[u][choice[u]] for u in range(n_units))

        if total <= budget:
            break

        limit-= max(1, total - budget)
    else:
        raise RuntimeError("could not meet the byte budget exactly")

    return {
        "bits": [bits[i] for i in choice],
        "bytes": total,
        "objective": sum(sens[u][choice[u]] for u in range(n_units))
    }


#--- cli: freeze p3/q3 allocations to splits/task1 before any held-out result exists for that method ---

def splits_dir(output_dir, smoke):
    return os.path.join(output_dir, "smoke", "task1", "splits") if smoke else os.path.join("splits", "task1")


def p3_path(base, sparsity):
    return os.path.join(base, f"p3_allocation_s{sparsity:g}.json")


def q3_path(base, bits):
    return os.path.join(base, f"q3_allocation_b{bits}.json")


def file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def assert_no_heldout(results_dir, condition):
    #p.5: allocations are chosen from calibration/dev data only, so freeze before any held-out result exists
    for e in ["wikitext2", "c4_val", "hellaswag"]:
        path= os.path.join(results_dir, condition, f"{e}.json")
        assert not os.path.exists(path), f"{path} exists: refusing to (re)choose an allocation after held-out results"


def unit_shapes(sens):
    return [tuple(u["shape"]) for u in sens["units"]]


def per_block(sens, values):
    #flat per-unit list -> one {proj: value} dict per block, as make_pruner / make_quantizer take it
    n_layers= max(u["layer"] for u in sens["units"]) + 1
    blocks= [{} for _ in range(n_layers)]

    for u, v in zip(sens["units"], values):
        blocks[u["layer"]][u["proj"]]= v

    return blocks


@torch.no_grad()
def dev_nll(model, sequences, device):
    #every next-token target of each dev sequence, batch 1
    from task1.evaluate_ppl import window_nll

    total, n= 0.0, 0

    for s in sequences:
        ids= torch.tensor(s["token_ids"], device=device)
        logits= model(ids[None]).logits[0]
        assert torch.isfinite(logits).all()

        nll, k= window_nll(logits, ids, 1, len(ids))
        total+= nll
        n+= k

    return total, n


def run_p3(args, cfg, base, sens, sens_file):
    from task1.load import load_model, load_sequences
    from task1.calibrate import sequential_compress
    from task1.prune import make_pruner

    pruning= cfg["pruning"]
    target= args.sparsity
    assert target in pruning["p3_sparsities"]

    data_root= resolve_data_root(cfg["data_repo"], args.data_root)
    calib= load_sequences(cfg, data_root, "calibration", cfg["calibration_sequences"])
    dev= load_sequences(cfg, data_root, "development", cfg["development_sequences"])

    if args.smoke:
        calib= [{"token_ids": s["token_ids"][:64]} for s in calib[:cfg["smoke"]["max_examples"]]]
        dev= [{"token_ids": s["token_ids"][:64]} for s in dev[:cfg["smoke"]["max_examples"]]]

    device= torch.device(args.device)
    shapes= unit_shapes(sens)
    scores= [u["taylor_pruned_mean"][str(target)] for u in sens["units"]]
    sweep= []

    for lam in pruning["p3_lambda_grid"]:
        path= os.path.join(base, "p3_dev", f"s{target:g}_lam{lam:g}.json")

        if os.path.exists(path):
            with open(path) as f:
                sweep.append(json.load(f))
            print(f"lam {lam:g}: done, dev nll {sweep[-1]['dev_mean_nll']:.5f}")
            continue

        alloc= p3_allocation(scores, shapes, target, lam, pruning["p3_min_unit_sparsity"], pruning["p3_max_unit_sparsity"])

        #fresh unchanged model per candidate, same calibration pass as p2
        set_seed(cfg["seed"])
        model= load_model(cfg, device, args.smoke)
        start= time.perf_counter()
        fn= make_pruner("wanda", target, row_zeros=per_block(sens, alloc["row_zeros"]))
        sequential_compress(model, calib, device, 8, fn)

        zeros_now= sum(int((m.weight == 0).sum().item()) for _, m in eligible_linears(model))
        nll, n= dev_nll(model, dev, device)

        row= {
            "lambda": lam,
            "dev_nll_sum": nll,
            "dev_tokens": n,
            "dev_mean_nll": nll / n,
            "dev_ppl": math.exp(nll / n),
            "eligible_zeros": zeros_now,
            "allocated_zeros": sum(alloc["zeros"]),
            "build_seconds": time.perf_counter() - start
        }
        write_json(path, row)
        sweep.append(row)
        print(f"lam {lam:g}: dev nll {row['dev_mean_nll']:.5f} (ppl {row['dev_ppl']:.3f})")

        del model

        if device.type == "cuda":
            torch.cuda.empty_cache()

    #lowest dev nll; ties -> smaller lambda (grid is ascending)
    best= min(sweep, key=lambda r: (r["dev_mean_nll"], r["lambda"]))
    alloc= p3_allocation(scores, shapes, target, best["lambda"], pruning["p3_min_unit_sparsity"], pruning["p3_max_unit_sparsity"])

    return {
        "method": "p3_nonuniform",
        "target_sparsity": target,
        "unit": pruning["p3_unit"],
        "sensitivity": "taylor_pruned_mean",
        "rule": "s = target + lam * (1 - 2 rank / (U - 1)), rank ascending in sensitivity; shift+clip to [s_min, s_max] (decision 1.6)",
        "selection": "lowest dev mean nll over the lambda grid (32 c4 dev sequences); ties -> smaller lambda",
        "lambda_grid": pruning["p3_lambda_grid"],
        "dev_sweep": sweep,
        "chosen_lambda": best["lambda"],
        "sensitivity_file": sens_file,
        "sensitivity_sha256": file_sha256(sens_file),
        "units": [
            {"name": u["name"], "layer": u["layer"], "proj": u["proj"], "shape": u["shape"], "sensitivity": sc,
             "raw": r, "sparsity": s, "achieved": a, "zeros": z}
            for u, sc, r, s, a, z in zip(sens["units"], scores, alloc["raw"], alloc["sparsity"], alloc["achieved"], alloc["zeros"])
        ],
        "total_zeros": sum(alloc["zeros"])
    }


def q3_budget(cfg, sens, bits):
    #exactly the q2 uniform-b serialized size: packed eligible + awq unfolded fp16 scales + all other params in fp16
    quant= cfg["quantization"]
    eligible= sum(packed_bytes(dout, din, bits, quant["group_size"]) for dout, din in unit_shapes(sens))
    unfolded= sum(2 * u["shape"][1] for u in sens["units"] if u["proj"] in quant["awq_unfolded_fp16_scales"])
    fixed= unfolded + 2 * sens["non_eligible_params"]

    return eligible + fixed, fixed


def run_q3(args, cfg, sens, sens_file):
    quant= cfg["quantization"]
    assert args.bits in [int(c.split("_b")[-1]) for c in quant["q3_budgets_from"]]

    allowed= quant["q3_allowed_bits"]
    budget, fixed= q3_budget(cfg, sens, args.bits)

    if not args.smoke:
        assert budget == quant["q3_expected_budget_bytes"][args.bits], (budget, quant["q3_expected_budget_bytes"])

    #cross-check against the measured q2 storage when that result is available (bytes only, no quality)
    q2= os.path.join(args.results_dir, f"q2_awq_b{args.bits}", "resources.json")

    if os.path.exists(q2) and not args.smoke:
        with open(q2) as f:
            measured= json.load(f)["storage"]["packed_total_bytes"]
        assert measured == budget, (measured, budget)

    shapes= unit_shapes(sens)
    scores= [[u["fisher_quant"][str(b)] for b in allowed] for u in sens["units"]]
    bytes_= [[packed_bytes(dout, din, b, quant["group_size"]) for b in allowed] for dout, din in shapes]
    alloc= q3_allocation(scores, bytes_, allowed, budget - fixed)

    uniform= allowed.index(args.bits)

    return {
        "method": "q3_mixed_budget",
        "budget_from": f"q2_awq_b{args.bits}",
        "budget_total_bytes": budget,
        "fixed_bytes": fixed,
        "packed_total_bytes": alloc["bytes"] + fixed,
        "unit": quant["q3_unit"],
        "allowed_bits": allowed,
        "sensitivity": "fisher_quant",
        "rule": "exact ilp (scipy milp, mip_rel_gap 0): minimize sum of fisher sensitivities s.t. packed bytes <= budget",
        "objective": alloc["objective"],
        "objective_uniform": sum(row[uniform] for row in scores),
        "sensitivity_file": sens_file,
        "sensitivity_sha256": file_sha256(sens_file),
        "bit_counts": {str(b): alloc["bits"].count(b) for b in allowed},
        "units": [
            {"name": u["name"], "layer": u["layer"], "proj": u["proj"], "shape": u["shape"], "bits": b,
             "packed_bytes": packed_bytes(u["shape"][0], u["shape"][1], b, quant["group_size"])}
            for u, b in zip(sens["units"], alloc["bits"])
        ]
    }


def main(args):
    cfg= load_config(args.config)
    base= splits_dir(args.output_dir, args.smoke)
    sens_file= os.path.join(base, "sensitivity.json")
    assert os.path.exists(sens_file), f"missing {sens_file}: run python -m task1.sensitivity first"

    with open(sens_file) as f:
        sens= json.load(f)

    results_dir= os.path.join(args.output_dir, "smoke", "task1") if args.smoke else args.results_dir
    prefix= "smoke_" if args.smoke else ""

    if args.method == "p3":
        path= p3_path(base, args.sparsity)
        condition= f"{prefix}p3_nonuniform_s{args.sparsity:g}"
    else:
        path= q3_path(base, args.bits)
        condition= f"{prefix}q3_mixed_budget_b{args.bits}"

    if os.path.exists(path):
        print(f"{path} is frozen, not recomputing")
        return

    assert_no_heldout(results_dir, condition)

    out= run_p3(args, cfg, base, sens, sens_file) if args.method == "p3" else run_q3(args, cfg, sens, sens_file)
    out.update({"git_commit": git_commit(), "seed": cfg["seed"], "smoke": args.smoke})

    write_json(path, out)

    if args.method == "p3":
        print(f"frozen {path}: lambda {out['chosen_lambda']:g}, zeros {out['total_zeros']}")
    else:
        print(f"frozen {path}: {out['bit_counts']}, {out['packed_total_bytes']} <= {out['budget_total_bytes']} bytes")


if __name__ == "__main__":
    parser= argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/task1/base.yaml"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["p3", "q3"],
        required=True
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.5
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=default_output_dir()
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=os.path.join("results", "task1")
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--smoke",
        action="store_true"
    )

    args= parser.parse_args()
    main(args)
