#paired bootstrap over units (prompts, examples, windows): resample unit indices with replacement, recompute the
#statistic, percentile interval. used by every task's make_tables (handout p.10 and p.15: >= 2,000 resamples)
import numpy as np

RESAMPLES= 2000


def paired_bootstrap(diffs, resamples=RESAMPLES, seed=6304, level=0.95):
    #diffs: one paired difference per unit (e.g. per-prompt mean of method A minus method B)
    diffs= np.asarray(diffs, dtype=float)
    rng= np.random.default_rng(seed)
    idx= rng.integers(0, len(diffs), size=(resamples, len(diffs)))
    means= diffs[idx].mean(axis=1)
    a= (1 - level) / 2

    return {
        "n_units": len(diffs), "mean_diff": float(diffs.mean()), "resamples": resamples, "level": level,
        "ci_low": float(np.quantile(means, a)), "ci_high": float(np.quantile(means, 1 - a)),
        "p_two_sided": float(min(1.0, 2 * min((means <= 0).mean(), (means >= 0).mean())))
    }


def bootstrap_stat(units, stat, resamples=RESAMPLES, seed=6304, level=0.95):
    #general form: units = list of per-unit records, stat(list_of_units) -> float (e.g. pooled nll -> ppl)
    rng= np.random.default_rng(seed)
    n= len(units)
    vals= np.array([stat([units[i] for i in rng.integers(0, n, n)]) for _ in range(resamples)])
    a= (1 - level) / 2

    return {"n_units": n, "estimate": float(stat(units)), "resamples": resamples, "level": level,
            "ci_low": float(np.quantile(vals, a)), "ci_high": float(np.quantile(vals, 1 - a))}
