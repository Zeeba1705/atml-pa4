import numpy as np

from common.bootstrap import paired_bootstrap, bootstrap_stat


def test_paired_bootstrap_contains_truth_and_detects_shift():
    rng= np.random.default_rng(0)
    null= paired_bootstrap(rng.normal(0, 1, 400))
    assert null["ci_low"] < 0 < null["ci_high"] and null["p_two_sided"] > 0.05
    shifted= paired_bootstrap(rng.normal(0.5, 1, 400))
    assert shifted["ci_low"] > 0 and shifted["p_two_sided"] < 0.01
    #deterministic for a fixed seed
    d= rng.normal(0, 1, 50)
    assert paired_bootstrap(d) == paired_bootstrap(d)


def test_bootstrap_stat_pooled_ratio():
    units= [{"nll": 2.0 * n, "n": n} for n in [10, 20, 30, 40]]
    r= bootstrap_stat(units, lambda us: sum(u["nll"] for u in us) / sum(u["n"] for u in us), resamples=200)
    assert r["estimate"] == 2.0 and r["ci_low"] == r["ci_high"] == 2.0
