"""Multi-seed statistics (D19 Level 1a): the CI and paired test must be correct."""
import math
import pytest

from cortex_eval.multiseed import _mean_ci, _paired_test, _t95


def test_t_critical_values_sane():
    assert _t95(4) == 2.776 and _t95(9) == 2.262
    assert _t95(100) == 1.96  # large df → normal


def test_mean_ci_known_values():
    # values with mean 0.10, sd 0.01, n=5 → half-width = 2.776*0.01/sqrt(5) ≈ 0.0124
    vals = [0.09, 0.10, 0.11, 0.10, 0.10]
    st = _mean_ci(vals)
    assert abs(st["mean"] - 0.10) < 1e-9
    lo, hi = st["ci95"]
    assert lo < 0.10 < hi
    assert abs((hi - lo) / 2 - 2.776 * st["sd"] / math.sqrt(5)) < 1e-3


def test_paired_test_detects_a_real_gap():
    # hopfield consistently 0.02 above control across 5 seeds → significant
    ctrl = [0.10, 0.11, 0.09, 0.10, 0.10]
    hop  = [0.12, 0.13, 0.11, 0.12, 0.12]
    t = _paired_test(hop, ctrl)
    assert t["significant_95"] is True and t["mean_diff"] == 0.02


def test_paired_test_rejects_noise():
    # gaps that flip sign across seeds → not significant
    ctrl = [0.10, 0.11, 0.09, 0.10, 0.10]
    dl   = [0.11, 0.10, 0.10, 0.09, 0.11]
    t = _paired_test(dl, ctrl)
    assert t["significant_95"] is False


def test_single_seed_makes_no_claim():
    t = _paired_test([0.12], [0.10])
    assert t["significant_95"] is None  # honest: one seed cannot claim
