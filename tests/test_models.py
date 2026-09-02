"""Metrics, significance and interval maths on small synthetic data."""
import numpy as np
import pytest

from deliveriq.evaluation.metrics import regression_metrics
from deliveriq.evaluation.significance import paired_mae_test
from deliveriq.models.intervals import SplitConformal, conformal_quantile, coverage_report
from deliveriq.stats.tests import holm


def test_business_metrics():
    y = np.array([10, 20, 30, 40])
    p = np.array([10, 26, 30, 30])          # one early by 6, one late by 10
    m = regression_metrics(y, p)
    assert m["MAE"] == 4
    assert m["late_gt5_pct"] == 25 and m["early_gt5_pct"] == 25
    assert m["within_5min_pct"] == 50


def test_conformal_quantile_rank():
    scores = np.arange(1, 11)                # n = 10, alpha 0.2 -> ceil(11*0.8)=9th value
    assert conformal_quantile(scores, 0.2) == 9


def test_split_conformal_covers_on_exchangeable_data():
    rng = np.random.default_rng(0)
    y_cal, y_test = rng.normal(0, 3, 5000), rng.normal(0, 3, 5000)
    sc = SplitConformal(0.2).calibrate(y_cal, np.zeros_like(y_cal))
    lo, hi = sc.interval(np.zeros_like(y_test))
    cov = coverage_report(y_test, lo, hi)["coverage_pct"]
    assert 78 <= cov <= 82


def test_paired_test_detects_a_real_difference():
    rng = np.random.default_rng(1)
    y = rng.normal(30, 5, 3000)
    good = y + rng.normal(0, 1, 3000)
    bad = y + rng.normal(0, 3, 3000)
    r = paired_mae_test(y, good, bad, n_boot=300)
    assert r["diff_A_minus_B"] < 0 and r["ci95_high"] < 0 and r["p_wilcoxon"] < 1e-6


def test_holm_is_monotone_and_bounded():
    raw = [0.01, 0.04, 0.03, 0.5]
    adj = holm(raw)
    in_raw_order = [adj[i] for i in np.argsort(raw)]
    assert in_raw_order == sorted(in_raw_order)   # never less significant than a smaller p
    assert all(0 <= a <= 1 for a in adj)
    assert adj[0] == pytest.approx(0.04)          # 4 * 0.01
    assert adj[2] == pytest.approx(0.09)          # 3 * 0.03
