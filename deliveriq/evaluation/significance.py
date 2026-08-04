"""Is model A really better than model B on the same test orders?

Both models are scored on the SAME orders, so the comparison is paired:
we look at the per-order difference in absolute error.

* Paired bootstrap: resample orders with replacement, recompute
  MAE(A) - MAE(B) each time -> 95% confidence interval + two-sided p-value.
* Wilcoxon signed-rank test on the per-order differences: a
  non-parametric check that does not assume normal errors.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from deliveriq import config


def paired_mae_test(y, pred_a, pred_b, n_boot: int = 2000,
                    seed: int = config.RANDOM_STATE) -> dict:
    y = np.asarray(y, float)
    ea = np.abs(y - np.asarray(pred_a, float))
    eb = np.abs(y - np.asarray(pred_b, float))
    diff = ea - eb                       # < 0 -> A is better on that order
    rng = np.random.default_rng(seed)
    n = len(diff)
    boots = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # two-sided bootstrap p-value: how often the resampled mean crosses 0
    p_boot = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    w = stats.wilcoxon(diff)
    return {
        "MAE_A": float(ea.mean()), "MAE_B": float(eb.mean()),
        "diff_A_minus_B": float(diff.mean()),
        "ci95_low": float(lo), "ci95_high": float(hi),
        "p_bootstrap": float(max(p_boot, 1 / n_boot)),
        "p_wilcoxon": float(w.pvalue),
        "A_better_share_pct": float((diff < 0).mean() * 100),
        "n_orders": int(n),
    }


def block_bootstrap_by_day(y, pred_a, pred_b, days, n_boot: int = 2000,
                           seed: int = config.RANDOM_STATE) -> dict:
    """Same idea, but resamples whole days (orders on one day are correlated)."""
    y = np.asarray(y, float)
    diff = np.abs(y - pred_a) - np.abs(y - pred_b)
    days = np.asarray(days)
    uniq = np.unique(days)
    groups = [diff[days == d] for d in uniq]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        boots.append(np.concatenate([groups[i] for i in pick]).mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"diff": float(diff.mean()), "ci95_low": float(lo), "ci95_high": float(hi),
            "n_days": int(len(uniq))}
