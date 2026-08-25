"""Statistical tests used in Phase 3, each returning a flat dict.

Every test reports an EFFECT SIZE next to the p-value. With ~36,000 orders
almost any difference is "significant"; the effect size tells you whether it
matters.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def mean_ci(x, conf: float = 0.95) -> tuple[float, float, float]:
    x = np.asarray(pd.Series(x).dropna(), float)
    m = x.mean()
    h = stats.sem(x) * stats.t.ppf((1 + conf) / 2, len(x) - 1)
    return float(m), float(m - h), float(m + h)


def group_summary(df: pd.DataFrame, group: str, value: str) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(group, dropna=False):
        m, lo, hi = mean_ci(g[value])
        rows.append({group: key, "n": len(g), "mean": m, "ci95_low": lo, "ci95_high": hi,
                     "median": float(g[value].median()), "sd": float(g[value].std())})
    return pd.DataFrame(rows)


def hedges_g(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    d = (a.mean() - b.mean()) / sp
    return float(d * (1 - 3 / (4 * (na + nb) - 9)))


def two_group_test(a, b, label_a: str, label_b: str, n_boot: int = 2000, seed: int = 42) -> dict:
    """Welch t-test + Mann-Whitney U + bootstrap CI of the mean difference."""
    a = np.asarray(pd.Series(a).dropna(), float)
    b = np.asarray(pd.Series(b).dropna(), float)
    t = stats.ttest_ind(a, b, equal_var=False)
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    rng = np.random.default_rng(seed)
    boots = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # rank-biserial correlation as the effect size for Mann-Whitney
    rbc = 1 - 2 * u.statistic / (len(a) * len(b))
    return {"comparison": f"{label_a} vs {label_b}", "n_a": len(a), "n_b": len(b),
            "mean_a": a.mean(), "mean_b": b.mean(), "diff": a.mean() - b.mean(),
            "diff_ci95_low": lo, "diff_ci95_high": hi,
            "welch_t": float(t.statistic), "p_welch": float(t.pvalue),
            "p_mannwhitney": float(u.pvalue), "hedges_g": hedges_g(a, b),
            "rank_biserial": float(-rbc)}


def multi_group_test(df: pd.DataFrame, group: str, value: str) -> dict:
    """Welch-type ANOVA (Alexander-Govern) + Kruskal-Wallis, with effect sizes."""
    d = df[[group, value]].dropna()
    samples = [g[value].to_numpy(float) for _, g in d.groupby(group)]
    k, n = len(samples), len(d)
    ag = stats.alexandergovern(*samples)
    kw = stats.kruskal(*samples)
    grand = d[value].mean()
    ss_between = sum(len(s) * (s.mean() - grand) ** 2 for s in samples)
    ss_total = ((d[value] - grand) ** 2).sum()
    return {"test": f"{value} by {group}", "groups": k, "n": n,
            "p_alexander_govern": float(ag.pvalue), "H_kruskal": float(kw.statistic),
            "p_kruskal": float(kw.pvalue),
            "eta_squared": float(ss_between / ss_total),
            "epsilon_squared": float((kw.statistic - k + 1) / (n - k))}


def chi_square(df: pd.DataFrame, a: str, b: str) -> dict:
    tab = pd.crosstab(df[a], df[b])
    chi2, p, dof, _ = stats.chi2_contingency(tab)
    n = tab.to_numpy().sum()
    v = np.sqrt(chi2 / (n * (min(tab.shape) - 1)))
    return {"test": f"{a} x {b}", "chi2": float(chi2), "dof": int(dof),
            "p_value": float(p), "cramers_v": float(v), "n": int(n)}


def spearman(df: pd.DataFrame, x: str, y: str) -> dict:
    d = df[[x, y]].dropna()
    r = stats.spearmanr(d[x], d[y])
    return {"x": x, "y": y, "spearman_rho": float(r.statistic), "p_value": float(r.pvalue), "n": len(d)}


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (controls family-wise error)."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def within_band_slope(df: pd.DataFrame, x: str, y: str, lo: float, hi: float) -> dict:
    """OLS slope of y on x inside [lo, hi]: ~0 means a flat step, not a trend."""
    d = df[(df[x] >= lo) & (df[x] <= hi)][[x, y]].dropna()
    r = stats.linregress(d[x], d[y])
    return {"band": f"{x} in [{lo}, {hi}]", "n": len(d), "slope_per_unit": float(r.slope),
            "slope_ci95": float(1.96 * r.stderr), "p_value": float(r.pvalue),
            "r_squared": float(r.rvalue ** 2)}
