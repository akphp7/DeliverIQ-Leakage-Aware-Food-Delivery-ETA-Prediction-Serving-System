"""Phase 3 - statistics and EDA on the TRAINING period only.

Run:  python -m pipelines.p3_statistics

Why training period only: looking at the test days while forming hypotheses
is a quiet form of leakage ("data snooping").

Answers the five EDA questions from the project plan with tests + effect
sizes, investigates why rider age / rating dominate, checks linear-regression
assumptions, and applies a multiple-testing correction.

Outputs: outputs/v2/phase3/*.csv and *.png
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sps
from sklearn.feature_selection import mutual_info_regression
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

from deliveriq import config, viz
from deliveriq.data.cleaning import load_clean
from deliveriq.evaluation.splits import temporal_split
from deliveriq.features.build import base_features
from deliveriq.stats import tests as T
from pipelines.common import banner, distance_bucket

Y = config.TARGET


def dot_whisker(ax, summ, key, order, title, color):
    s = summ.set_index(key).reindex(order)
    x = np.arange(len(order))
    ax.errorbar(x, s["mean"], yerr=[s["mean"] - s["ci95_low"], s["ci95_high"] - s["mean"]],
                fmt="o", ms=7, color=color, ecolor=color, elinewidth=2, capsize=0)
    for xi, (m, n) in enumerate(zip(s["mean"], s["n"])):
        ax.text(xi + 0.1, m, f"{m:.1f}", ha="left", va="center", fontsize=8, color=viz.TEXT_2)
    ax.set_xticks(x, [f"{o}\n(n={int(n):,})" for o, n in zip(order, s["n"])], fontsize=8)
    ax.set_title(title)
    ax.set_ylabel("Mean delivery time (min), 95% CI")


def main():
    out = config.phase_dir(3)
    viz.style()
    clean_all, audit = load_clean()
    tr, _, cutoff, _ = temporal_split(clean_all)
    df = clean_all.iloc[tr].copy()
    X = base_features(df)
    df["distance_bucket"] = distance_bucket(df["distance_km"]).astype(str)
    df["is_peak_hour"] = X["is_peak_hour"].astype(int)
    df["hour"] = X["order_hour"]
    print(f"EDA on {len(df):,} training orders (before {cutoff.date()})")

    # ------------------------------------------------------------ 0. quality
    banner("0. Data quality")
    miss = (clean_all.isna().mean() * 100).rename("missing_pct").round(2)
    miss = miss[miss > 0].sort_values(ascending=False)
    flags = clean_all[["traffic_missing", "weather_missing", "rider_profile_invalid",
                       "order_time_imputed", "restaurant_coords_missing"]].mean().mul(100).round(2)
    quality = pd.concat([miss, flags.rename("missing_pct")]).rename("pct_of_orders")
    quality.to_csv(out / "data_quality.csv")
    print(quality.to_string())
    co = T.chi_square(clean_all, "traffic_missing", "order_time_imputed")
    print(f"\nMissing traffic vs missing order time: chi2={co['chi2']:.0f}, "
          f"Cramer's V={co['cramers_v']:.2f}, p={co['p_value']:.1e}")

    # ------------------------------------------------------------ 1. target
    banner("1. Target distribution")
    y = df[Y]
    tgt = {"n": len(y), "mean": y.mean(), "median": y.median(), "sd": y.std(),
           "min": y.min(), "max": y.max(), "p10": y.quantile(.1), "p90": y.quantile(.9),
           "skew": sps.skew(y), "excess_kurtosis": sps.kurtosis(y),
           "dagostino_p": sps.normaltest(y).pvalue}
    pd.Series(tgt).to_csv(out / "target_summary.csv")
    print(pd.Series(tgt).round(3).to_string())
    fig, ax = viz.plt.subplots(figsize=(7, 3.6))
    ax.hist(y, bins=np.arange(9.5, 55.5, 1), color=viz.SERIES[0], edgecolor=viz.SURFACE, linewidth=1)
    ax.axvline(y.median(), color=viz.TEXT_2, lw=1, ls="--")
    ax.text(y.median() + 0.5, ax.get_ylim()[1] * 0.92, f"median {y.median():.0f} min", fontsize=8, color=viz.TEXT_2)
    ax.set_xlabel("Delivery time (min)"); ax.set_ylabel("Orders")
    ax.set_title("Delivery time is right-skewed and multi-peaked")
    viz.save(fig, out / "target_distribution.png")

    # ------------------------------------------------------------ 2. five questions
    banner("2. The five EDA questions")
    multi, pairs, corr = [], [], []

    # Q1 distance
    corr.append(T.spearman(df, "distance_km", Y))
    multi.append(T.multi_group_test(df, "distance_bucket", Y))
    s_dist = T.group_summary(df, "distance_bucket", Y)
    pairs.append(T.two_group_test(df.loc[df.distance_bucket == ">10 km", Y],
                                  df.loc[df.distance_bucket == "0-3 km", Y], ">10 km", "0-3 km"))
    # Q2 traffic
    s_tr = T.group_summary(df.assign(traffic=df.traffic.fillna("Missing")), "traffic", Y)
    multi.append(T.multi_group_test(df, "traffic", Y))
    pairs.append(T.two_group_test(df.loc[df.traffic == "Jam", Y], df.loc[df.traffic == "Low", Y], "Jam", "Low"))
    # Q3 prep time
    s_prep = T.group_summary(df.dropna(subset=["pickup_delay_min"]), "pickup_delay_min", Y)
    multi.append(T.multi_group_test(df, "pickup_delay_min", Y))
    corr.append(T.spearman(df, "pickup_delay_min", Y))
    # Q4 peak hour, overall and within traffic level (confounding check)
    pairs.append(T.two_group_test(df.loc[df.is_peak_hour == 1, Y], df.loc[df.is_peak_hour == 0, Y],
                                  "peak", "off-peak"))
    # In this data traffic is a function of the clock hour, so some traffic
    # levels only ever occur at peak (Jam) -> the stratum cannot be compared.
    pd.crosstab(df["hour"], df["traffic"].fillna("Missing")).to_csv(out / "hour_x_traffic_crosstab.csv")
    strat = []
    for lvl, g in df.dropna(subset=["traffic"]).groupby("traffic"):
        a_, b_ = g.loc[g.is_peak_hour == 1, Y], g.loc[g.is_peak_hour == 0, Y]
        if min(len(a_), len(b_)) < 30:
            strat.append({"comparison": f"peak|{lvl} vs off-peak|{lvl}", "n_a": len(a_), "n_b": len(b_),
                          "note": "not estimable: this traffic level occurs almost only on one side"})
            continue
        strat.append(T.two_group_test(a_, b_, f"peak|{lvl}", f"off-peak|{lvl}"))
    strat = pd.DataFrame(strat)
    strat.to_csv(out / "peak_hour_within_traffic.csv", index=False)
    peak_traffic = T.chi_square(df.dropna(subset=["traffic"]), "is_peak_hour", "traffic")

    # other drivers
    multi.append(T.multi_group_test(df, "weather", Y))
    multi.append(T.multi_group_test(df, "city_type", Y))
    multi.append(T.multi_group_test(df, "multiple_deliveries", Y))
    multi.append(T.multi_group_test(df, "vehicle_condition", Y))
    multi.append(T.multi_group_test(df, "order_type", Y))
    pairs.append(T.two_group_test(df.loc[df.festival == "Yes", Y], df.loc[df.festival == "No", Y],
                                  "festival", "no festival"))
    for c in ["rider_age", "rider_rating", "vehicle_condition", "multiple_deliveries", "hour"]:
        corr.append(T.spearman(df, c, Y))

    multi = pd.DataFrame(multi)
    pairs = pd.DataFrame(pairs)
    corr = pd.DataFrame(corr)
    # Holm correction across every p-value reported in this section
    all_p = list(multi.p_kruskal) + list(pairs.p_welch) + list(corr.p_value)
    adj = T.holm(all_p)
    multi["p_kruskal_holm"] = adj[:len(multi)]
    pairs["p_welch_holm"] = adj[len(multi):len(multi) + len(pairs)]
    corr["p_holm"] = adj[len(multi) + len(pairs):]
    multi.to_csv(out / "tests_multi_group.csv", index=False)
    pairs.to_csv(out / "tests_two_group.csv", index=False)
    corr.to_csv(out / "tests_spearman.csv", index=False)
    for name, s in [("distance", s_dist), ("traffic", s_tr), ("prep", s_prep)]:
        s.to_csv(out / f"summary_by_{name}.csv", index=False)
    print(multi.round(4).to_string(index=False))
    print(pairs[["comparison", "n_a", "n_b", "diff", "diff_ci95_low", "diff_ci95_high",
                 "p_welch", "hedges_g"]].round(4).to_string(index=False))
    print(corr.round(4).to_string(index=False))
    print("\nPeak vs off-peak within traffic level:\n" +
          strat.reindex(columns=["comparison", "n_a", "n_b", "diff", "diff_ci95_low", "diff_ci95_high", "p_welch", "hedges_g", "note"])
          .round(3).to_string(index=False))
    print(f"peak hour x traffic: Cramer's V = {peak_traffic['cramers_v']:.3f}")

    # Q5 strongest relationships: mutual information (captures non-linear steps)
    feats = X[["rider_age", "rider_rating", "vehicle_condition", "multiple_deliveries",
               "distance_km", "order_hour", "day_of_week", "traffic_level"]].copy()
    for c in ["weather", "festival", "city_type", "order_type", "vehicle_type", "city_code"]:
        feats[c] = X[c].astype("category").cat.codes
    feats["pickup_delay_min"] = df["pickup_delay_min"]
    feats = feats.fillna(feats.median(numeric_only=True))
    disc = [c in ["weather", "festival", "city_type", "order_type", "vehicle_type", "city_code"]
            for c in feats.columns]
    mi = mutual_info_regression(feats, y, discrete_features=disc, random_state=config.RANDOM_STATE)
    mi = pd.Series(mi, index=feats.columns, name="mutual_information").sort_values(ascending=False)
    mi.to_csv(out / "mutual_information.csv")
    print("\nMutual information with delivery time:\n" + mi.round(3).to_string())

    fig, ax = viz.plt.subplots(1, 2, figsize=(11, 3.8))
    dot_whisker(ax[0], s_tr, "traffic", ["Low", "Medium", "High", "Jam", "Missing"],
                "Delivery time by traffic", viz.SERIES[0])
    s_w = T.group_summary(df.assign(weather=df.weather.fillna("Missing")), "weather", Y)
    wo = s_w.sort_values("mean").weather.tolist()
    dot_whisker(ax[1], s_w, "weather", wo, "Delivery time by weather", viz.SERIES[0])
    ax[1].set_ylabel("")
    viz.save(fig, out / "time_by_traffic_weather.png")

    # ------------------------------------------------------------ 3. age / rating
    banner("3. Why do rider age and rating dominate?")
    by_age = T.group_summary(df.dropna(subset=["rider_age"]), "rider_age", Y)
    by_age.to_csv(out / "summary_by_age.csv", index=False)
    df["rating_r"] = df["rider_rating"].round(1)
    by_rating = T.group_summary(df.dropna(subset=["rating_r"]), "rating_r", Y)
    by_rating.to_csv(out / "summary_by_rating.csv", index=False)
    steps = pd.DataFrame([
        T.within_band_slope(df, "rider_age", Y, 20, 29),
        T.within_band_slope(df, "rider_age", Y, 30, 39),
        T.within_band_slope(df, "rider_rating", Y, 4.5, 4.9),
        T.within_band_slope(df, "rider_rating", Y, 4.0, 4.4),
    ])
    jumps = pd.DataFrame([
        T.two_group_test(df.loc[df.rider_age == 30, Y], df.loc[df.rider_age == 29, Y], "age 30", "age 29"),
        T.two_group_test(df.loc[df.rider_age.between(30, 39), Y], df.loc[df.rider_age.between(20, 29), Y],
                         "age 30-39", "age 20-29"),
        T.two_group_test(df.loc[df.rider_rating.between(4.0, 4.44), Y], df.loc[df.rider_rating >= 4.45, Y],
                         "rating 4.0-4.4", "rating >= 4.5"),
    ])
    indep = pd.DataFrame([T.spearman(df, "rider_age", c) for c in ["distance_km", "rider_rating", "multiple_deliveries"]]
                         + [T.spearman(df, "rider_rating", c) for c in ["distance_km", "vehicle_condition"]])
    steps.to_csv(out / "age_rating_within_band_slopes.csv", index=False)
    jumps.to_csv(out / "age_rating_step_tests.csv", index=False)
    indep.to_csv(out / "age_rating_independence.csv", index=False)
    print(steps.round(4).to_string(index=False))
    print(jumps[["comparison", "n_a", "n_b", "diff", "diff_ci95_low", "diff_ci95_high", "hedges_g"]]
          .round(3).to_string(index=False))
    print(indep.round(3).to_string(index=False))

    fig, ax = viz.plt.subplots(1, 2, figsize=(11, 3.8))
    a = by_age[by_age.n >= 30]
    ax[0].errorbar(a.rider_age, a["mean"], yerr=[a["mean"] - a.ci95_low, a.ci95_high - a["mean"]],
                   fmt="o-", ms=5, color=viz.SERIES[0], lw=2, elinewidth=1.5)
    ax[0].set_xticks(range(20, 40, 2)); ax[0].set_xlabel("Rider age"); ax[0].set_ylabel("Mean delivery time (min), 95% CI")
    ax[0].set_title("A step at age 30, flat on either side")
    r = by_rating[by_rating.n >= 30]
    ax[1].errorbar(r.rating_r, r["mean"], yerr=[r["mean"] - r.ci95_low, r.ci95_high - r["mean"]],
                   fmt="o", ms=5, color=viz.SERIES[0], elinewidth=1.5)
    ax[1].set_xlabel("Rider rating (rounded to 0.1)")
    ax[1].set_title("Rating works in three bands")
    viz.save(fig, out / "age_rating_steps.png")

    # ------------------------------------------------------------ 4. OLS
    banner("4. Linear regression: coefficients and assumptions")
    cols = ["distance_km", "traffic_level", "rider_age", "rider_rating", "vehicle_condition",
            "multiple_deliveries", "is_peak_hour"]
    d = X[cols].join(df[[Y]]).dropna()
    d["festival_yes"] = df.loc[d.index, "festival"].eq("Yes").fillna(False).astype(float)
    d["age_30_plus"] = (d.rider_age >= 30).astype(float)
    d["rating_ge_4_5"] = (d.rider_rating >= 4.5).astype(float)
    specs = {
        "linear_age_rating": cols + ["festival_yes"],
        "step_age_rating": [c for c in cols if c not in ("rider_age", "rider_rating")]
                           + ["age_30_plus", "rating_ge_4_5", "festival_yes"],
    }
    ols_rows, coef_tables = [], []
    for name, xs in specs.items():
        Xo = sm.add_constant(d[xs])
        fit = sm.OLS(d[Y], Xo).fit(cov_type="HC3")   # heteroscedasticity-robust SEs
        bp = het_breuschpagan(fit.resid, Xo)
        ols_rows.append({"spec": name, "n": int(fit.nobs), "r2": fit.rsquared, "adj_r2": fit.rsquared_adj,
                         "aic": fit.aic, "breusch_pagan_p": bp[1],
                         "resid_skew": sps.skew(fit.resid), "resid_mae": float(np.mean(np.abs(fit.resid)))})
        ct = pd.DataFrame({"coef": fit.params, "ci95_low": fit.conf_int()[0],
                           "ci95_high": fit.conf_int()[1], "p_value": fit.pvalues})
        ct["spec"] = name
        coef_tables.append(ct)
    ols = pd.DataFrame(ols_rows)
    coefs = pd.concat(coef_tables).reset_index(names="term")
    ols.to_csv(out / "ols_fit.csv", index=False)
    coefs.to_csv(out / "ols_coefficients.csv", index=False)
    print(ols.round(4).to_string(index=False))
    print(coefs.round(3).to_string(index=False))

    Xv = sm.add_constant(X[["distance_km", "traffic_level", "distance_x_traffic"]].dropna())
    vif = pd.Series([variance_inflation_factor(Xv.values, i) for i in range(1, Xv.shape[1])],
                    index=Xv.columns[1:], name="VIF")
    vif.to_csv(out / "vif_distance_traffic.csv")
    print("\nVIF (distance, traffic, interaction):\n" + vif.round(2).to_string())
    print(f"\nSaved outputs to {out}")


if __name__ == "__main__":
    main()
