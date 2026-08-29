"""Phase 4 - SQL analytics and SQL-built history features.

Run:  python -m pipelines.p4_sql_features

1. Runs the analytical SQL files (KPIs, hour profile, leaderboard, daily trend).
2. Builds rider / restaurant history features in SQL (previous days only).
3. Verifies in pandas that the SQL features never look at the same or later days.
4. Tests whether the features improve the tuned XGBoost (date-based test +
   rolling-origin time CV + paired significance test).

Outputs: outputs/v2/phase4/*.csv
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from deliveriq import config
from deliveriq.data.cleaning import load_clean
from deliveriq.evaluation.metrics import regression_metrics
from deliveriq.evaluation.significance import block_bootstrap_by_day, paired_mae_test
from deliveriq.evaluation.splits import rolling_origin_splits, temporal_split
from deliveriq.features.build import FINAL_DROP, base_features, feature_columns
from deliveriq.models.factory import make_model
from deliveriq.sqlfeatures import duck
from pipelines.common import banner


def leakage_check(clean: pd.DataFrame, hist: pd.DataFrame, n: int = 300) -> pd.DataFrame:
    """Recompute rider_prev_orders / avg in pandas for random orders."""
    rng = np.random.default_rng(config.RANDOM_STATE)
    rows = []
    day = clean["order_date"].dt.normalize()
    for i in rng.choice(len(clean), n, replace=False):
        r = clean.iloc[i]
        past = clean[(clean.rider_id == r.rider_id) & (day < day.iloc[i])]
        exp_avg = past[config.TARGET].mean() if len(past) else np.nan
        got_n = hist.iloc[i]["rider_prev_orders"]
        got_avg = hist.iloc[i]["rider_prev_avg_minutes"]
        rows.append({"row": int(i), "expected_prev_orders": len(past), "sql_prev_orders": got_n,
                     "expected_prev_avg": exp_avg, "sql_prev_avg": got_avg})
    chk = pd.DataFrame(rows)
    chk["match"] = (chk.expected_prev_orders == chk.sql_prev_orders) & (
        np.isclose(chk.expected_prev_avg, chk.sql_prev_avg, equal_nan=True))
    return chk


def main():
    out = config.phase_dir(4)
    clean, _ = load_clean()

    banner("1. Analytical SQL")
    con = duck.connect(clean)
    for f in ["01_city_kpis.sql", "02_hour_traffic_profile.sql",
              "05_rider_leaderboard.sql", "06_daily_trend.sql"]:
        res = duck.run(con, f)
        res.to_csv(out / f.replace(".sql", ".csv"), index=False)
        print(f"\n-- {f} ({len(res)} rows)\n" + res.head(8).to_string(index=False))
    con.close()

    banner("2-3. SQL history features + leakage check")
    hist = duck.history_features(clean)
    hist.describe().T.to_csv(out / "sql_feature_summary.csv")
    print(hist.describe().T.round(2).to_string())
    chk = leakage_check(clean, hist)
    chk.to_csv(out / "sql_leakage_check.csv", index=False)
    print(f"\npandas re-computation matches SQL for {chk.match.sum()}/{len(chk)} sampled orders")
    assert chk.match.all(), "SQL history features disagree with the pandas check"

    banner("4. Do the SQL features help the tuned XGBoost?")
    params = joblib.load(config.MODEL_DIR / "final_bundle.joblib")["xgb_params"]
    num, cat = feature_columns(FINAL_DROP)
    X_all = base_features(clean).join(hist)
    y_all = clean[config.TARGET].to_numpy(float)
    variants = {
        "base": num,
        "base + rider history": num + ["rider_prev_orders", "rider_prev_avg_minutes",
                                       "rider_prev_slow_pct", "rider_days_since_active"],
        "base + restaurant history": num + ["rest_orders_prev7d", "rest_avg_minutes_prev7d"],
        "base + all SQL features": num + duck.SQL_HISTORY_COLUMNS,
    }

    tr, te, cutoff, _ = temporal_split(clean)
    rows, preds = [], {}
    for name, cols in variants.items():
        m = make_model("XGBoost", cols, cat, **params).fit(X_all.iloc[tr], y_all[tr])
        p = m.predict(X_all.iloc[te])
        preds[name] = p
        rows.append({"variant": name, **regression_metrics(y_all[te], p)})
    test_cmp = pd.DataFrame(rows)
    test_cmp.to_csv(out / "sql_features_test.csv", index=False)
    print(test_cmp[["variant", "MAE", "RMSE", "R2", "P90_AE", "late_gt5_pct"]].round(3).to_string(index=False))

    days = clean.iloc[te]["order_date"].dt.date.astype(str).to_numpy()
    sig = []
    for name in list(variants)[1:]:
        r = paired_mae_test(y_all[te], preds[name], preds["base"])
        b = block_bootstrap_by_day(y_all[te], preds[name], preds["base"], days)
        sig.append({"A": name, "B": "base", **r, "day_ci_low": b["ci95_low"], "day_ci_high": b["ci95_high"]})
    sig = pd.DataFrame(sig)
    sig.to_csv(out / "sql_features_significance.csv", index=False)
    print(sig[["A", "diff_A_minus_B", "ci95_low", "ci95_high", "p_wilcoxon",
               "day_ci_low", "day_ci_high"]].round(4).to_string(index=False))

    ts = []
    for k, (tr_k, te_k, d0, d1) in enumerate(rolling_origin_splits(clean)):
        for name, cols in variants.items():
            m = make_model("XGBoost", cols, cat, **params).fit(X_all.iloc[tr_k], y_all[tr_k])
            ts.append({"fold": k, "test_from": d0.date(), "variant": name,
                       "MAE": regression_metrics(y_all[te_k], m.predict(X_all.iloc[te_k]))["MAE"]})
    ts = pd.DataFrame(ts).pivot(index=["fold", "test_from"], columns="variant", values="MAE")
    ts = ts[list(variants)]
    ts.to_csv(out / "sql_features_time_cv.csv")
    print("\nRolling time CV MAE:\n" + ts.round(3).to_string())
    print("\nmean:\n" + ts.mean().round(4).to_string())

    # Feature importance of the SQL features in the 'all' model (gain)
    m = make_model("XGBoost", variants["base + all SQL features"], cat, **params).fit(X_all.iloc[tr], y_all[tr])
    names = m.named_steps["pre"].get_feature_names_out()
    gain = pd.Series(m.named_steps["model"].feature_importances_, index=names).sort_values(ascending=False)
    gain.to_csv(out / "feature_importance_with_sql.csv", header=["importance"])
    rank = {n: i + 1 for i, n in enumerate(gain.index)}
    print("\nImportance rank of SQL features (of %d columns):" % len(gain))
    for c in duck.SQL_HISTORY_COLUMNS:
        print(f"  {c:28s} rank {rank['num__' + c]:3d}  importance {gain['num__' + c]:.4f}")
    print(f"\nSaved outputs to {out}")


if __name__ == "__main__":
    main()
