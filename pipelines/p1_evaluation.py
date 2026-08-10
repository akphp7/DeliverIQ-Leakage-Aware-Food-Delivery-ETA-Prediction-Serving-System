"""Phase 1 - evaluation you can defend.

Run:  python -m pipelines.p1_evaluation

Answers four evaluation questions with numbers:
  1. Does a random split flatter the model compared with a date-based split?
  2. How stable are the scores?  (5-fold CV and rolling-origin time CV)
  3. Is RF really better than XGBoost, or is the gap noise?  (paired tests)
  4. Which feature groups actually matter?  (ablation with CV)

Outputs: outputs/v2/phase1/*.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from deliveriq import config
from deliveriq.data.cleaning import load_clean
from deliveriq.evaluation.metrics import regression_metrics
from deliveriq.evaluation.significance import block_bootstrap_by_day, paired_mae_test
from deliveriq.evaluation.splits import (kfold_splits, random_split,
                                         rolling_origin_splits, temporal_split)
from deliveriq.features.build import feature_columns
from deliveriq.models.factory import BASE_MODELS, make_model
from pipelines.common import banner, prepare

ABLATIONS = {
    "full": [],
    "no_restaurant_history": ["hist_prep_mean", "hist_restaurant_orders"],
    "no_city_code": ["city_code"],
    "no_missing_flags": ["traffic_missing", "weather_missing", "rider_profile_invalid",
                         "order_time_imputed", "restaurant_coords_missing"],
    "no_distance_x_traffic": ["distance_x_traffic"],
    "no_cyclic_hour": ["hour_sin", "hour_cos"],
    "no_rider_age_rating": ["rider_age", "rider_rating"],
    "no_distance": ["distance_km", "distance_x_traffic"],
}


def fit_eval(name, d, drop=()):
    num, cat = feature_columns(list(drop))
    m = make_model(name, num, cat).fit(d["X_tr"], d["y_tr"])
    pred = m.predict(d["X_te"])
    return m, pred, regression_metrics(d["y_te"], pred)


def main():
    out = config.phase_dir(1)
    clean, _ = load_clean()

    # ------------------------------------------------------------------ 1
    banner("1. Random split vs date-based split")
    rs = random_split(clean)
    tr, te, cutoff, n_days = temporal_split(clean)
    print(f"Temporal: train {len(tr)} orders before {cutoff.date()}, "
          f"test {len(te)} orders on the last {n_days} days")
    rows, preds = [], {}
    for split_name, idx in [("random", rs), ("temporal", (tr, te))]:
        d = prepare(idx, clean)
        for name in BASE_MODELS:
            _, pred, r = fit_eval(name, d)
            r.update(split=split_name, Model=name, n_test=len(d["y_te"]))
            rows.append(r)
            preds[(split_name, name)] = (d, pred)
    split_cmp = pd.DataFrame(rows)
    split_cmp.to_csv(out / "random_vs_temporal.csv", index=False)
    print(split_cmp[["split", "Model", "MAE", "RMSE", "R2", "P90_AE",
                     "within_5min_pct", "late_gt5_pct"]].round(3).to_string(index=False))

    # ------------------------------------------------------------------ 2
    banner("2a. 5-fold cross-validation (random folds)")
    cv_rows = []
    for k, idx in enumerate(kfold_splits(clean)):
        d = prepare(idx, clean)
        for name in BASE_MODELS:
            _, _, r = fit_eval(name, d)
            cv_rows.append({"fold": k, "Model": name, **r})
    cv = pd.DataFrame(cv_rows)
    cv.to_csv(out / "cv5_folds.csv", index=False)
    cv_sum = cv.groupby("Model")[["MAE", "RMSE", "R2"]].agg(["mean", "std"])
    cv_sum.to_csv(out / "cv5_summary.csv")
    print(cv_sum.round(3).to_string())

    banner("2b. Rolling-origin time-series CV (4 folds x 4 days)")
    ts_rows = []
    for k, (tr_k, te_k, d0, d1) in enumerate(rolling_origin_splits(clean)):
        d = prepare((tr_k, te_k), clean)
        for name in ["Random Forest", "XGBoost"]:
            _, _, r = fit_eval(name, d)
            ts_rows.append({"fold": k, "test_from": d0.date(), "test_to": d1.date(),
                            "n_train": len(tr_k), "n_test": len(te_k), "Model": name, **r})
    ts = pd.DataFrame(ts_rows)
    ts.to_csv(out / "time_cv_folds.csv", index=False)
    print(ts[["fold", "test_from", "test_to", "n_train", "Model", "MAE", "RMSE", "R2"]]
          .round(3).to_string(index=False))
    ts_sum = ts.groupby("Model")[["MAE", "RMSE", "R2"]].agg(["mean", "std"])
    ts_sum.to_csv(out / "time_cv_summary.csv")
    print(ts_sum.round(3).to_string())

    # ------------------------------------------------------------------ 3
    banner("3. Is the difference between models real? (paired tests)")
    sig_rows = []
    for split_name in ["random", "temporal"]:
        d, p_rf = preds[(split_name, "Random Forest")]
        _, p_xgb = preds[(split_name, "XGBoost")]
        _, p_lr = preds[(split_name, "Linear Regression")]
        for a, b, pa, pb in [("Random Forest", "XGBoost", p_rf, p_xgb),
                             ("XGBoost", "Linear Regression", p_xgb, p_lr)]:
            r = paired_mae_test(d["y_te"], pa, pb)
            days = d["c_te"]["order_date"].dt.date.astype(str).to_numpy()
            blk = block_bootstrap_by_day(d["y_te"], pa, pb, days)
            r.update(split=split_name, A=a, B=b,
                     day_block_ci_low=blk["ci95_low"], day_block_ci_high=blk["ci95_high"])
            sig_rows.append(r)
    sig = pd.DataFrame(sig_rows)
    sig.to_csv(out / "significance.csv", index=False)
    print(sig[["split", "A", "B", "diff_A_minus_B", "ci95_low", "ci95_high",
               "p_bootstrap", "p_wilcoxon", "day_block_ci_low", "day_block_ci_high"]]
          .round(4).to_string(index=False))

    # ------------------------------------------------------------------ 4
    banner("4. Feature-group ablation (XGBoost, 5-fold CV)")
    folds = [prepare(idx, clean) for idx in kfold_splits(clean)]
    ab_rows = []
    for variant, drop in ABLATIONS.items():
        maes = [fit_eval("XGBoost", d, drop)[2]["MAE"] for d in folds]
        ab_rows.append({"variant": variant, "dropped": ", ".join(drop) or "-",
                        "cv_MAE_mean": np.mean(maes), "cv_MAE_std": np.std(maes, ddof=1)})
    ab = pd.DataFrame(ab_rows)
    base = ab.loc[ab.variant == "full", "cv_MAE_mean"].iloc[0]
    ab["delta_vs_full"] = ab["cv_MAE_mean"] - base
    ab.to_csv(out / "ablation.csv", index=False)
    print(ab.round(4).to_string(index=False))
    final_feature_check(clean, folds)
    print(f"\nSaved outputs to {out}")


def final_feature_check(clean, folds=None):
    """5. Per-fold check of the final feature-set decision."""
    banner("5. Final feature set check (per fold, 5-fold CV)")
    out = config.phase_dir(1)
    folds = folds or [prepare(idx, clean) for idx in kfold_splits(clean)]
    H = ["hist_prep_mean", "hist_restaurant_orders"]
    variants = {
        "xgb_full": ("XGBoost", []),
        "xgb_no_history": ("XGBoost", H),
        "xgb_no_history_no_city": ("XGBoost", H + ["city_code"]),
        "xgb_no_history_no_dxt": ("XGBoost", H + ["distance_x_traffic"]),
        "rf_full": ("Random Forest", []),
        "rf_no_history": ("Random Forest", H),
    }
    res = pd.DataFrame({k: [fit_eval(m, d, dr)[2]["MAE"] for d in folds]
                        for k, (m, dr) in variants.items()})
    res.index.name = "fold"
    res.to_csv(out / "final_feature_check.csv")
    print(res.round(3).to_string())
    print("\nmean MAE:\n" + res.mean().round(4).to_string())


if __name__ == "__main__":
    import sys
    if "--final-check-only" in sys.argv:
        final_feature_check(load_clean()[0])
    else:
        main()
