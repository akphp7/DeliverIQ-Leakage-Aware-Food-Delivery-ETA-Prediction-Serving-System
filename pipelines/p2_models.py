"""Phase 2 - tuning, RF+XGB blending, prediction intervals, late-vs-early trade-off, SHAP.

Run:  python -m pipelines.p2_models

Protocol (date-based, nothing chosen on test data):
    train_core  : all days before the validation block
    validation  : last 4 training days  -> tuning, blend weights, interval calibration
    test        : last 9 days (same test set as Phase 1)

Outputs: outputs/v2/phase2/*  and  models/v2/final_bundle.joblib
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler

from deliveriq import config
from deliveriq.data.cleaning import city_reference, load_clean
from deliveriq.evaluation.metrics import regression_metrics
from deliveriq.evaluation.significance import block_bootstrap_by_day, paired_mae_test
from deliveriq.evaluation.splits import temporal_split
from deliveriq.features.build import FINAL_DROP, feature_columns
from deliveriq.models.ensemble import StackedBlend, WeightedBlend, best_weight
from deliveriq.models.explain import GroupedShap
from deliveriq.models.factory import make_model, xgb_params
from deliveriq.models.intervals import CQR, SplitConformal, coverage_report, quantile_model
from deliveriq import viz
from pipelines.common import Timer, banner, distance_bucket, prepare, save_json

VAL_DAYS = 4
SEARCH_SPACE = {
    "n_estimators": [300, 500, 800],
    "learning_rate": [0.03, 0.05, 0.08],
    "max_depth": [4, 5, 6, 7, 8],
    "min_child_weight": [1, 3, 5, 10],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_lambda": [1.0, 5.0, 10.0],
    "objective": ["reg:squarederror", "reg:absoluteerror", "reg:pseudohubererror"],
}
N_SEARCH = 20


def split_indices(clean):
    tr, te, cutoff, n_test_days = temporal_split(clean)
    days = np.sort(clean.iloc[tr]["order_date"].dt.normalize().unique())
    val_start = days[-VAL_DAYS]
    d = clean["order_date"].dt.normalize().to_numpy()
    core = tr[d[tr] < val_start]
    val = tr[d[tr] >= val_start]
    return core, val, tr, te, pd.Timestamp(val_start), cutoff


def main():
    out = config.phase_dir(2)
    viz.style()
    clean, _ = load_clean()
    num, cat = feature_columns(FINAL_DROP)
    core, val, train, test, val_start, cutoff = split_indices(clean)
    print(f"core {len(core)} | validation {len(val)} (from {val_start.date()}) | "
          f"train {len(train)} | test {len(test)} (from {cutoff.date()})")

    d_cv = prepare((core, val), clean)       # fit on core, evaluate on validation
    d_ct = prepare((core, test), clean)      # core-trained models, scored on test (intervals)
    d_ft = prepare((train, test), clean)     # full-train models, scored on test

    # ------------------------------------------------------------ 1. tuning
    banner("1. XGBoost random search on the validation block")
    rows = []
    for i, p in enumerate(ParameterSampler(SEARCH_SPACE, N_SEARCH, random_state=config.RANDOM_STATE)):
        with Timer() as t:
            m = make_model("XGBoost", num, cat, **p).fit(d_cv["X_tr"], d_cv["y_tr"])
        r = regression_metrics(d_cv["y_te"], m.predict(d_cv["X_te"]))
        rows.append({"trial": i, **p, "val_MAE": r["MAE"], "val_RMSE": r["RMSE"], "seconds": round(t.s, 1)})
    default = make_model("XGBoost", num, cat).fit(d_cv["X_tr"], d_cv["y_tr"])
    default_mae = regression_metrics(d_cv["y_te"], default.predict(d_cv["X_te"]))["MAE"]
    search = pd.DataFrame(rows).sort_values("val_MAE")
    search.to_csv(out / "xgb_search.csv", index=False)
    print(search.head(8).round(3).to_string(index=False))
    best = search.iloc[0][list(SEARCH_SPACE)].to_dict()
    for k in ["n_estimators", "max_depth", "min_child_weight"]:
        best[k] = int(best[k])
    print(f"\nv1 default params val MAE {default_mae:.3f} -> best {search.iloc[0].val_MAE:.3f}\nbest: {best}")

    # ------------------------------------------------------------ 2. blend
    banner("2. RF + XGBoost blending (weights chosen on validation)")
    rf_c = make_model("Random Forest", num, cat).fit(d_cv["X_tr"], d_cv["y_tr"])
    xgb_c = make_model("XGBoost", num, cat, **best).fit(d_cv["X_tr"], d_cv["y_tr"])
    pv_rf, pv_xgb = rf_c.predict(d_cv["X_te"]), xgb_c.predict(d_cv["X_te"])
    w_rf, curve = best_weight(d_cv["y_te"], pv_rf, pv_xgb)
    pd.DataFrame(curve, columns=["w_rf", "val_MAE"]).to_csv(out / "blend_weight_curve.csv", index=False)
    stack = StackedBlend().fit(d_cv["y_te"], pv_rf, pv_xgb)
    print(f"weighted blend: w_rf = {w_rf}; stacking coefs = {stack.coefs}")

    rf_f = make_model("Random Forest", num, cat).fit(d_ft["X_tr"], d_ft["y_tr"])
    xgb_def_f = make_model("XGBoost", num, cat).fit(d_ft["X_tr"], d_ft["y_tr"])
    xgb_f = make_model("XGBoost", num, cat, **best).fit(d_ft["X_tr"], d_ft["y_tr"])
    lr_f = make_model("Linear Regression", num, cat).fit(d_ft["X_tr"], d_ft["y_tr"])
    y = d_ft["y_te"]
    P = {
        "Linear Regression": lr_f.predict(d_ft["X_te"]),
        "Random Forest": rf_f.predict(d_ft["X_te"]),
        "XGBoost (v1 params)": xgb_def_f.predict(d_ft["X_te"]),
        "XGBoost (tuned)": xgb_f.predict(d_ft["X_te"]),
    }
    P["Weighted blend"] = WeightedBlend(w_rf).predict(P["Random Forest"], P["XGBoost (tuned)"])
    P["Stacked blend"] = stack.predict(P["Random Forest"], P["XGBoost (tuned)"])
    final_cmp = pd.DataFrame([{"Model": k, **regression_metrics(y, v)} for k, v in P.items()])
    final_cmp.to_csv(out / "final_model_comparison.csv", index=False)
    print(final_cmp[["Model", "MAE", "RMSE", "R2", "P90_AE", "within_5min_pct",
                     "late_gt5_pct", "early_gt5_pct"]].round(3).to_string(index=False))

    days = d_ft["c_te"]["order_date"].dt.date.astype(str).to_numpy()
    sig = []
    for a, b in [("Weighted blend", "XGBoost (tuned)"), ("Weighted blend", "Random Forest"),
                 ("XGBoost (tuned)", "Random Forest"), ("XGBoost (tuned)", "XGBoost (v1 params)")]:
        r = paired_mae_test(y, P[a], P[b])
        blk = block_bootstrap_by_day(y, P[a], P[b], days)
        sig.append({"A": a, "B": b, **r, "day_ci_low": blk["ci95_low"], "day_ci_high": blk["ci95_high"]})
    sig = pd.DataFrame(sig)
    sig.to_csv(out / "significance.csv", index=False)
    print(sig[["A", "B", "diff_A_minus_B", "ci95_low", "ci95_high", "p_wilcoxon",
               "day_ci_low", "day_ci_high"]].round(4).to_string(index=False))

    # ------------------------------------------------------------ 3. intervals
    banner("3. Prediction intervals (calibrated on validation, scored on test)")
    xgb_ct_val = xgb_c.predict(d_cv["X_te"])
    xgb_ct_test = xgb_c.predict(d_ct["X_te"])
    iv_rows, cqr_models = [], {}
    tuned_tree = {k: v for k, v in best.items() if k != "objective"}
    for alpha in [0.2, 0.1]:
        sc = SplitConformal(alpha).calibrate(d_cv["y_te"], xgb_ct_val)
        lo, hi = sc.interval(xgb_ct_test)
        iv_rows.append({"method": "split conformal", "target_coverage_pct": (1 - alpha) * 100,
                        "q_min": sc.q_, **coverage_report(d_ct["y_te"], lo, hi)})
        cqr = CQR(num, cat, alpha, **tuned_tree).fit(d_cv["X_tr"], d_cv["y_tr"]).calibrate(d_cv["X_te"], d_cv["y_te"])
        lo, hi = cqr.interval(d_ct["X_te"])
        iv_rows.append({"method": "CQR (XGB quantile)", "target_coverage_pct": (1 - alpha) * 100,
                        "q_min": cqr.q_, **coverage_report(d_ct["y_te"], lo, hi)})
        cqr_models[alpha] = (cqr, sc, lo, hi)
    iv = pd.DataFrame(iv_rows)
    iv.to_csv(out / "interval_coverage.csv", index=False)
    print(iv.round(2).to_string(index=False))

    cqr80, sc80, lo80, hi80 = cqr_models[0.2]
    slo, shi = sc80.interval(xgb_ct_test)
    seg = pd.DataFrame({"traffic": d_ct["c_te"]["traffic"].fillna("Missing").to_numpy(),
                        "distance_bucket": distance_bucket(d_ct["c_te"]["distance_km"]).astype(str).to_numpy(),
                        "y": d_ct["y_te"], "cqr_lo": lo80, "cqr_hi": hi80, "sc_lo": slo, "sc_hi": shi})
    seg_rows = []
    for col in ["traffic", "distance_bucket"]:
        for key, g in seg.groupby(col):
            seg_rows.append({"segment": col, "value": key, "n": len(g),
                             "split_cov_pct": ((g.y >= g.sc_lo) & (g.y <= g.sc_hi)).mean() * 100,
                             "split_width": (g.sc_hi - g.sc_lo).mean(),
                             "cqr_cov_pct": ((g.y >= g.cqr_lo) & (g.y <= g.cqr_hi)).mean() * 100,
                             "cqr_width": (g.cqr_hi - g.cqr_lo).mean()})
    seg_iv = pd.DataFrame(seg_rows)
    seg_iv.to_csv(out / "interval_coverage_by_segment.csv", index=False)
    print("\n80% intervals by segment\n" + seg_iv.round(2).to_string(index=False))

    order = ["Low", "Medium", "High", "Jam", "Missing"]
    s = seg_iv[seg_iv.segment == "traffic"].set_index("value").reindex(order)
    fig, ax = viz.plt.subplots(1, 2, figsize=(10, 3.8))
    x = np.arange(len(order))
    ax[0].bar(x - 0.2, s.split_width, 0.38, color=viz.SERIES[0], label="Split conformal")
    ax[0].bar(x + 0.2, s.cqr_width, 0.38, color=viz.SERIES[1], label="CQR")
    ax[0].set_xticks(x, order); ax[0].set_ylabel("Mean interval width (min)")
    ax[0].set_title("80% interval width by traffic")
    ax[0].legend(loc="upper left")
    ax[1].bar(x - 0.2, s.split_cov_pct, 0.38, color=viz.SERIES[0], label="Split conformal")
    ax[1].bar(x + 0.2, s.cqr_cov_pct, 0.38, color=viz.SERIES[1], label="CQR")
    ax[1].axhline(80, color=viz.NEUTRAL, lw=1, ls="--")
    ax[1].text(len(order) - 0.5, 81, "target 80%", color=viz.TEXT_2, fontsize=8, ha="right")
    ax[1].set_xticks(x, order); ax[1].set_ylim(40, 100); ax[1].set_ylabel("Coverage (%)")
    ax[1].set_title("80% interval coverage by traffic")
    viz.save(fig, out / "interval_by_traffic.png")

    # ------------------------------------------------------------ 4. late vs early
    banner("4. Late vs early trade-off (predict a higher quantile)")
    tr_rows = []
    for q in [0.5, 0.6, 0.7, 0.8]:
        qm = quantile_model(num, cat, q, **tuned_tree).fit(d_ft["X_tr"], d_ft["y_tr"])
        r = regression_metrics(y, qm.predict(d_ft["X_te"]))
        tr_rows.append({"approach": f"quantile q={q}", **r})
    for b in [1, 2, 3]:
        r = regression_metrics(y, P["XGBoost (tuned)"] + b)
        tr_rows.append({"approach": f"tuned XGB + {b} min buffer", **r})
    trade = pd.DataFrame(tr_rows)
    trade.to_csv(out / "late_vs_early_tradeoff.csv", index=False)
    print(trade[["approach", "MAE", "late_gt5_pct", "early_gt5_pct", "mean_bias",
                 "within_5min_pct"]].round(2).to_string(index=False))

    # ------------------------------------------------------------ 5. SHAP
    banner("5. SHAP on the final XGBoost (full train)")
    gs = GroupedShap(xgb_f, cat)
    sample = d_ft["X_te"].sample(2000, random_state=config.RANDOM_STATE)
    g = gs.grouped(sample)
    imp = g.abs().mean().sort_values(ascending=False)
    imp.rename("mean_abs_shap_min").to_csv(out / "shap_global_grouped.csv")
    print(imp.round(3).to_string())

    top = imp.head(10)[::-1]
    fig, ax = viz.plt.subplots(figsize=(7, 4.2))
    ax.barh(top.index, top.values, color=viz.SERIES[0], height=0.6)
    for yi, v in enumerate(top.values):
        ax.text(v + 0.03, yi, f"{v:.2f}", va="center", fontsize=8, color=viz.TEXT_2)
    ax.set_xlabel("Mean |SHAP| (minutes of predicted ETA)")
    ax.set_title("What drives the final model's ETA (2,000 test orders)")
    ax.grid(axis="y", visible=False)
    viz.save(fig, out / "shap_global_grouped.png")

    import shap
    sv = gs.raw_values(sample)
    viz.plt.figure()
    shap.summary_plot(sv, gs.pre.transform(sample),
                      feature_names=gs.pre.get_feature_names_out(), show=False, max_display=15)
    viz.plt.gcf().savefig(out / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    viz.plt.close("all")

    # local explanations: the worst under-prediction and a typical order
    err = y - P["XGBoost (tuned)"]
    picks = {"typical": int(np.argsort(np.abs(err))[len(err) // 2]),
             "worst_late": int(np.argmax(err))}
    local_rows = []
    for label, i in picks.items():
        xi = d_ft["X_te"].iloc[[i]]
        contrib = gs.grouped(xi).iloc[0].sort_values(key=np.abs, ascending=False)
        for f, v in contrib.head(6).items():
            local_rows.append({"case": label, "order_id": d_ft["c_te"]["order_id"].iloc[i],
                               "actual": y[i], "predicted": float(P["XGBoost (tuned)"][i]),
                               "base_value": gs.base_value, "factor": f, "shap_min": float(v)})
    local = pd.DataFrame(local_rows)
    local.to_csv(out / "shap_local_examples.csv", index=False)
    print(local.round(2).to_string(index=False))

    # ------------------------------------------------------------ save bundle
    mdir = config.model_dir()
    bundle = {
        "version": "v2",
        "point_model": xgb_f, "point_model_name": "XGBoost (tuned)",
        "xgb_params": best, "numeric": num, "categorical": cat,
        "cqr80": cqr80, "split_conformal_q80": sc80.q_,
        "city_reference": city_reference(clean),
        "train_period": [str(clean.iloc[train].order_date.min().date()),
                         str(clean.iloc[train].order_date.max().date())],
        "test_metrics": final_cmp.set_index("Model").loc["XGBoost (tuned)"].to_dict(),
    }
    joblib.dump(bundle, mdir / "final_bundle.joblib", compress=3)
    save_json({k: v for k, v in bundle.items() if k not in ("point_model", "cqr80", "city_reference")}
              | {"blend_w_rf": w_rf, "stack": stack.coefs,
                 "interval_coverage": iv.to_dict(orient="records")},
              mdir / "final_metadata.json")
    print(f"\nSaved bundle + outputs ({out})")


if __name__ == "__main__":
    main()
