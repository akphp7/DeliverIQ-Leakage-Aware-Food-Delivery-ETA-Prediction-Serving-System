"""Phase 0 - data fixes and the v2 baseline.

Run from the project root:
    python -m pipelines.p0_data_fixes

Produces (outputs/v2/phase0/):
    data_audit.csv            what the cleaning repaired
    model_comparison.csv      LR / RF / XGB on the v1 random split, v2 data
    v1_vs_v2.csv              side-by-side with v1's saved numbers
    leakage_demo.csv          v1-style (leaky) vs out-of-fold history feature
    test_predictions.csv      v2 XGBoost predictions on the test set
    error_by_<segment>.csv    segment-wise errors
and models/v2/phase0_xgb.joblib (+ metadata json).
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from deliveriq import config
from deliveriq.data.cleaning import load_clean
from deliveriq.evaluation.metrics import regression_metrics, segment_errors
from deliveriq.evaluation.splits import random_split
from deliveriq.features.build import feature_columns
from deliveriq.models.factory import BASE_MODELS, make_model
from pipelines.common import Timer, banner, distance_bucket, prepare, save_json


def main():
    out = config.phase_dir(0)
    clean, audit = load_clean()

    banner("Data audit")
    audit_df = pd.DataFrame(list(audit.items()), columns=["check", "rows"])
    audit_df.to_csv(out / "data_audit.csv", index=False)
    print(audit_df.to_string(index=False))

    split = random_split(clean)
    d = prepare(split, clean)
    num, cat = feature_columns()

    banner("Models on the v1 random split (v2 data + features)")
    rows, fitted = [], {}
    for name in BASE_MODELS:
        with Timer() as t:
            m = make_model(name, num, cat).fit(d["X_tr"], d["y_tr"])
        pred = m.predict(d["X_te"])
        r = regression_metrics(d["y_te"], pred)
        r.update(Model=name, train_MAE=regression_metrics(d["y_tr"], m.predict(d["X_tr"]))["MAE"],
                 fit_seconds=round(t.s, 1))
        rows.append(r)
        fitted[name] = (m, pred)
    comp = pd.DataFrame(rows).set_index("Model")
    comp.to_csv(out / "model_comparison.csv")
    print(comp.round(3).to_string())

    # ---- v1 vs v2 -----------------------------------------------------------
    v1_path = config.ROOT / "data" / "reference" / "v1_model_comparison.csv"  # v1 scores, kept for comparison
    if v1_path.exists():
        v1 = pd.read_csv(v1_path).set_index("Model")
        both = pd.concat({"v1": v1[["MAE", "RMSE", "R2"]],
                          "v2": comp[["MAE", "RMSE", "R2"]]}, axis=1)
        both.to_csv(out / "v1_vs_v2.csv")
        banner("v1 vs v2 (same random split)")
        print(both.round(3).to_string())

    # ---- leakage demo: v1-style history vs out-of-fold ------------------------
    banner("Leakage demo: restaurant prep history (XGBoost)")
    demo = []
    for mode in ["leaky", "oof"]:
        dm = prepare(split, clean, history_mode=mode)
        m = make_model("XGBoost", num, cat).fit(dm["X_tr"], dm["y_tr"])
        demo.append({
            "history_mode": mode,
            "train_MAE": regression_metrics(dm["y_tr"], m.predict(dm["X_tr"]))["MAE"],
            "test_MAE": regression_metrics(dm["y_te"], m.predict(dm["X_te"]))["MAE"],
            "corr_feature_vs_own_prep": float(np.corrcoef(
                dm["X_tr"]["hist_prep_mean"],
                dm["c_tr"]["pickup_delay_min"].fillna(dm["c_tr"]["pickup_delay_min"].mean()))[0, 1]),
        })
    demo = pd.DataFrame(demo)
    demo.to_csv(out / "leakage_demo.csv", index=False)
    print(demo.round(4).to_string(index=False))

    # ---- predictions + segment errors for the saved XGBoost -----------------
    xgb, pred = fitted["XGBoost"]
    c_te = d["c_te"]
    preds = pd.DataFrame({
        "order_id": c_te["order_id"].to_numpy(),
        "order_date": c_te["order_date"].dt.date.to_numpy(),
        "actual": d["y_te"], "predicted": pred,
        "traffic": c_te["traffic"].fillna("Missing").to_numpy(),
        "weather": c_te["weather"].fillna("Missing").to_numpy(),
        "vehicle": c_te["vehicle_type"].to_numpy(),
        "city_type": c_te["city_type"].fillna("Missing").to_numpy(),
        "city_code": c_te["city_code"].to_numpy(),
        "is_peak_hour": d["X_te"]["is_peak_hour"].astype(int).to_numpy(),
        "distance_km": c_te["distance_km"].to_numpy(),
        "multiple_deliveries": c_te["multiple_deliveries"].to_numpy(),
        "coords_recovered": c_te["restaurant_coords_missing"].to_numpy(),
        "order_time_imputed": c_te["order_time_imputed"].to_numpy(),
    })
    preds["distance_bucket"] = distance_bucket(preds["distance_km"])
    preds["abs_error"] = (preds["actual"] - preds["predicted"]).abs()
    preds.to_csv(out / "test_predictions.csv", index=False)

    banner("Segment errors (v2 XGBoost)")
    for seg in ["traffic", "weather", "vehicle", "is_peak_hour", "distance_bucket",
                "city_type", "multiple_deliveries", "coords_recovered", "order_time_imputed"]:
        s = segment_errors(preds, seg)
        s.to_csv(out / f"error_by_{seg}.csv", index=False)
        print(f"\n-- {seg} --\n" + s.round(2).to_string(index=False))

    worst = preds.sort_values("abs_error", ascending=False).head(15)
    worst.to_csv(out / "worst_15.csv", index=False)

    # ---- save model ---------------------------------------------------------
    mdir = config.model_dir()
    joblib.dump({"pipeline": xgb, "prep_history": d["encoder"],
                 "numeric": num, "categorical": cat}, mdir / "phase0_xgb.joblib")
    save_json({"phase": 0, "split": "random 80/20 seed 42 (same as v1)",
               "metrics": comp.reset_index().to_dict(orient="records"),
               "audit": audit}, mdir / "phase0_metadata.json")
    print(f"\nSaved outputs to {out}")


if __name__ == "__main__":
    main()
