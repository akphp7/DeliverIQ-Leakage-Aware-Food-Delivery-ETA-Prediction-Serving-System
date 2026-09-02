"""Phase 6 - benchmark against published work on the same dataset.

Reference: A. Garg, M. Ayaan, S. Parekh, V. Udandarao,
"Food Delivery Time Prediction in Indian Cities Using Machine Learning Models",
arXiv:2503.15177 (2025). Best reported model: LightGBM, MSE 20.59, R2 0.76,
after dropping rows with missing values (41,368 rows) and a random hold-out split.

Run:  python -m pipelines.p6_paper_benchmark

This script re-creates that protocol (drop incomplete rows, random 80/20 split)
and scores DeliverIQ's features and models under it, plus the same models on
all rows. LightGBM is optional: `pip install lightgbm` to include it.

Caveats (also in docs/BENCHMARK.md): the paper's split seed and exact cleaning
are not published, and DeliverIQ's XGBoost settings were tuned on this dataset,
so this is a like-for-like protocol comparison, not an identical-split one.

Outputs: outputs/v2/phase6/paper_benchmark.csv
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from deliveriq import config
from deliveriq.data.cleaning import clean_orders, load_raw
from deliveriq.evaluation.metrics import regression_metrics
from deliveriq.features.build import FINAL_DROP, base_features, feature_columns
from deliveriq.models.factory import make_model, make_preprocessor
from pipelines.common import banner

PAPER = {"protocol": "published (arXiv:2503.15177)", "Model": "LightGBM (paper)",
         "rows": 41368, "MSE": 20.59, "R2": 0.76}


def lightgbm_pipeline(num, cat):
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        return None
    return Pipeline([("pre", make_preprocessor(num, cat)),
                     ("model", LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=63,
                                             random_state=config.RANDOM_STATE, verbose=-1))])


def main():
    out = config.phase_dir(6)
    raw = load_raw()
    complete = raw.dropna().index                     # paper: drop rows with any missing value
    clean, _ = clean_orders(raw)
    X = base_features(clean)
    y = clean[config.TARGET].to_numpy(float)
    num, cat = feature_columns(FINAL_DROP)
    xgb_params = joblib.load(config.MODEL_DIR / "final_bundle.joblib")["xgb_params"]

    rows = [PAPER]
    for protocol, idx in [("paper-style: complete rows, random 80/20", complete),
                          ("all rows, random 80/20", X.index)]:
        tr, te = train_test_split(idx, test_size=0.2, random_state=config.RANDOM_STATE)
        banner(f"{protocol}: train {len(tr):,} / test {len(te):,}")
        models = {
            "Linear Regression": make_model("Linear Regression", num, cat),
            "Random Forest": make_model("Random Forest", num, cat),
            "XGBoost (DeliverIQ tuned)": make_model("XGBoost", num, cat, **xgb_params),
            "LightGBM (DeliverIQ features)": lightgbm_pipeline(num, cat),
        }
        for name, m in models.items():
            if m is None:
                print(f"{name}: skipped (pip install lightgbm to include)")
                continue
            m.fit(X.loc[tr], y[tr])
            r = regression_metrics(y[te], m.predict(X.loc[te]))
            rows.append({"protocol": protocol, "Model": name, "rows": len(idx),
                         "MSE": r["RMSE"] ** 2, "RMSE": r["RMSE"], "MAE": r["MAE"], "R2": r["R2"]})
            print(f"{name:32s} MSE {r['RMSE'] ** 2:6.2f}  RMSE {r['RMSE']:.2f}  "
                  f"MAE {r['MAE']:.2f}  R2 {r['R2']:.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(out / "paper_benchmark.csv", index=False)
    print(f"\nPaper best: MSE {PAPER['MSE']}, R2 {PAPER['R2']}")
    print(f"Saved {out / 'paper_benchmark.csv'}")


if __name__ == "__main__":
    main()
