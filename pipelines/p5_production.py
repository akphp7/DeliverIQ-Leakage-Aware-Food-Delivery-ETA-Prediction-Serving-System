"""Phase 5 - production readiness checks.

Run:  python -m pipelines.p5_production

1. Builds the rule-based fallback table (used if the model fails).
2. Smoke-tests the v2 service on real test orders (range contains ETA, coverage).
3. Measures latency (single order with/without SHAP, batch of 100).
4. Computes A/B-test sample sizes for launching a new ETA model.
5. Computes a drift baseline (PSI) between the training and test periods.

Outputs: outputs/v2/phase5/*.csv, models/v2/fallback.json
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from statsmodels.stats.power import NormalIndPower, TTestIndPower
from statsmodels.stats.proportion import proportion_effectsize

from deliveriq import config
from deliveriq.api.service import ETAService, OrderInput
from deliveriq.data.cleaning import load_clean, load_raw
from deliveriq.evaluation.splits import temporal_split
from deliveriq.features.build import base_features
from pipelines.common import banner, save_json


def raw_to_input(r: pd.Series) -> OrderInput:
    def num(v):
        v = pd.to_numeric(v, errors="coerce")
        return None if pd.isna(v) else float(v)

    def txt(v):
        v = None if pd.isna(v) else str(v).strip()
        return None if v in (None, "NaN", "") else v
    return OrderInput(
        restaurant_latitude=float(r.Restaurant_latitude), restaurant_longitude=float(r.Restaurant_longitude),
        delivery_latitude=float(r.Delivery_location_latitude), delivery_longitude=float(r.Delivery_location_longitude),
        order_date=r.Order_Date, time_ordered=str(r.Time_Orderd),
        city_code=str(r.Delivery_person_ID).split("RES")[0],
        delivery_person_age=num(r.Delivery_person_Age), delivery_person_rating=num(r.Delivery_person_Ratings),
        vehicle_condition=num(r.Vehicle_condition), multiple_deliveries=num(r.multiple_deliveries),
        weather_conditions=txt(r.Weather_conditions), road_traffic_density=txt(r.Road_traffic_density),
        type_of_order=txt(r.Type_of_order) or "Meal", type_of_vehicle=txt(r.Type_of_vehicle) or "motorcycle",
        festival=txt(r.Festival) or "No", city=txt(r.City))


def psi(expected, actual, bins=10) -> float:
    """Population Stability Index; > 0.2 is usually treated as significant drift."""
    expected, actual = pd.Series(expected).dropna(), pd.Series(actual).dropna()
    if expected.dtype == object or expected.nunique() <= 12:
        e = expected.value_counts(normalize=True)
        a = actual.value_counts(normalize=True).reindex(e.index).fillna(0)
    else:
        edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
        e = pd.cut(expected, edges, include_lowest=True).value_counts(normalize=True, sort=False)
        a = pd.cut(actual.clip(edges[0], edges[-1]), edges, include_lowest=True).value_counts(normalize=True, sort=False)
    e, a = e.clip(lower=1e-4), a.clip(lower=1e-4)
    return float(((a - e) * np.log(a / e)).sum())


def main():
    out = config.phase_dir(5)
    clean, _ = load_clean()
    raw = load_raw()
    tr, te, cutoff, _ = temporal_split(clean)

    banner("1. Fallback table (training period only)")
    trc = clean.iloc[tr]
    fb = {"global_median": float(trc[config.TARGET].median()),
          "median_by_traffic": trc.groupby("traffic")[config.TARGET].median().astype(float).to_dict()}
    save_json(fb, config.model_dir() / "fallback.json")
    print(fb)

    banner("2. Service smoke test on 1,000 test orders (sent as raw requests)")
    svc = ETAService()
    rng = np.random.default_rng(config.RANDOM_STATE)
    idx = rng.choice(te, 1000, replace=False)
    # only rows whose clock is HH:MM, as the API requires
    idx = [i for i in idx if ":" in str(raw.iloc[i].Time_Orderd)]
    inputs = [raw_to_input(raw.iloc[i]) for i in idx]
    res = svc.predict(inputs, explain=True)
    y = clean.iloc[idx][config.TARGET].to_numpy(float)
    eta = np.array([r["eta_minutes"] for r in res])
    lo = np.array([r["eta_range_80"][0] for r in res])
    hi = np.array([r["eta_range_80"][1] for r in res])
    smoke = {"orders": len(idx), "MAE": float(np.mean(np.abs(y - eta))),
             "coverage_80_pct": float(np.mean((y >= lo) & (y <= hi)) * 100),
             "mean_width": float(np.mean(hi - lo)),
             "range_contains_eta_pct": float(np.mean((lo <= eta) & (eta <= hi)) * 100),
             "orders_with_warnings_pct": float(np.mean([bool(r["warnings"]) for r in res]) * 100),
             "source_model_pct": float(np.mean([r["source"] == "model" for r in res]) * 100)}
    pd.Series(smoke).to_csv(out / "service_smoke_test.csv")
    print(pd.Series(smoke).round(3).to_string())
    print("\nexample response:", res[0])

    banner("3. Latency (this machine, single process)")
    one = inputs[:1]
    for _ in range(5):
        svc.predict(one)
    lat = {}
    for label, fn in [("single_with_shap", lambda: svc.predict(one, explain=True)),
                      ("single_no_shap", lambda: svc.predict(one, explain=False)),
                      ("batch100_with_shap", lambda: svc.predict(inputs[:100], explain=True))]:
        ts = []
        for _ in range(50 if "single" in label else 10):
            t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1000)
        lat[label] = {"p50_ms": float(np.percentile(ts, 50)), "p95_ms": float(np.percentile(ts, 95))}
    lat = pd.DataFrame(lat).T
    lat["per_order_p50_ms"] = lat["p50_ms"] / np.where(lat.index.str.startswith("batch"), 100, 1)
    lat.to_csv(out / "latency.csv")
    print(lat.round(2).to_string())

    banner("4. A/B test sizing for a new ETA model")
    ab = []
    Xte = base_features(clean.iloc[te])
    err = clean.iloc[te][config.TARGET].to_numpy(float) - svc.model.predict(Xte)
    base_late = float(np.mean(err > 5))
    sd_abs = float(np.std(np.abs(err), ddof=1))
    base_mae = float(np.mean(np.abs(err)))
    print(f"test baseline: late>5 = {base_late:.3f}, MAE = {base_mae:.3f}, sd(|error|) = {sd_abs:.3f}")
    # (a) primary guardrail: share of orders late by > 5 min
    for rel in [0.05, 0.10, 0.20]:
        new = base_late * (1 - rel)
        es = proportion_effectsize(base_late, new)
        n = NormalIndPower().solve_power(es, alpha=0.05, power=0.8, ratio=1)
        ab.append({"metric": "late > 5 min rate", "baseline": round(base_late, 4), "target": round(new, 4),
                   "relative_change": rel, "orders_per_arm": int(np.ceil(n))})
    # (b) absolute ETA error per order (sd of |error| on the test set)
    for delta in [0.05, 0.10, 0.20]:
        n = TTestIndPower().solve_power(delta / sd_abs, alpha=0.05, power=0.8, ratio=1)
        ab.append({"metric": "absolute ETA error (min)", "baseline": round(base_mae, 3), "target": round(base_mae - delta, 2),
                   "relative_change": round(delta / base_mae, 3), "orders_per_arm": int(np.ceil(n))})
    ab = pd.DataFrame(ab)
    ab.to_csv(out / "ab_test_sample_size.csv", index=False)
    print(ab.to_string(index=False))

    banner("5. Drift baseline: PSI train period vs test period")
    X = base_features(clean)
    rows = []
    for c in ["distance_km", "rider_age", "rider_rating", "order_hour", "multiple_deliveries",
              "traffic", "weather", "city_code", "festival", "vehicle_type"]:
        rows.append({"feature": c, "psi": psi(X.iloc[tr][c], X.iloc[te][c])})
    rows.append({"feature": "target: time_taken_min",
                 "psi": psi(clean.iloc[tr][config.TARGET], clean.iloc[te][config.TARGET])})
    drift = pd.DataFrame(rows).sort_values("psi", ascending=False)
    drift["status"] = pd.cut(drift.psi, [-1, 0.1, 0.2, 99], labels=["stable", "watch", "drift"])
    drift.to_csv(out / "drift_psi.csv", index=False)
    print(drift.round(4).to_string(index=False))
    print(f"\nSaved outputs to {out}")


if __name__ == "__main__":
    main()
