"""Order-time feature construction for DeliverIQ v2.

Two kinds of features:
  1. Stateless: computed row by row from the cleaned order (`base_features`).
  2. Learned from training data: restaurant history (`add_history`).
Keeping them separate makes it impossible to fit (2) on test rows by accident.

Changes vs v1 (see journal, Phase 0):
  * hand-written weather factor and distance x weather removed: the priors
    (Fog slowest, Sunny = Clear) contradicted the data (Sunny is fastest,
    Cloudy = Fog slowest), and one-hot weather already carries the signal;
  * traffic is an ordinal level 0-3 (Low..Jam) plus one-hot, NaN kept;
  * hour also encoded cyclically (sin/cos) so 23:00 and 00:00 are neighbours;
  * city_code (22 cities) added; missing-value flags added;
  * rating x vehicle-condition interaction dropped (no clear rationale).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from deliveriq import config
from deliveriq.features.prep_history import RestaurantPrepHistory

NUMERIC = [
    "rider_age", "rider_rating", "vehicle_condition", "multiple_deliveries",
    "distance_km", "order_hour", "hour_sin", "hour_cos", "day_of_week",
    "is_weekend", "is_peak_hour", "traffic_level", "distance_x_traffic",
    "traffic_missing", "weather_missing", "rider_profile_invalid",
    "order_time_imputed", "restaurant_coords_missing",
    "hist_prep_mean", "hist_restaurant_orders",
]
CATEGORICAL = [
    "weather", "traffic", "order_type", "vehicle_type", "festival",
    "city_type", "city_code", "time_of_day",
]
HISTORY = ["hist_prep_mean", "hist_restaurant_orders"]

# Phase 1 decision: the restaurant prep-history feature is dropped from the
# FINAL model. In this dataset pickup delay is random (5/10/15 min, no link to
# delivery time), and removing the feature lowered XGBoost CV MAE in all 5
# folds (3.199 -> 3.109). The leak-free encoder is kept for the methodology.
FINAL_DROP = list(HISTORY)


def _time_of_day(hour: float) -> str:
    if pd.isna(hour):
        return "Unknown"
    h = int(hour)
    if 5 <= h <= 11:
        return "Morning"
    if 12 <= h <= 16:
        return "Afternoon"
    if 17 <= h <= 21:
        return "Evening"
    return "Night"


def base_features(clean: pd.DataFrame) -> pd.DataFrame:
    """Stateless order-time features. Does not include history columns."""
    X = pd.DataFrame(index=clean.index)
    for c in ["rider_age", "rider_rating", "vehicle_condition", "multiple_deliveries",
              "distance_km", "traffic_missing", "weather_missing",
              "rider_profile_invalid", "order_time_imputed", "restaurant_coords_missing"]:
        X[c] = clean[c].astype(float)

    hour = np.floor(clean["order_minute_of_day"] / 60)
    X["order_hour"] = hour
    angle = 2 * np.pi * clean["order_minute_of_day"] / 1440
    X["hour_sin"] = np.sin(angle)
    X["hour_cos"] = np.cos(angle)
    X["day_of_week"] = clean["order_date"].dt.dayofweek.astype(float)
    X["is_weekend"] = (X["day_of_week"] >= 5).astype(float)
    X["is_peak_hour"] = hour.isin(config.PEAK_HOURS).astype(float)

    X["traffic_level"] = clean["traffic"].map(config.TRAFFIC_LEVEL).astype(float)
    X["distance_x_traffic"] = X["distance_km"] * X["traffic_level"]

    for c in ["weather", "traffic", "order_type", "vehicle_type", "festival",
              "city_type", "city_code"]:
        X[c] = clean[c].astype("object").where(clean[c].notna(), "Missing")
    X["time_of_day"] = hour.map(_time_of_day)
    return X


def add_history(X_train: pd.DataFrame, clean_train: pd.DataFrame,
                others: list[tuple[pd.DataFrame, pd.DataFrame]] | None = None,
                mode: str = "oof") -> tuple[RestaurantPrepHistory, pd.DataFrame, list[pd.DataFrame]]:
    """Attach restaurant-history columns.

    mode="oof"   -> leak-free (v2 default)
    mode="leaky" -> v1 behaviour, only used to demonstrate the leakage effect
    `others` is a list of (X, clean) pairs (test / inference) that receive the
    full-train mapping.
    """
    enc = RestaurantPrepHistory()
    X_train = X_train.copy()
    if mode == "oof":
        h = enc.fit_transform_oof(clean_train)
    elif mode == "leaky":
        enc.fit(clean_train)
        h = enc.in_sample_leaky(clean_train)
    else:
        raise ValueError(mode)
    X_train[HISTORY] = h[HISTORY].to_numpy()

    out = []
    for X_o, clean_o in (others or []):
        X_o = X_o.copy()
        X_o[HISTORY] = enc.transform(clean_o)[HISTORY].to_numpy()
        out.append(X_o)
    return enc, X_train, out


def feature_columns(drop: list[str] | None = None) -> tuple[list[str], list[str]]:
    drop = set(drop or [])
    return ([c for c in NUMERIC if c not in drop],
            [c for c in CATEGORICAL if c not in drop])
