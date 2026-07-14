"""Load and clean the raw orders into one canonical, snake_case table.

Design rule: cleaning never uses the target, and never drops an order.
Bad values become NaN plus an explicit flag, so the model (and a
reviewer) can see what was repaired.

Every repair is counted in the audit dict returned by `clean_orders`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from deliveriq import config
from deliveriq.data.parsing import parse_clock_minutes, minutes_between

RAW_TO_CANON = {
    "ID": "order_id",
    "Delivery_person_ID": "rider_id",
    "Delivery_person_Age": "rider_age",
    "Delivery_person_Ratings": "rider_rating",
    "Weather_conditions": "weather",
    "Road_traffic_density": "traffic",
    "Vehicle_condition": "vehicle_condition",
    "Type_of_order": "order_type",
    "Type_of_vehicle": "vehicle_type",
    "multiple_deliveries": "multiple_deliveries",
    "Festival": "festival",
    "City": "city_type",
    "Time_taken (min)": config.TARGET,
}

CATEGORICAL_TEXT = ["weather", "traffic", "order_type", "vehicle_type", "festival", "city_type"]


def load_raw(path=config.RAW_DATA) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def city_reference(clean: pd.DataFrame) -> dict:
    """City centroid table, saved with the model so inference can reuse it."""
    ok = clean["restaurant_coords_missing"] == 0
    g = clean.loc[ok].groupby("city_code")[["restaurant_lat", "restaurant_lon"]].median()
    return {"city_lat": g["restaurant_lat"].to_dict(), "city_lon": g["restaurant_lon"].to_dict()}


def clean_orders(raw: pd.DataFrame, reference: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Return (clean dataframe, audit counts). Row order and count are preserved.

    `reference` (from `city_reference`) supplies city centroids at inference
    time, when the incoming batch is a single order.
    """
    audit: dict[str, int] = {"rows": len(raw)}
    df = raw.rename(columns=RAW_TO_CANON).copy()

    # ---- text categoricals: strip spaces, keep NaN as NaN -----------------
    for c in CATEGORICAL_TEXT:
        df[c] = df[c].astype("string").str.strip()
        df.loc[df[c].isin(["NaN", "nan", ""]), c] = pd.NA
        audit[f"missing_{c}"] = int(df[c].isna().sum())
    df["traffic_missing"] = df["traffic"].isna().astype(int)
    df["weather_missing"] = df["weather"].isna().astype(int)

    # ---- IDs: rider_id encodes city + restaurant, e.g. DEHRES17DEL01 -------
    parts = df["rider_id"].astype("string").str.extract(r"^([A-Z]+)RES(\d+)DEL(\d+)$")
    df["city_code"] = parts[0]
    df["restaurant_id"] = parts[0] + "RES" + parts[1]
    audit["unparsed_rider_id"] = int(parts[0].isna().sum())

    # ---- rider profile: sentinel rows and impossible ratings ---------------
    df["rider_age"] = pd.to_numeric(df["rider_age"], errors="coerce")
    df["rider_rating"] = pd.to_numeric(df["rider_rating"], errors="coerce")
    sentinel = pd.Series(False, index=df.index)
    for age, rating in config.SENTINEL_RIDER_PROFILES:
        sentinel |= (df["rider_age"] == age) & (df["rider_rating"] == rating)
    df["rider_profile_invalid"] = sentinel.astype(int)
    df.loc[sentinel, ["rider_age", "rider_rating"]] = np.nan
    bad_rating = df["rider_rating"].notna() & ~df["rider_rating"].between(1, 5)
    df.loc[bad_rating, "rider_rating"] = np.nan
    audit["sentinel_rider_rows"] = int(sentinel.sum())
    audit["rating_out_of_range_after_sentinel"] = int(bad_rating.sum())

    for c in ["vehicle_condition", "multiple_deliveries"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    audit["missing_multiple_deliveries"] = int(df["multiple_deliveries"].isna().sum())

    # ---- time: order date, order clock, pickup clock -----------------------
    df["order_date"] = pd.to_datetime(df["Order_Date"], format="%d-%m-%Y", errors="coerce")
    audit["bad_order_date"] = int(df["order_date"].isna().sum())

    order_min = parse_clock_minutes(df["Time_Orderd"])
    pickup_min = parse_clock_minutes(df["Time_Order_picked"])
    v1_style = pd.to_datetime(df["Time_Orderd"], format="%H:%M", errors="coerce")
    audit["order_time_unparsed_by_v1"] = int(v1_style.isna().sum())
    audit["order_time_recovered_from_fraction"] = int((v1_style.isna() & order_min.notna()).sum())

    # Pickup-to-order gap is used ONLY for history features and for repairing
    # missing order clocks in the training data; never as a same-order feature.
    df["pickup_delay_min"] = minutes_between(order_min, pickup_min)
    typical_gap = float(df["pickup_delay_min"].median())
    if not np.isfinite(typical_gap):      # e.g. inference rows without pickup
        typical_gap = 10.0
    missing_order = order_min.isna() & pickup_min.notna()
    df["order_time_imputed"] = missing_order.astype(int)
    order_min = order_min.where(~missing_order, (pickup_min - typical_gap) % 1440)
    df["order_minute_of_day"] = order_min
    audit["order_time_imputed_from_pickup"] = int(missing_order.sum())
    audit["order_time_still_missing"] = int(order_min.isna().sum())

    # ---- coordinates -------------------------------------------------------
    for c in ["Restaurant_latitude", "Restaurant_longitude",
              "Delivery_location_latitude", "Delivery_location_longitude"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").abs()
    rlat, rlon = df["Restaurant_latitude"], df["Restaurant_longitude"]
    dlat, dlon = df["Delivery_location_latitude"], df["Delivery_location_longitude"]

    in_india = (rlat.between(*config.INDIA_LAT) & rlon.between(*config.INDIA_LON))
    df["restaurant_coords_missing"] = (~in_india).astype(int)
    audit["restaurant_coords_invalid"] = int((~in_india).sum())

    # In this data the drop point is always restaurant + (d, d) with d in a
    # small fixed set. For rows whose restaurant was logged as (0, 0) the
    # delivery columns therefore still hold the offset d itself.
    offset_lat = np.where(in_india, dlat - rlat, dlat)
    offset_lon = np.where(in_india, dlon - rlon, dlon)

    if reference is None:
        city_lat = df.loc[in_india].groupby("city_code")["Restaurant_latitude"].median()
        city_lon = df.loc[in_india].groupby("city_code")["Restaurant_longitude"].median()
    else:
        city_lat = pd.Series(reference["city_lat"])
        city_lon = pd.Series(reference["city_lon"])
    base_lat = rlat.where(in_india, df["city_code"].map(city_lat))
    base_lon = rlon.where(in_india, df["city_code"].map(city_lon))
    df["restaurant_lat"] = base_lat
    df["restaurant_lon"] = base_lon
    df["delivery_lat"] = base_lat + offset_lat
    df["delivery_lon"] = base_lon + offset_lon
    df["distance_km"] = _haversine_km(base_lat, base_lon, df["delivery_lat"], df["delivery_lon"])
    audit["distance_recovered_with_city_centroid"] = int((~in_india & df["distance_km"].notna()).sum())
    audit["distance_missing"] = int(df["distance_km"].isna().sum())

    # ---- target ------------------------------------------------------------
    df[config.TARGET] = pd.to_numeric(df[config.TARGET], errors="coerce")
    audit["missing_target"] = int(df[config.TARGET].isna().sum())

    keep = [
        "order_id", "rider_id", "restaurant_id", "city_code", "order_date",
        "order_minute_of_day", "pickup_delay_min",
        "rider_age", "rider_rating", "vehicle_condition", "vehicle_type",
        "multiple_deliveries", "weather", "traffic", "order_type", "festival",
        "city_type", "restaurant_lat", "restaurant_lon", "delivery_lat",
        "delivery_lon", "distance_km",
        "traffic_missing", "weather_missing", "rider_profile_invalid",
        "order_time_imputed", "restaurant_coords_missing", config.TARGET,
    ]
    return df[keep].reset_index(drop=True), audit


def load_clean(path=config.RAW_DATA) -> tuple[pd.DataFrame, dict]:
    return clean_orders(load_raw(path))
