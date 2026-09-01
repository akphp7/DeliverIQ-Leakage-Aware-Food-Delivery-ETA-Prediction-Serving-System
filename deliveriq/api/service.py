"""ETA prediction service used by the v2 API (and by the tests).

One order in -> ETA, 80% range, top reasons, warnings.
The request goes through exactly the same cleaning and feature code as
training (no train/serve skew), using the city centroids saved in the bundle.
If the model fails, a rule-based fallback answers instead of an error.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd

from deliveriq import config
from deliveriq.data.cleaning import clean_orders
from deliveriq.features.build import base_features
from deliveriq.models.explain import GroupedShap

BUNDLE_PATH = config.MODEL_DIR / "final_bundle.joblib"
FALLBACK_PATH = config.MODEL_DIR / "fallback.json"
MODEL_VERSION = "deliveriq-v2-xgb-cqr"


@dataclass
class OrderInput:
    restaurant_latitude: float
    restaurant_longitude: float
    delivery_latitude: float
    delivery_longitude: float
    order_date: str                       # DD-MM-YYYY
    time_ordered: str                     # HH:MM
    city_code: str | None = None          # e.g. "BANG"; needed only if coordinates are missing
    delivery_person_age: float | None = None
    delivery_person_rating: float | None = None
    vehicle_condition: float | None = None
    multiple_deliveries: float | None = None
    weather_conditions: str | None = None
    road_traffic_density: str | None = None
    type_of_order: str = "Meal"
    type_of_vehicle: str = "motorcycle"
    festival: str = "No"
    city: str | None = "Metropolitian"
    extra: dict = field(default_factory=dict)

    def to_raw(self, i: int = 0) -> dict:
        city = (self.city_code or "UNK").upper()
        return {
            "ID": f"req{i}", "Delivery_person_ID": f"{city}RES00DEL00",
            "Delivery_person_Age": self.delivery_person_age,
            "Delivery_person_Ratings": self.delivery_person_rating,
            "Restaurant_latitude": self.restaurant_latitude,
            "Restaurant_longitude": self.restaurant_longitude,
            "Delivery_location_latitude": self.delivery_latitude,
            "Delivery_location_longitude": self.delivery_longitude,
            "Order_Date": self.order_date, "Time_Orderd": self.time_ordered,
            "Time_Order_picked": None,            # unknown at order time, never used
            "Weather_conditions": self.weather_conditions,
            "Road_traffic_density": self.road_traffic_density,
            "Vehicle_condition": self.vehicle_condition,
            "Type_of_order": self.type_of_order, "Type_of_vehicle": self.type_of_vehicle,
            "multiple_deliveries": self.multiple_deliveries,
            "Festival": self.festival, "City": self.city,
            "Time_taken (min)": None,
        }


class ETAService:
    def __init__(self, bundle_path=BUNDLE_PATH, fallback_path=FALLBACK_PATH):
        self.bundle = joblib.load(bundle_path)
        self.model = self.bundle["point_model"]
        self.cqr = self.bundle["cqr80"]
        self.reference = self.bundle["city_reference"]
        self.explainer = GroupedShap(self.model, self.bundle["categorical"])
        self.fallback = json.loads(fallback_path.read_text()) if fallback_path.exists() else None

    # ------------------------------------------------------------------
    def _features(self, orders: list[OrderInput]) -> tuple[pd.DataFrame, pd.DataFrame]:
        raw = pd.DataFrame([o.to_raw(i) for i, o in enumerate(orders)])
        clean, _ = clean_orders(raw, reference=self.reference)
        return clean, base_features(clean)

    def _warnings(self, c: pd.Series, width: float) -> list[str]:
        w = []
        if c["traffic_missing"]:
            w.append("traffic unknown: range widened")
        if c["weather_missing"]:
            w.append("weather unknown")
        if pd.isna(c["rider_age"]) or pd.isna(c["rider_rating"]):
            w.append("rider profile missing or invalid")
        if c["restaurant_coords_missing"]:
            w.append("restaurant location invalid: city centre used")
        if pd.isna(c["distance_km"]):
            w.append("distance unknown")
        if width > 15:
            w.append("low confidence: wide range")
        return w

    def _fallback(self, c: pd.Series) -> float:
        fb = self.fallback or {"global_median": 26.0}
        by_t = fb.get("median_by_traffic", {})
        return float(by_t.get(str(c.get("traffic")), fb["global_median"]))

    def predict(self, orders: list[OrderInput], explain: bool = True) -> list[dict]:
        t0 = time.perf_counter()
        clean, X = self._features(orders)
        try:
            eta = self.model.predict(X)
            lo, hi = self.cqr.interval(X)
            reasons = self.explainer.reasons(X, top=3) if explain else [[] for _ in orders]
            source = "model"
        except Exception as exc:                      # graceful degradation
            eta = np.array([self._fallback(clean.iloc[i]) for i in range(len(orders))])
            lo, hi = eta - 6.0, eta + 6.0
            reasons = [[{"factor": "fallback", "minutes": 0.0, "error": str(exc)[:80]}]
                       for _ in orders]
            source = "fallback"
        ms = (time.perf_counter() - t0) * 1000
        out = []
        for i in range(len(orders)):
            e = float(max(eta[i], 1.0))
            low = float(max(1.0, min(lo[i], e)))
            high = float(max(hi[i], e))
            out.append({
                "eta_minutes": round(e, 1),
                "eta_range_80": [round(low, 1), round(high, 1)],
                "top_reasons": reasons[i],
                "baseline_minutes": round(self.explainer.base_value, 1),
                "warnings": self._warnings(clean.iloc[i], high - low),
                "source": source,
                "model_version": MODEL_VERSION,
                "latency_ms": round(ms / len(orders), 2),
            })
        return out
