"""SHAP explanations for the XGBoost pipeline, grouped back to raw features.

One-hot encoding splits "traffic" into traffic_Low, traffic_Jam, ... For a
human-readable reason we add those SHAP values back into one "traffic"
contribution (SHAP values are additive, so summing within a group is valid).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

READABLE = {
    "rider_age": "Rider age", "rider_rating": "Rider rating",
    "vehicle_condition": "Vehicle condition", "multiple_deliveries": "Multiple deliveries",
    "distance_km": "Distance", "distance_x_traffic": "Distance x traffic",
    "traffic_level": "Traffic", "traffic": "Traffic", "weather": "Weather",
    "order_hour": "Order time", "hour_sin": "Order time", "hour_cos": "Order time",
    "time_of_day": "Order time", "is_peak_hour": "Peak hour",
    "day_of_week": "Day of week", "is_weekend": "Day of week",
    "city_code": "City", "city_type": "City type", "festival": "Festival",
    "order_type": "Order type", "vehicle_type": "Vehicle type",
    "traffic_missing": "Missing data flags", "weather_missing": "Missing data flags",
    "rider_profile_invalid": "Missing data flags", "order_time_imputed": "Missing data flags",
    "restaurant_coords_missing": "Missing data flags",
}


def _raw_name(transformed: str, categorical: list[str]) -> str:
    name = transformed.split("__", 1)[1]
    for c in sorted(categorical, key=len, reverse=True):
        if name.startswith(c + "_"):
            return c
    return name


class GroupedShap:
    def __init__(self, pipeline, categorical: list[str]):
        self.pre = pipeline.named_steps["pre"]
        self.model = pipeline.named_steps["model"]
        self.explainer = shap.TreeExplainer(self.model)
        names = self.pre.get_feature_names_out()
        self.groups = pd.Series([READABLE.get(_raw_name(n, categorical), _raw_name(n, categorical))
                                 for n in names])
        self.base_value = float(np.ravel(self.explainer.expected_value)[0])

    def raw_values(self, X: pd.DataFrame) -> np.ndarray:
        return self.explainer.shap_values(self.pre.transform(X))

    def grouped(self, X: pd.DataFrame) -> pd.DataFrame:
        sv = self.raw_values(X)
        df = pd.DataFrame(sv, columns=self.groups.to_numpy(), index=X.index)
        return df.T.groupby(level=0).sum().T

    def reasons(self, X: pd.DataFrame, top: int = 3) -> list[list[dict]]:
        g = self.grouped(X)
        out = []
        for _, row in g.iterrows():
            r = row.reindex(row.abs().sort_values(ascending=False).index).head(top)
            out.append([{"factor": k, "minutes": round(float(v), 1)} for k, v in r.items()])
        return out
