"""Prediction intervals: "22 min (18-27)".

Two methods, both calibrated on a held-out validation block:

1. Split conformal (absolute residual)
   q = the ceil((n+1)(1-alpha))/n quantile of |y - yhat| on validation.
   Interval = yhat +/- q. Guaranteed marginal coverage >= 1-alpha if the
   validation and test orders are exchangeable. Same width for every order.

2. Conformalized quantile regression (CQR, Romano et al. 2019)
   Fit XGBoost quantile models for alpha/2 and 1-alpha/2, then widen both
   ends by the conformal quantile of max(lo - y, y - hi) on validation.
   Width adapts: hard orders (jam, long distance) get wider ranges.
"""
from __future__ import annotations

import math

import numpy as np
from xgboost import XGBRegressor

from deliveriq.models.factory import make_preprocessor, xgb_params
from sklearn.pipeline import Pipeline


def conformal_quantile(scores, alpha: float) -> float:
    scores = np.sort(np.asarray(scores, float))
    n = len(scores)
    k = min(n, math.ceil((n + 1) * (1 - alpha)))
    return float(scores[k - 1])


class SplitConformal:
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
        self.q_ = None

    def calibrate(self, y_val, p_val):
        self.q_ = conformal_quantile(np.abs(np.asarray(y_val) - np.asarray(p_val)), self.alpha)
        return self

    def interval(self, p):
        p = np.asarray(p, float)
        return p - self.q_, p + self.q_


def quantile_model(numeric, categorical, q: float, **overrides) -> Pipeline:
    params = xgb_params(objective="reg:quantileerror", quantile_alpha=q, **overrides)
    return Pipeline([("pre", make_preprocessor(numeric, categorical)),
                     ("model", XGBRegressor(**params))])


class CQR:
    def __init__(self, numeric, categorical, alpha: float = 0.2, **xgb_overrides):
        self.alpha = alpha
        self.lo = quantile_model(numeric, categorical, alpha / 2, **xgb_overrides)
        self.hi = quantile_model(numeric, categorical, 1 - alpha / 2, **xgb_overrides)
        self.q_ = None

    def fit(self, X, y):
        self.lo.fit(X, y)
        self.hi.fit(X, y)
        return self

    def calibrate(self, X_val, y_val):
        lo, hi = self.lo.predict(X_val), self.hi.predict(X_val)
        y_val = np.asarray(y_val, float)
        self.q_ = conformal_quantile(np.maximum(lo - y_val, y_val - hi), self.alpha)
        return self

    def interval(self, X):
        lo, hi = self.lo.predict(X), self.hi.predict(X)
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
        return lo - self.q_, hi + self.q_


def coverage_report(y, lo, hi) -> dict:
    y = np.asarray(y, float)
    inside = (y >= lo) & (y <= hi)
    return {"coverage_pct": float(inside.mean() * 100),
            "mean_width_min": float(np.mean(hi - lo)),
            "median_width_min": float(np.median(hi - lo)),
            "above_upper_pct": float((y > hi).mean() * 100),
            "below_lower_pct": float((y < lo).mean() * 100)}
