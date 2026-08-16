"""Combining Random Forest and XGBoost.

Blending protocol (no test data is touched while choosing weights):
  1. fit both models on train_core
  2. predict the validation block (the last days before the test period)
  3. choose the blend on those validation predictions:
       * weighted average: w * RF + (1 - w) * XGB, w on a 0.05 grid, min MAE
       * stacking: a linear model (non-negative weights + intercept) fitted on
         the two validation predictions
  4. refit both models on the full training period and apply the chosen blend
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression


def best_weight(y_val, p_rf, p_xgb, grid=np.round(np.arange(0, 1.0001, 0.05), 2)):
    maes = [np.mean(np.abs(y_val - (w * p_rf + (1 - w) * p_xgb))) for w in grid]
    i = int(np.argmin(maes))
    return float(grid[i]), [(float(w), float(m)) for w, m in zip(grid, maes)]


class WeightedBlend:
    def __init__(self, w_rf: float):
        self.w_rf = w_rf

    def predict(self, p_rf, p_xgb):
        return self.w_rf * np.asarray(p_rf) + (1 - self.w_rf) * np.asarray(p_xgb)


class StackedBlend:
    def __init__(self):
        self.lr = LinearRegression(positive=True)

    def fit(self, y_val, p_rf, p_xgb):
        self.lr.fit(np.column_stack([p_rf, p_xgb]), y_val)
        return self

    def predict(self, p_rf, p_xgb):
        return self.lr.predict(np.column_stack([p_rf, p_xgb]))

    @property
    def coefs(self):
        return {"rf": float(self.lr.coef_[0]), "xgb": float(self.lr.coef_[1]),
                "intercept": float(self.lr.intercept_)}
