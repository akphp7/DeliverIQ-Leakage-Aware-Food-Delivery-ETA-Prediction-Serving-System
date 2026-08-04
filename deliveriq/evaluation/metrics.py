"""Accuracy and business metrics for ETA predictions.

Business framing: a customer is hurt more by an order arriving LATER than
promised than by one arriving early. So besides MAE/RMSE/R2 we report how
often the order is late or early by more than 5 minutes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred            # > 0  -> order arrived later than predicted
    abs_err = np.abs(err)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "MedianAE": float(np.median(abs_err)),
        "P90_AE": float(np.percentile(abs_err, 90)),
        "within_5min_pct": float(np.mean(abs_err <= 5) * 100),
        "within_10min_pct": float(np.mean(abs_err <= 10) * 100),
        "late_gt5_pct": float(np.mean(err > 5) * 100),
        "early_gt5_pct": float(np.mean(err < -5) * 100),
        "mean_bias": float(np.mean(err)),   # + = model under-predicts on average
    }


CORE = ["MAE", "RMSE", "R2"]
BUSINESS = ["P90_AE", "within_5min_pct", "late_gt5_pct", "early_gt5_pct", "mean_bias"]


def segment_errors(frame: pd.DataFrame, by: str, actual="actual", pred="predicted") -> pd.DataFrame:
    d = frame.assign(abs_error=(frame[actual] - frame[pred]).abs(),
                     signed_error=frame[actual] - frame[pred])
    return (d.groupby(by, observed=False, dropna=False)
              .agg(MAE=("abs_error", "mean"), median_AE=("abs_error", "median"),
                   bias=("signed_error", "mean"), n=("abs_error", "size"))
              .reset_index())
