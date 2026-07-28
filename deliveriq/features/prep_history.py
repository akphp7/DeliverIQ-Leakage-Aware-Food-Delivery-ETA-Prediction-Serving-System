"""Leak-free historical restaurant preparation-time feature.

Why this exists: the current order's pickup time is unknown when the ETA
is shown, so v1 replaced it with "average prep time of this restaurant in
the training data". But v1 computed that average over ALL training rows,
so each training row's feature contained its own prep time. That is
in-sample leakage: the model sees a cleaner signal in training than it
will ever see in production.

v2 fix:
  * training rows get OUT-OF-FOLD values (K folds: a row's value is
    computed only from the other folds);
  * test / inference rows get the value computed on the full training set;
  * averages are smoothed toward the global mean (m-estimate), so a
    restaurant with 2 orders does not get an extreme value;
  * restaurants never seen in training get the global mean (cold start).
The key is `restaurant_id` (parsed from rider_id), not rounded
coordinates, so the 3,640 rows logged at (0, 0) are no longer merged into
one fake restaurant.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from deliveriq import config


class RestaurantPrepHistory:
    def __init__(self, k: float = config.PREP_SMOOTHING_K,
                 key: str = "restaurant_id", value: str = "pickup_delay_min"):
        self.k = k
        self.key = key
        self.value = value
        self.global_mean_: float | None = None
        self.table_: pd.DataFrame | None = None

    def _stats(self, df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
        d = df[[self.key, self.value]].dropna()
        g = d.groupby(self.key)[self.value].agg(["sum", "count"])
        return g, float(d[self.value].mean())

    def _apply(self, keys: pd.Series, g: pd.DataFrame, gm: float) -> pd.DataFrame:
        s = keys.map(g["sum"])
        n = keys.map(g["count"]).fillna(0)
        mean = ((s.fillna(0) + self.k * gm) / (n + self.k))
        return pd.DataFrame({"hist_prep_mean": mean.astype(float),
                             "hist_restaurant_orders": n.astype(float)},
                            index=keys.index)

    def fit(self, train: pd.DataFrame) -> "RestaurantPrepHistory":
        self.table_, self.global_mean_ = self._stats(train)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.table_ is None:
            raise RuntimeError("call fit() first")
        return self._apply(df[self.key], self.table_, self.global_mean_)

    def fit_transform_oof(self, train: pd.DataFrame,
                          n_splits: int = config.PREP_OOF_FOLDS,
                          seed: int = config.RANDOM_STATE) -> pd.DataFrame:
        """Out-of-fold values for training rows; also fits on all of train."""
        out = pd.DataFrame(index=train.index,
                           columns=["hist_prep_mean", "hist_restaurant_orders"],
                           dtype=float)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        pos = np.arange(len(train))
        for fit_idx, val_idx in kf.split(pos):
            g, gm = self._stats(train.iloc[fit_idx])
            vals = self._apply(train[self.key].iloc[val_idx], g, gm)
            out.iloc[val_idx] = vals.to_numpy()
        self.fit(train)
        return out

    def in_sample_leaky(self, train: pd.DataFrame) -> pd.DataFrame:
        """v1-style values (row sees its own prep). Only for the leakage demo."""
        g, gm = self._stats(train)
        return self._apply(train[self.key], g, gm)
