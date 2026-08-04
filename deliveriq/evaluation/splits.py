"""Train/test split strategies.

random_split   -> identical rows to v1 (same size, seed and row order)
temporal_split -> train on earlier days, test on the last days (Phase 1)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from deliveriq import config


def random_split(clean: pd.DataFrame):
    idx = np.arange(len(clean))
    tr, te = train_test_split(idx, test_size=config.TEST_SIZE,
                              random_state=config.RANDOM_STATE)
    return tr, te


def temporal_split(clean: pd.DataFrame, test_days: int | None = None,
                   test_fraction: float = config.TEST_SIZE):
    """Hold out the most recent calendar days.

    If test_days is None, pick the smallest number of final days that holds at
    least `test_fraction` of orders, so the test set is comparable in size to
    the random split.
    """
    dates = clean["order_date"].dt.normalize()
    days = np.sort(dates.unique())
    if test_days is None:
        counts = dates.value_counts().sort_index()
        cum = counts[::-1].cumsum()
        test_days = int((cum < test_fraction * len(clean)).sum()) + 1
    cutoff = days[-test_days]
    tr = np.where(dates < cutoff)[0]
    te = np.where(dates >= cutoff)[0]
    return tr, te, pd.Timestamp(cutoff), test_days


def rolling_origin_splits(clean: pd.DataFrame, n_folds: int = 4, block_days: int = 4):
    """Expanding-window time-series CV.

    Fold k trains on every day before its test block and tests on the next
    `block_days` days. Blocks are taken from the end of the data. An even
    block length keeps the alternating fast/slow days balanced.
    """
    dates = clean["order_date"].dt.normalize()
    days = np.sort(dates.unique())
    folds = []
    for k in range(n_folds, 0, -1):
        start = len(days) - k * block_days
        test_days = days[start:start + block_days]
        tr = np.where(dates < test_days[0])[0]
        te = np.where(dates.isin(test_days))[0]
        folds.append((tr, te, pd.Timestamp(test_days[0]), pd.Timestamp(test_days[-1])))
    return folds


def kfold_splits(clean: pd.DataFrame, n_splits: int = 5):
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
    return list(kf.split(np.arange(len(clean))))
