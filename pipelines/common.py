"""Helpers shared by the phase scripts."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from deliveriq import config
from deliveriq.data.cleaning import load_clean
from deliveriq.features.build import base_features, add_history


def prepare(split_idx, clean=None, history_mode="oof"):
    """Clean data -> features for a given (train_idx, test_idx) split."""
    if clean is None:
        clean, _ = load_clean()
    tr, te = split_idx
    c_tr, c_te = clean.iloc[tr], clean.iloc[te]
    X_all = base_features(clean)
    enc, X_tr, (X_te,) = add_history(X_all.iloc[tr], c_tr, [(X_all.iloc[te], c_te)], mode=history_mode)
    y_tr = c_tr[config.TARGET].to_numpy(float)
    y_te = c_te[config.TARGET].to_numpy(float)
    return dict(clean=clean, c_tr=c_tr, c_te=c_te, X_tr=X_tr, X_te=X_te,
                y_tr=y_tr, y_te=y_te, encoder=enc)


def distance_bucket(km: pd.Series) -> pd.Series:
    return pd.cut(km, bins=[-np.inf, 3, 6, 10, np.inf],
                  labels=["0-3 km", "3-6 km", "6-10 km", ">10 km"])


def save_json(obj, path):
    path.write_text(json.dumps(obj, indent=2, default=str))


class Timer:
    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, *a):
        self.s = time.time() - self.t


def banner(text):
    print("\n" + "=" * 70 + f"\n{text}\n" + "=" * 70)
