"""Guards against the leakage mistakes discussed in the journal."""
import numpy as np
import pandas as pd

from deliveriq import config
from deliveriq.evaluation.splits import rolling_origin_splits, temporal_split
from deliveriq.features.build import FINAL_DROP, NUMERIC, CATEGORICAL, feature_columns
from deliveriq.features.prep_history import RestaurantPrepHistory


def _toy():
    # restaurant A: prep 5 or 15; restaurant B: prep 10
    return pd.DataFrame({
        "restaurant_id": ["A"] * 10 + ["B"] * 10,
        "pickup_delay_min": [5, 15] * 5 + [10] * 10,
    })


def test_oof_value_never_sees_own_row():
    df = _toy()
    enc = RestaurantPrepHistory(k=0)
    oof = enc.fit_transform_oof(df, n_splits=5, seed=0)
    leaky = enc.in_sample_leaky(df)
    # in-sample: every A row gets exactly 10; out-of-fold values move with the held-out rows
    assert np.allclose(leaky.loc[:9, "hist_prep_mean"], 10)
    assert not np.allclose(oof.loc[:9, "hist_prep_mean"], 10)
    assert (oof["hist_restaurant_orders"] < 10).all()


def test_unseen_restaurant_gets_global_mean():
    enc = RestaurantPrepHistory(k=5).fit(_toy())
    out = enc.transform(pd.DataFrame({"restaurant_id": ["NEW"]}))
    assert out["hist_prep_mean"].iloc[0] == enc.global_mean_
    assert out["hist_restaurant_orders"].iloc[0] == 0


def test_no_future_information_in_features():
    for col in NUMERIC + CATEGORICAL:
        assert "pickup" not in col
        assert col != config.TARGET


def test_final_model_drops_history():
    num, _ = feature_columns(FINAL_DROP)
    assert not set(FINAL_DROP) & set(num)


def test_temporal_split_is_ordered(clean_data):
    clean, _ = clean_data
    tr, te, cutoff, _ = temporal_split(clean)
    assert clean.iloc[tr].order_date.max() < cutoff <= clean.iloc[te].order_date.min()
    assert len(set(tr) & set(te)) == 0


def test_rolling_folds_only_use_the_past(clean_data):
    clean, _ = clean_data
    for tr, te, start, _ in rolling_origin_splits(clean):
        assert clean.iloc[tr].order_date.max() < start


def test_sql_history_uses_previous_days_only(clean_data):
    from pipelines.p4_sql_features import leakage_check
    from deliveriq.sqlfeatures.duck import history_features
    clean, _ = clean_data
    hist = history_features(clean)
    chk = leakage_check(clean, hist, n=50)
    assert chk["match"].all()
    first_day = clean.order_date == clean.order_date.min()
    assert (hist.loc[first_day, "rider_prev_orders"] == 0).all()
