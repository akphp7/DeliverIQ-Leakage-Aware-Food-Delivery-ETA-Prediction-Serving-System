"""Parsing and cleaning behave as documented in the journal (Phase 0)."""
import numpy as np
import pandas as pd
import pytest

from deliveriq.data.parsing import minutes_between, parse_clock_minutes


@pytest.mark.parametrize("raw, expected", [
    ("21:55", 21 * 60 + 55),
    ("9:05", 9 * 60 + 5),
    ("24:05:00", 5),              # hour 24 means just after midnight
    ("0.458333333", 660),         # Excel day fraction -> 11:00
    ("1", 0),                     # a full day -> midnight
    ("NaN", np.nan),
    ("25:99", np.nan),
    (None, np.nan),
])
def test_parse_clock(raw, expected):
    got = parse_clock_minutes(pd.Series([raw])).iloc[0]
    if np.isnan(expected):
        assert np.isnan(got)
    else:
        assert got == expected


def test_minutes_between_wraps_midnight():
    assert minutes_between(pd.Series([23 * 60 + 55]), pd.Series([5])).iloc[0] == 10


def test_cleaning_keeps_every_order(clean_data):
    clean, audit = clean_data
    assert len(clean) == audit["rows"] == 45584
    assert clean["order_minute_of_day"].notna().all()
    assert clean["distance_km"].notna().all()


def test_audit_counts(clean_data):
    _, audit = clean_data
    assert audit["order_time_recovered_from_fraction"] == 4068
    assert audit["order_time_imputed_from_pickup"] == 1731
    assert audit["distance_recovered_with_city_centroid"] == 3640
    assert audit["sentinel_rider_rows"] == 91


def test_sentinel_riders_are_blanked(clean_data):
    clean, _ = clean_data
    bad = clean[clean.rider_profile_invalid == 1]
    assert bad[["rider_age", "rider_rating"]].isna().all().all()
    assert clean["rider_rating"].dropna().between(1, 5).all()


def test_restaurant_id_has_one_location(clean_data):
    clean, _ = clean_data
    ok = clean[clean.restaurant_coords_missing == 0]
    per_id = ok.groupby("restaurant_id")["restaurant_lat"].nunique()
    assert (per_id == 1).all()


def test_distances_are_plausible(clean_data):
    clean, _ = clean_data
    assert clean["distance_km"].between(0.5, 25).all()
