"""Robust parsing of the messy clock-time columns.

The raw file stores clock times in three shapes:
  * "21:55"            -> normal HH:MM
  * "24:05:00"         -> HH:MM:SS, sometimes with hour 24 (= 00:05)
  * "0.458333333"      -> Excel fraction of a day (0.458333 * 1440 = 660 = 11:00)
v1 only understood HH:MM, so 4,068 order times silently became NaN.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MINUTES_PER_DAY = 1440
_CLOCK_RE = r"^\d{1,2}:\d{2}(:\d{2})?$"


def parse_clock_minutes(values: pd.Series) -> pd.Series:
    """Return minutes after midnight (0-1439) as float; NaN when unparseable."""
    s = values.astype("string").str.strip()
    out = pd.Series(np.nan, index=s.index, dtype="float64")

    is_clock = s.str.match(_CLOCK_RE, na=False)
    if is_clock.any():
        parts = s[is_clock].str.split(":", expand=True)
        hours = parts[0].astype(int)
        mins = parts[1].astype(int)
        valid = (hours <= 24) & (mins < 60)
        minutes = (hours * 60 + mins) % MINUTES_PER_DAY
        out.loc[minutes[valid].index] = minutes[valid].astype(float)

    frac = pd.to_numeric(s[~is_clock], errors="coerce")
    ok = frac.notna() & (frac >= 0) & (frac <= 1)
    if ok.any():
        out.loc[frac[ok].index] = (frac[ok] * MINUTES_PER_DAY).round() % MINUTES_PER_DAY

    return out


def minutes_between(start_min: pd.Series, end_min: pd.Series) -> pd.Series:
    """Elapsed minutes from start to end, wrapping past midnight."""
    return (end_min - start_min) % MINUTES_PER_DAY
