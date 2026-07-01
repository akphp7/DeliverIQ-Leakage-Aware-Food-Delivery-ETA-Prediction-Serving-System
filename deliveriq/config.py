"""Central configuration for DeliverIQ v2.

Every path, constant and shared list lives here so that each phase script
uses exactly the same settings.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "raw" / "food_delivery.csv"
MODEL_DIR = ROOT / "models" / "v2"
OUT_DIR = ROOT / "outputs" / "v2"
SQL_DIR = ROOT / "sql"

RANDOM_STATE = 42
TEST_SIZE = 0.20          # same as v1, so the random split is identical
TARGET = "time_taken_min"

# Same peak-hour definition as v1 (lunch + dinner).
PEAK_HOURS = [12, 13, 14, 19, 20, 21, 22]

# Traffic has a natural order, so an ordinal code is meaningful.
TRAFFIC_LEVEL = {"Low": 0, "Medium": 1, "High": 2, "Jam": 3}

# Sentinel rider profiles found in the raw data (see Phase 0 notes):
# age 50 + rating 6 (53 rows) and age 15 + rating 1 (38 rows).
SENTINEL_RIDER_PROFILES = [(50.0, 6.0), (15.0, 1.0)]

# Plausible bounding box for India; coordinates outside it are invalid.
INDIA_LAT = (6.0, 37.5)
INDIA_LON = (68.0, 97.5)

# Smoothing strength for the historical restaurant prep-time encoder.
PREP_SMOOTHING_K = 5
PREP_OOF_FOLDS = 5


def phase_dir(phase: int) -> Path:
    """Output folder for a phase, created on first use."""
    p = OUT_DIR / f"phase{phase}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def model_dir() -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return MODEL_DIR
