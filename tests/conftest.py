import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def clean_data():
    from deliveriq.data.cleaning import load_clean
    return load_clean()
