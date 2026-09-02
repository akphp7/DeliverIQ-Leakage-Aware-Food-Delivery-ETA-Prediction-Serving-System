"""End-to-end API checks (needs models/v2/final_bundle.joblib)."""
import pytest
from fastapi.testclient import TestClient

from deliveriq import config

pytestmark = pytest.mark.skipif(not (config.MODEL_DIR / "final_bundle.joblib").exists(),
                                reason="run python -m pipelines.p2_models first")

ORDER = {
    "restaurant_latitude": 12.914264, "restaurant_longitude": 77.6784,
    "delivery_latitude": 13.004264, "delivery_longitude": 77.7684,
    "order_date": "25-03-2022", "time_ordered": "19:45", "city_code": "BANG",
    "delivery_person_age": 27, "delivery_person_rating": 4.7,
    "vehicle_condition": 2, "multiple_deliveries": 1,
    "weather_conditions": "Cloudy", "road_traffic_density": "Jam",
}


@pytest.fixture(scope="module")
def client():
    from deliveriq.api.app import app
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_shape(client):
    r = client.post("/v2/predict", json=ORDER)
    assert r.status_code == 200
    d = r.json()
    lo, hi = d["eta_range_80"]
    assert lo <= d["eta_minutes"] <= hi
    assert 5 < d["eta_minutes"] < 60
    assert len(d["top_reasons"]) == 3
    assert d["source"] == "model"


def test_jam_slower_than_low(client):
    low = client.post("/v2/predict", json={**ORDER, "road_traffic_density": "Low",
                                           "time_ordered": "09:30"}).json()
    jam = client.post("/v2/predict", json=ORDER).json()
    assert jam["eta_minutes"] > low["eta_minutes"]


def test_unknown_traffic_widens_range(client):
    known = client.post("/v2/predict", json=ORDER).json()
    unknown = client.post("/v2/predict", json={**ORDER, "road_traffic_density": None}).json()
    w = lambda d: d["eta_range_80"][1] - d["eta_range_80"][0]
    assert w(unknown) > w(known)
    assert any("traffic unknown" in m for m in unknown["warnings"])


def test_zero_coordinates_use_city_centre(client):
    d = client.post("/v2/predict", json={**ORDER, "restaurant_latitude": 0, "restaurant_longitude": 0,
                                         "delivery_latitude": 0.09, "delivery_longitude": 0.09}).json()
    assert any("city centre" in m for m in d["warnings"])
    assert 5 < d["eta_minutes"] < 60


def test_validation_errors(client):
    assert client.post("/v2/predict", json={**ORDER, "road_traffic_density": "Heavy"}).status_code == 422
    assert client.post("/v2/predict", json={**ORDER, "order_date": "2022-03-25"}).status_code == 422
    assert client.post("/v2/predict", json={**ORDER, "delivery_person_rating": 6}).status_code == 422


def test_batch(client):
    r = client.post("/v2/predict/batch", json=[ORDER, {**ORDER, "road_traffic_density": "Low"}])
    assert r.status_code == 200 and len(r.json()) == 2
