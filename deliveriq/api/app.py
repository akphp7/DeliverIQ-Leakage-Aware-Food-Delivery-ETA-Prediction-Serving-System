"""DeliverIQ v2 API (FastAPI).

Run from the project root:
    uvicorn deliveriq.api.app:app --port 8002
Then open http://localhost:8002 (UI) or http://localhost:8002/docs (Swagger).
The v1 API (api/main.py, port 8001) is unchanged.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from deliveriq import config
from deliveriq.api.service import MODEL_VERSION, ETAService, OrderInput

app = FastAPI(title="DeliverIQ ETA API v2", version="2.0.0",
              description="Order-time ETA with an 80% range and the top reasons.")

TRAFFIC = {"Low", "Medium", "High", "Jam"}
WEATHER = {"Sunny", "Cloudy", "Fog", "Windy", "Stormy", "Sandstorms"}


class OrderRequest(BaseModel):
    restaurant_latitude: float = Field(..., examples=[12.914264])
    restaurant_longitude: float = Field(..., examples=[77.6784])
    delivery_latitude: float = Field(..., examples=[13.004264])
    delivery_longitude: float = Field(..., examples=[77.7684])
    order_date: str = Field(..., pattern=r"^\d{2}-\d{2}-\d{4}$", examples=["25-03-2022"])
    time_ordered: str = Field(..., pattern=r"^\d{1,2}:\d{2}$", examples=["19:45"])
    city_code: Optional[str] = Field(None, examples=["BANG"])
    delivery_person_age: Optional[float] = Field(None, ge=18, le=70, examples=[27])
    delivery_person_rating: Optional[float] = Field(None, ge=1, le=5, examples=[4.7])
    vehicle_condition: Optional[float] = Field(None, ge=0, le=3, examples=[2])
    multiple_deliveries: Optional[float] = Field(None, ge=0, le=5, examples=[1])
    weather_conditions: Optional[str] = Field(None, examples=["Cloudy"])
    road_traffic_density: Optional[str] = Field(None, examples=["Jam"])
    type_of_order: str = "Meal"
    type_of_vehicle: str = "motorcycle"
    festival: str = "No"
    city: Optional[str] = "Metropolitian"

    @field_validator("road_traffic_density")
    @classmethod
    def _traffic(cls, v):
        if v is not None and v not in TRAFFIC:
            raise ValueError(f"must be one of {sorted(TRAFFIC)} or null")
        return v

    @field_validator("weather_conditions")
    @classmethod
    def _weather(cls, v):
        if v is not None and v not in WEATHER:
            raise ValueError(f"must be one of {sorted(WEATHER)} or null")
        return v


class Reason(BaseModel):
    factor: str
    minutes: float


class ETAResponse(BaseModel):
    eta_minutes: float
    eta_range_80: list[float]
    top_reasons: list[Reason]
    baseline_minutes: float
    warnings: list[str]
    source: str
    model_version: str
    latency_ms: float


@lru_cache(maxsize=1)
def service() -> ETAService:
    if not (config.MODEL_DIR / "final_bundle.joblib").exists():
        raise RuntimeError("models/v2/final_bundle.joblib missing: run python -m pipelines.p2_models")
    return ETAService()


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(config.ROOT / "frontend" / "v2" / "index.html")


@app.get("/health")
def health():
    try:
        service()
        return {"status": "ok", "model_version": MODEL_VERSION}
    except Exception as exc:
        raise HTTPException(503, f"model not loaded: {exc}")


@app.get("/model-info")
def model_info():
    b = service().bundle
    return {"model_version": MODEL_VERSION, "point_model": b["point_model_name"],
            "params": b["xgb_params"], "train_period": b["train_period"],
            "test_metrics": {k: round(float(v), 3) for k, v in b["test_metrics"].items()},
            "interval": "CQR 80% (conformalized XGBoost quantiles)",
            "features": {"numeric": b["numeric"], "categorical": b["categorical"]}}


@app.post("/v2/predict", response_model=ETAResponse)
def predict(req: OrderRequest):
    return service().predict([OrderInput(**req.model_dump())])[0]


@app.post("/v2/predict/batch", response_model=list[ETAResponse])
def predict_batch(reqs: list[OrderRequest]):
    if not reqs or len(reqs) > 500:
        raise HTTPException(422, "send between 1 and 500 orders")
    return service().predict([OrderInput(**r.model_dump()) for r in reqs])
