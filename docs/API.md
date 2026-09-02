# API reference

The service is a FastAPI app in `deliveriq/api/app.py`, backed by `ETAService` in
`deliveriq/api/service.py`.

```bash
uvicorn deliveriq.api.app:app --port 8002
```

- Web UI: <http://localhost:8002>
- Interactive docs (Swagger): <http://localhost:8002/docs>

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | `{"status": "ok", "model_version": ...}`; 503 if the model cannot load |
| GET | `/model-info` | Model name, parameters, training period, test metrics, interval method, feature lists |
| POST | `/v2/predict` | One order → ETA, 80% range, top-3 reasons, warnings |
| POST | `/v2/predict/batch` | List of 1–500 orders → list of responses |

## Request body (`/v2/predict`)

| Field | Type | Required | Rules |
|---|---|---|---|
| `restaurant_latitude`, `restaurant_longitude` | float | yes | (0, 0) is accepted and replaced by the city centre |
| `delivery_latitude`, `delivery_longitude` | float | yes | |
| `order_date` | string | yes | `DD-MM-YYYY` |
| `time_ordered` | string | yes | `HH:MM` |
| `city_code` | string | no | e.g. `BANG`; used when restaurant coordinates are invalid |
| `delivery_person_age` | float | no | 18–70 |
| `delivery_person_rating` | float | no | 1–5 |
| `vehicle_condition` | float | no | 0–3 |
| `multiple_deliveries` | float | no | 0–5 |
| `weather_conditions` | string | no | `Sunny`, `Cloudy`, `Fog`, `Windy`, `Stormy`, `Sandstorms` or null |
| `road_traffic_density` | string | no | `Low`, `Medium`, `High`, `Jam` or null |
| `type_of_order` | string | no | default `Meal` |
| `type_of_vehicle` | string | no | default `motorcycle` |
| `festival` | string | no | default `No` |
| `city` | string | no | city type, default `Metropolitian` (spelling as in the dataset) |

Unknown optional fields may be sent as `null`; the model treats them as missing, and the
response explains the effect in `warnings`.

## Example

```bash
curl -X POST http://localhost:8002/v2/predict \
  -H "Content-Type: application/json" \
  -d '{"restaurant_latitude": 12.914264, "restaurant_longitude": 77.6784,
       "delivery_latitude": 13.004264, "delivery_longitude": 77.7684,
       "order_date": "25-03-2022", "time_ordered": "19:45", "city_code": "BANG",
       "delivery_person_age": 27, "delivery_person_rating": 4.7,
       "vehicle_condition": 2, "multiple_deliveries": 1,
       "weather_conditions": "Cloudy", "road_traffic_density": "Jam"}'
```

PowerShell:

```powershell
$body = @{restaurant_latitude=12.914264; restaurant_longitude=77.6784;
          delivery_latitude=13.004264; delivery_longitude=77.7684;
          order_date="25-03-2022"; time_ordered="19:45"; city_code="BANG";
          delivery_person_age=27; delivery_person_rating=4.7; vehicle_condition=2;
          multiple_deliveries=1; weather_conditions="Cloudy"; road_traffic_density="Jam"} | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8002/v2/predict -Method Post -Body $body -ContentType "application/json"
```

Response:

```json
{
  "eta_minutes": 35.5,
  "eta_range_80": [30.1, 42.2],
  "top_reasons": [
    {"factor": "Weather", "minutes": 6.2},
    {"factor": "Distance x traffic", "minutes": 4.4},
    {"factor": "Rider age", "minutes": -2.4}
  ],
  "baseline_minutes": 26.2,
  "warnings": [],
  "source": "model",
  "model_version": "deliveriq-v2-xgb-cqr",
  "latency_ms": 82
}
```

| Field | Meaning |
|---|---|
| `eta_minutes` | Point prediction (tuned XGBoost), at least 1 minute |
| `eta_range_80` | Conformalized quantile range; about 80% of orders fall inside |
| `top_reasons` | Three largest SHAP contributions, grouped by feature; minutes added (+) or removed (−) versus `baseline_minutes` |
| `baseline_minutes` | Average model prediction (SHAP expected value) |
| `warnings` | Data-quality notes that affect the estimate |
| `source` | `model`, or `fallback` if the model raised an error |
| `latency_ms` | Service time for this request (varies by machine) |

## Same order with traffic and rider rating unknown

```json
{
  "eta_minutes": 27.3,
  "eta_range_80": [22.6, 35.4],
  "top_reasons": [
    {"factor": "Weather", "minutes": 3.0},
    {"factor": "Distance", "minutes": 2.6},
    {"factor": "Rider age", "minutes": -2.1}
  ],
  "baseline_minutes": 26.2,
  "warnings": ["traffic unknown: range widened", "rider profile missing or invalid"],
  "source": "model",
  "model_version": "deliveriq-v2-xgb-cqr",
  "latency_ms": 69
}
```

## Warnings

| Warning | Trigger |
|---|---|
| `traffic unknown: range widened` | `road_traffic_density` is null |
| `weather unknown` | `weather_conditions` is null |
| `rider profile missing or invalid` | age or rating missing, or a placeholder profile |
| `restaurant location invalid: city centre used` | restaurant coordinates outside India, e.g. (0, 0) |
| `distance unknown` | distance could not be computed |
| `low confidence: wide range` | range wider than 15 minutes |

## Errors

Invalid input returns HTTP 422 with a description, for example:

```json
{"detail": [{"type": "value_error", "loc": ["body", "road_traffic_density"],
             "msg": "Value error, must be one of ['High', 'Jam', 'Low', 'Medium'] or null",
             "input": "Heavy"}]}
```

A batch outside 1–500 orders also returns 422.

## Fallback

If prediction fails (for example a corrupted model file), the service answers with the
training-period median for the order's traffic level (Low 20, Medium 26, High 27, Jam 31 min,
otherwise 26), a ±6 minute range and `source: "fallback"` instead of an error.
The table is `models/v2/fallback.json`, written by `pipelines/p5_production.py`.

## Known limitation

The model has seen deliveries up to about 21 km. Tree models do not extrapolate, so any longer
distance receives the same ETA as ~22 km. The API does not yet reject out-of-range distances.
