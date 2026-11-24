# Demand Forecasting

A 1-day-ahead retail demand forecasting application. The FastAPI backend reuses the fitted model and the project's existing feature pipeline; the single-page dashboard lets users select a store-product series, review defaults and forecast, inspect recent history, and run a historical backtest.

The model uses its latest observed price automatically. Price is not exposed as a form field; same-day operational fields that would leak the target are excluded from the feature set.

## Project structure

```text
backend/main.py             FastAPI application
frontend/index.html         Responsive dashboard (single self-contained page)
tests/test_api.py           API and prediction-consistency tests
predict.py                  Shared prediction and batch-backtest functions
preprocessing.py            Shared feature engineering and preprocessing
feature_config.json         Model feature schema
models/
  demand_model.joblib       Deployment model
  model_metadata.json       Metrics, valid series, assumptions, and defaults
data/history.csv            Source history for charts and forecasting
requirements.txt            Runtime, analysis, and test dependencies
```

## Install and run

From the project root, install the project dependencies:

```powershell
python -m pip install -r requirements.txt
```

Start the backend in one terminal:

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Start the frontend in a second terminal:

```powershell
Set-Location .\frontend
python -m http.server 5500
```

Open the dashboard at <http://localhost:5500> and the interactive API documentation at <http://localhost:8000/docs>. The backend allows the `localhost:5500` and `127.0.0.1:5500` frontend origins.

Run the API tests from the project root:

```powershell
python -m pytest -q tests/test_api.py
```

## API

All endpoints are served from `http://localhost:8000`.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Readiness, model name, last history date, and next forecast date |
| `GET` | `/metadata` | CV/test metrics, baseline improvement, top drivers, assumptions, weak spots, and supported dates |
| `GET` | `/series` | Stores and valid products with categories |
| `GET` | `/defaults` | Prefill values and the rule used for each field |
| `POST` | `/predict` | Backtest or next-day prediction with inputs, defaults, and typical error range |
| `GET` | `/history` | Actual daily demand and epidemic flags for a series |
| `GET` | `/backtest` | Batched predicted-versus-actual range and range MAE |

### Example requests and responses

Check readiness:

```powershell
curl.exe http://localhost:8000/health
```

```json
{"status":"ok","model_name":"LightGBM","last_history_date":"2024-01-30","forecast_date":"2024-01-31"}
```

Get defaults for a series and planning date:

```powershell
curl.exe "http://localhost:8000/defaults?store_id=S001&product_id=P0001&date=2024-01-31"
```

```json
{
  "values": {"discount": 10.0, "promotion": 0, "weather_condition": "Cloudy", "epidemic": 0},
  "rules": {
    "discount": "Last observed discount and promotion values for the series",
    "promotion": "Last observed discount and promotion values for the series",
    "weather_condition": "Most frequent weather for the store's Region in the requested calendar month",
    "epidemic": "Last known epidemic value for the series"
  },
  "model_context": {"price": 71.56, "competitor_pricing": 66.66}
}
```

Request a next-day forecast (omitted inputs are defaulted):

```powershell
curl.exe --% -H "Content-Type: application/json" -d "{\"store_id\":\"S001\",\"product_id\":\"P0001\",\"date\":\"2024-01-31\"}" http://localhost:8000/predict
```

The executed forecast returned `prediction: 114.43`, `raw_prediction: 114.43266759426413`, and a typical error range of `110.72265214343707` to `118.14268304509119` using the metadata test MAE. The response also includes the effective inputs, their sources, and the backtest/forecast mode.

Run a short range backtest:

```powershell
curl.exe "http://localhost:8000/backtest?store_id=S001&product_id=P0001&start=2023-12-01&end=2023-12-03"
```

The executed range response contained three daily predictions and an MAE of `7.430180614899712`. Backtest requests are limited to a maximum of 120 inclusive calendar days and are predicted in one batch.

Invalid prediction inputs receive a clear validation response (HTTP 400 for unsupported series/date and HTTP 422 for invalid request fields).

## Dashboard screenshots

![Dashboard overview placeholder](screenshots/dashboard-overview.png)

![Prediction and backtest charts placeholder](screenshots/prediction-backtest.png)
