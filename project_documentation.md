# Project Documentation

## 1. Project Overview and Business Goal

The project builds a 1-day-ahead demand forecast for each Store/Product series using historical demand and forecast-available operational and calendar inputs.

## 2. Dataset Summary

- Raw rows: 76000.
- Raw columns: 16.
- Series: 100 Store/Product combinations.
- Dates: 760 daily observations.
- Target: `Demand`.
- Training rows: 61200; test rows: 12000.
- Training range: `2022-01-29` to `2023-10-02`.
- Test range: `2023-10-03` to `2024-01-30`.

## 3. Critical Findings

- Same-day `Units Sold` had Pearson correlation 0.833421 with Demand, while lag-1 had 0.302939; same-day operational fields were excluded.
- `Price` had mutual information 0.334105 with Demand but near-zero linear correlation, indicating category/series-level structure.
- `Price` and `Competitor Pricing` had Pearson correlation 0.976648 and were partially addressed through `price_gap` and `price_ratio` after dropping `Competitor Pricing`.
- `Discount` and `Promotion` had Pearson correlation 0.784039.
- The test period includes both epidemic states.
- `Region` and `Seasonality` were dropped as redundant with `Store ID` and month.

## 4. Forecasting Setup and Assumptions

- Unit of forecast: 1-day-ahead for `(Store ID, Product ID)`.
- Calendar features and planned Price, Discount and Promotion are assumed available.
- Weather Condition, Competitor Pricing and Epidemic are assumed known from forecasts or announcements.
- Same-day Units Sold, Units Ordered and Inventory Level are never direct features.

## 5. Cleaning and Preprocessing Decisions

- Date was converted to datetime and observations were sorted per series.
- `series_id` was created from Store ID and Product ID.
- 27.631579% of rows had Units Sold greater than Demand; these rows and high-side outliers were retained.
- The first 28 days of each series were removed: 2800 rows.
- The 31 final feature list includes 27 numerical and 4 categorical features.
- The optional `price_relative_series_28` feature was added after improving CV MAE.
- Preprocessing used training-fit-only one-hot encoding and StandardScaler.

## 6. Modeling

### Baselines

- Moving Average 7 CV MAE: 32.164824.
- Naive Lag-1 CV MAE: 41.205078.
- Seasonal Lag-7 CV MAE: 43.335627.

### Comparison

| Model | MAE CV | RMSE CV | WAPE CV | R2 CV |
|---|---:|---:|---:|---:|
| LightGBM | 16.724148 +/- 1.884772 | 23.200741 +/- 2.595610 | 0.160077 +/- 0.023107 | 0.754698 +/- 0.052180 |
| XGBoost | 18.529101 +/- 1.532920 | 25.410153 +/- 2.155867 | 0.177015 +/- 0.018651 | 0.706390 +/- 0.051493 |
| HistGradientBoosting | 19.733529 +/- 1.298067 | 26.842348 +/- 1.911151 | 0.188518 +/- 0.017214 | 0.672675 +/- 0.052005 |
| Random Forest | 25.142407 +/- 1.462021 | 33.508579 +/- 2.034672 | 0.239965 +/- 0.017727 | 0.489544 +/- 0.079877 |
| Ridge | 26.417526 +/- 0.840093 | 34.423489 +/- 1.207646 | 0.252011 +/- 0.010831 | 0.462482 +/- 0.064059 |
| Linear Regression | 26.478861 +/- 0.753621 | 34.520934 +/- 1.031372 | 0.252604 +/- 0.010404 | 0.459069 +/- 0.065529 |
| Lasso | 27.763370 +/- 1.009476 | 36.162501 +/- 1.520428 | 0.264867 +/- 0.012827 | 0.407511 +/- 0.067277 |

### Experiments and Tuning

- Sqrt target outperformed raw target: 16.412240 versus 16.724148 CV MAE.
- With Price CV MAE: 16.724148; without Price CV MAE: 27.499129.
- Relative series Price feature improved CV MAE from 16.724148 to 4.552064.
- Tuned CV MAE: 4.381249.

### Final Test

- Final model: LightGBM.
- Test MAE: 3.710015.
- Test RMSE: 4.828937.
- Test WAPE: 0.037698.
- Test R2: 0.988430.
- No-Price test MAE: 7.592269.
- MAE improvement over Moving Average 7: 88.274186%.

## 7. Error Analysis and Feature Importance

- Overall mean residual: -0.120038.
- Epidemic mean residual: 0.189069.
- Non-epidemic mean residual: -0.227705.
- Actual/prediction and error plots are linked in [Findings_Report.md](Findings_Report.md).
- Permutation importance top features: `price_relative_series_28`, `Discount`, `demand_roll_mean_28`.

## 8. Deployment Artifacts

- `models/demand_model_eval.joblib`: pipeline trained on training set only for reported test evaluation.
- `models/demand_model.joblib`: the same pipeline refitted on train plus test for deployment.
- `models/model_metadata.json`: model parameters and deployed metadata.
- `data/history.csv`: full cleaned raw history.
- `data/train.csv` and `data/test.csv`: engineered split outputs.
- `predict.py`: deployment prediction interface.

Example:

```python
from predict import predict_demand

result = predict_demand(
    store_id="S001",
    product_id="P0001",
    date="2024-01-31",
    price=50.0,
    discount=10,
    promotion=1,
)
print(result)
```

The predict function accepts history dates for backtest mode and `2024-01-31` for next-day forecast mode.

## 9. Limitations, Risks and Next Steps

- The application supplies the latest observed Price automatically; Discount, Promotion, Weather and Epidemic can be edited in the forecast form.
- Unknown store/product pairs and unsupported dates return clear errors.
- Price is a potentially sensitive or structural feature and should be monitored in no-Price deployment comparisons.
- The application layer and end-to-end checks are implemented; monitor forecast behavior and update model metadata when the model is retrained.

## 10. Application (API & Frontend)

### Architecture

The browser dashboard loads model metadata and valid store-product pairs from FastAPI. The API loads the deployed model, feature configuration, metadata, and `data/history.csv` once during application lifespan. Predictions delegate to `predict.py`, which calls the shared `preprocessing.add_features()` implementation; the API does not duplicate the feature logic.

```text
frontend/index.html → FastAPI → predict.py → preprocessing.add_features()
                                      ├── models/demand_model.joblib
                                      ├── feature_config.json
                                      ├── models/model_metadata.json
                                      └── data/history.csv
```

### Endpoint reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | API status, model name, last history date, and next forecast date |
| `GET` | `/metadata` | CV/test metrics, improvement over baseline, top drivers, assumptions, weak spots, supported dates |
| `GET` | `/series` | Stores with valid products and category labels from metadata |
| `GET` | `/defaults` | Prefill values and their defaulting rules for a selected series and date |
| `POST` | `/predict` | One backtest or next-day prediction, effective inputs, default sources, and test-MAE error range |
| `GET` | `/history` | Daily actual demand and epidemic indicators for charting |
| `GET` | `/backtest` | Batched historical predictions and MAE for a date range (up to 120 inclusive days) |

Interactive request and response schemas are available at `/docs`.

### Defaults and prediction modes

Discount and Promotion default to the series' last observed values. Weather defaults to the most frequent value for the store's region in the selected calendar month. Epidemic defaults to the last known series value. The date's rule descriptions are returned by `/defaults`; the frontend marks each prefilled field and removes its tag when the user edits it. The model's latest observed Price and Competitor Pricing are supplied as non-editable context because the deployed feature configuration includes Price.

Dates in history use recorded historical inputs and return actual demand and error. The day after the last history date uses forecast defaults and any user-supplied overrides. Historical range backtests engineer features once and make one batch prediction call.

### Execution and verification

Install dependencies with `python -m pip install -r requirements.txt`, start the API with `python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload`, then serve `frontend/` with `python -m http.server 5500`. Run tests using `python -m pytest -q tests/test_api.py`.

The API test run completed with **8 passed**. It covers health, series, defaults, forecast and backtest predictions, batched range metrics, invalid inputs, and exact prediction agreement with `predict_demand()` for three historical rows. HTTP smoke checks returned 200 for health, series, defaults, forecast, backtest, and history; the intentionally invalid discount returned 422. Both `/docs` and the frontend root returned 200, and the dashboard rendered and completed a forecast and period backtest in the browser.
