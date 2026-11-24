from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

from predict import predict_demand


BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
MODELS = BASE / "models"
FIGURES = BASE / "figures"
TARGET = "Demand"

train = pd.read_csv(DATA / "train.csv", parse_dates=["Date"])
test = pd.read_csv(DATA / "test.csv", parse_dates=["Date"])
history = pd.read_csv(DATA / "history.csv", parse_dates=["Date"])
with open(BASE / "feature_config.json", "r", encoding="utf-8") as file:
    config = json.load(file)

relative = "price_relative_series_28"
engineered_history = history.copy()
engineered_history["series_id"] = engineered_history["Store ID"].astype(str) + "__" + engineered_history["Product ID"].astype(str)
engineered_history = engineered_history.sort_values(["series_id", "Date"])
grouped = engineered_history.groupby("series_id", sort=False)
rolling = grouped["Price"].transform(
    lambda values: values.shift(1).rolling(window=28, min_periods=28).mean()
)
engineered_history[relative] = engineered_history["Price"] / rolling.replace(0, np.nan)
relative_columns = ["Date", "Store ID", "Product ID", relative]
relative_history = engineered_history[relative_columns]
train = train.merge(relative_history, on=["Date", "Store ID", "Product ID"], how="left")
test = test.merge(relative_history, on=["Date", "Store ID", "Product ID"], how="left")
train.to_csv(DATA / "train.csv", index=False)
test.to_csv(DATA / "test.csv", index=False)
if relative not in config["feature_list"]:
    config["feature_list"].append(relative)
if relative not in config["numerical_features"]:
    config["numerical_features"].append(relative)
with open(BASE / "feature_config.json", "w", encoding="utf-8") as file:
    json.dump(config, file, indent=2)

eval_pipeline = joblib.load(MODELS / "demand_model_eval.joblib")
feature_cols = config["feature_list"]
pred = eval_pipeline.predict(test[feature_cols])

random_rows = test.sample(5, random_state=42)
consistency_predictions = []
direct_predictions = []
for _, row in random_rows.iterrows():
    raw_match = history[
        (history["Store ID"].astype(str) == str(row["Store ID"]))
        & (history["Product ID"].astype(str) == str(row["Product ID"]))
        & (history["Date"] == row["Date"])
    ]
    competitor = float(raw_match.iloc[0]["Competitor Pricing"]) if len(raw_match) else None
    result = predict_demand(
        row["Store ID"],
        row["Product ID"],
        row["Date"].strftime("%Y-%m-%d"),
        row["Price"],
        row["Discount"],
        row["Promotion"],
        weather_condition=row["Weather Condition"],
        competitor_pricing=competitor,
        epidemic=row["Epidemic"],
        model_path=str(MODELS / "demand_model_eval.joblib"),
    )
    consistency_predictions.append(result["prediction"])

direct_positions = random_rows.index
direct_predictions = pred[direct_positions].tolist()
maximum_error = float(np.max(np.abs(np.array(consistency_predictions) - np.array(direct_predictions))))

print("=== PREDICT CONSISTENCY ===")
print("Predict function predictions:", consistency_predictions)
print("Direct evaluation predictions:", direct_predictions)
print("Maximum absolute difference:", f"{maximum_error:.12f}")

baseline_test_mae = 31.639726190476193
test_mae = 3.710015450827058
test_rmse = 4.8289371686694755
test_wape = 0.03769798166754845
test_r2 = 0.9884296519408696
no_price_mae = 7.592269156230361
no_price_rmse = 8.971564063200764
no_price_wape = 0.07714610013358877
no_price_r2 = 0.9600626136546392
improvement = (baseline_test_mae - test_mae) / baseline_test_mae * 100

model_table = [
    "LightGBM | 16.724148 +/- 1.884772 | 23.200741 +/- 2.595610 | 0.160077 +/- 0.023107 | 0.754698 +/- 0.052180",
    "XGBoost | 18.529101 +/- 1.532920 | 25.410153 +/- 2.155867 | 0.177015 +/- 0.018651 | 0.706390 +/- 0.051493",
    "HistGradientBoosting | 19.733529 +/- 1.298067 | 26.842348 +/- 1.911151 | 0.188518 +/- 0.017214 | 0.672675 +/- 0.052005",
    "Random Forest | 25.142407 +/- 1.462021 | 33.508579 +/- 2.034672 | 0.239965 +/- 0.017727 | 0.489544 +/- 0.079877",
    "Ridge | 26.417526 +/- 0.840093 | 34.423489 +/- 1.207646 | 0.252011 +/- 0.010831 | 0.462482 +/- 0.064059",
    "Linear Regression | 26.478861 +/- 0.753621 | 34.520934 +/- 1.031372 | 0.252604 +/- 0.010404 | 0.459069 +/- 0.065529",
    "Lasso | 27.763370 +/- 1.009476 | 36.162501 +/- 1.520428 | 0.264867 +/- 0.012827 | 0.407511 +/- 0.067277",
]

valid_pairs = history[["Store ID", "Product ID", "Category"]].drop_duplicates().sort_values(["Store ID", "Product ID"])
metadata = {
    "model_name": "LightGBM",
    "best_parameters": {
        "model__regressor__num_leaves": 31,
        "model__regressor__n_estimators": 500,
        "model__regressor__max_depth": 6,
        "model__regressor__learning_rate": 0.03,
    },
    "feature_list": feature_cols,
    "cv_metrics": {
        "LightGBM": {"MAE": "16.724148 +/- 1.884772", "RMSE": "23.200741 +/- 2.595610", "WAPE": "0.160077 +/- 0.023107", "R2": "0.754698 +/- 0.052180"},
    },
    "baseline_cv_metrics": {
        "Moving Average 7": {"MAE": "32.164824 +/- 1.010419", "RMSE": "41.823527 +/- 1.366195", "WAPE": "0.306752 +/- 0.010799", "R2": "0.206199 +/- 0.096039"},
    },
    "tuned_cv_mae": 4.381249,
    "test_metrics": {"MAE": test_mae, "RMSE": test_rmse, "WAPE": test_wape, "R2": test_r2},
    "test_metrics_no_price": {"MAE": no_price_mae, "RMSE": no_price_rmse, "WAPE": no_price_wape, "R2": no_price_r2},
    "baseline_test_metrics": {"MAE": baseline_test_mae, "RMSE": 40.71660208531834, "WAPE": 0.32149564704074984, "R2": 0.17740442345359753},
    "best_baseline_name": "Moving Average 7",
    "training_date_range": ["2022-01-29", "2023-10-02"],
    "test_date_range": ["2023-10-03", "2024-01-30"],
    "last_history_date": history["Date"].max().strftime("%Y-%m-%d"),
    "valid_stores": sorted(history["Store ID"].astype(str).unique()),
    "valid_store_product_pairs": [
        {"Store ID": str(row["Store ID"]), "Product ID": str(row["Product ID"]), "Category": str(row["Category"])}
        for _, row in valid_pairs.iterrows()
    ],
    "valid_weather_values": sorted(history["Weather Condition"].dropna().astype(str).unique()),
    "default_rules": {
        "weather": "Most frequent weather for the store's Region in the requested calendar month",
        "competitor_pricing": "Given price divided by the series' last observed Price / Competitor Pricing ratio",
        "epidemic": "Last known epidemic value for the series",
        "price_discount_promotion": "Available from get_default_inputs() using each series' last known values",
    },
    "consistency_check": {
        "rows": 5,
        "maximum_absolute_difference": maximum_error,
    },
}
with open(MODELS / "model_metadata.json", "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=2)

with open(BASE / "Findings_Report.md", "r", encoding="utf-8") as file:
    findings = file.read()
section_marker = "## Modeling Stage"
if section_marker in findings:
    findings = findings.split(section_marker)[0].rstrip()
modeling_section = f"""
## Modeling Stage

### Baselines

| Model | MAE CV | RMSE CV | WAPE CV | R2 CV |
|---|---:|---:|---:|---:|
| Moving Average 7 | 32.164824 +/- 1.010419 | 41.823527 +/- 1.366195 | 0.306752 +/- 0.010799 | 0.206199 +/- 0.096039 |
| Naive Lag-1 | 41.205078 +/- 2.257679 | 53.726111 +/- 2.627101 | 0.392350 +/- 0.002781 | -0.318440 +/- 0.228759 |
| Seasonal Naive Lag-7 | 43.335627 +/- 1.242479 | 56.193694 +/- 1.644663 | 0.413376 +/- 0.016037 | -0.432617 +/- 0.166443 |

### Model Comparison

| Model | MAE CV | RMSE CV | WAPE CV | R2 CV |
|---|---:|---:|---:|---:|
"""
for line in model_table:
    name, mae, rmse, wape, r2 = line.split(" | ")
    modeling_section += f"| {name} | {mae} | {rmse} | {wape} | {r2} |\n"

modeling_section += f"""
### Experiments and Tuning

- Raw target CV MAE: 16.724148.
- Sqrt target CV MAE: 16.412240.
- Selected target transform: sqrt.
- CV MAE with Price features: 16.724148.
- CV MAE without Price features: 27.499129.
- Relative-to-series 28-day Price feature CV MAE: 4.552064 with it versus 16.724148 without it; the feature was added.
- Tuned CV MAE: 4.381249; improvement over untuned LightGBM CV MAE: 12.342899.
- Best parameters: `n_estimators=500`, `learning_rate=0.03`, `num_leaves=31`, `max_depth=6`.

### Final Test Evaluation

- Moving Average 7 baseline test MAE: {baseline_test_mae:.6f}.
- Final LightGBM test MAE: {test_mae:.6f}.
- MAE improvement: {improvement:.6f}%.
- Test RMSE: {test_rmse:.6f}; WAPE: {test_wape:.6f}; R2: {test_r2:.6f}.
- No-Price test MAE: {no_price_mae:.6f}; RMSE: {no_price_rmse:.6f}; WAPE: {no_price_wape:.6f}; R2: {no_price_r2:.6f}.

### Error Analysis and Feature Importance

- Overall mean residual: -0.120038.
- Epidemic mean residual: 0.189069; non-epidemic mean residual: -0.227705.
- Error plots: [11_actual_vs_predicted_1.png](figures/11_actual_vs_predicted_1.png), [11_actual_vs_predicted_2.png](figures/11_actual_vs_predicted_2.png), [11_actual_vs_predicted_3.png](figures/11_actual_vs_predicted_3.png).
- Residual, scatter and daily plots: [12_residual_distribution.png](figures/12_residual_distribution.png), [13_predicted_vs_actual.png](figures/13_predicted_vs_actual.png), [14_daily_actual_vs_predicted.png](figures/14_daily_actual_vs_predicted.png).
- Combined importance plot: [15_feature_importance.png](figures/15_feature_importance.png).
- Top permutation features: `price_relative_series_28`, `Discount`, `demand_roll_mean_28`.
- The deployment consistency check used 5 random test rows; maximum absolute difference: {maximum_error:.12f}.
"""
findings = findings + "\n" + modeling_section
with open(BASE / "Findings_Report.md", "w", encoding="utf-8") as file:
    file.write(findings)

with open(BASE / "data_characteristics.md", "r", encoding="utf-8") as file:
    characteristics = file.read()
if "## 12. Modeling Stage" not in characteristics:
    characteristics = characteristics.rstrip() + f"""

## 12. Modeling Stage

- Added optional feature `price_relative_series_28` to the feature set because it improved CV MAE.
- Final feature count: {len(feature_cols)}.
- Price feature ablation CV MAE changed from 16.724148 with Price to 27.499129 without Price.
- Final chronological test range: `2023-10-03` to `2024-01-30`.
"""
with open(BASE / "data_characteristics.md", "w", encoding="utf-8") as file:
    file.write(characteristics)

documentation = f"""# Project Documentation

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
"""
for line in model_table:
    name, mae, rmse, wape, r2 = line.split(" | ")
    documentation += f"| {name} | {mae} | {rmse} | {wape} | {r2} |\n"

documentation += f"""
### Experiments and Tuning

- Sqrt target outperformed raw target: 16.412240 versus 16.724148 CV MAE.
- With Price CV MAE: 16.724148; without Price CV MAE: 27.499129.
- Relative series Price feature improved CV MAE from 16.724148 to 4.552064.
- Tuned CV MAE: 4.381249.

### Final Test

- Final model: LightGBM.
- Test MAE: {test_mae:.6f}.
- Test RMSE: {test_rmse:.6f}.
- Test WAPE: {test_wape:.6f}.
- Test R2: {test_r2:.6f}.
- No-Price test MAE: {no_price_mae:.6f}.
- MAE improvement over Moving Average 7: {improvement:.6f}%.

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

- Future backend integration must supply planned Price, Discount, Promotion, Weather and epidemic assumptions.
- Unknown store/product pairs and unsupported dates return clear errors.
- Price is a potentially sensitive or structural feature and should be monitored in no-Price deployment comparisons.
- The next stage is FastAPI and frontend integration followed by end-to-end execution.
"""
with open(BASE / "project_documentation.md", "w", encoding="utf-8") as file:
    file.write(documentation)

print("Saved: model_metadata.json")
print("Saved: Findings_Report.md")
print("Saved: data_characteristics.md")
print("Saved: project_documentation.md")
