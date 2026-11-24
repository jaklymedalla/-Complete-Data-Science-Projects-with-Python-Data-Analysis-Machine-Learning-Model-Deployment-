# Findings Report

## Project Objective

Describe the problem and objective here.

## Data Cleaning

### Cleaning Decisions

| Issue | Decision | Reason | Affected Data | Impact |
|---|---|---|---|---|

### Cleaning Summary

Summarize the overall impact of data cleaning here.

## Key EDA Insights

- Date: Top group means: 2023-07-08=140.930000; 2023-08-10=139.360000; 2023-06-11=137.730000; 2022-07-26=137.720000; 2023-07-28=137.060000; 2023-06-20=136.840000; 2022-08-25=134.940000; 2023-07-03=134.520000; 2023-07-26=134.110000; 2022-06-26=133.490000
- Category: Top group means: Groceries=120.976447; Clothing=112.619737; Electronics=97.482018; Toys=92.606955; Furniture=73.581140
- Region: Top group means: South=106.938882; East=106.468684; North=103.771645; West=100.634934
- Inventory Level: Pearson=0.126618; Spearman=0.117184
- Units Sold: Pearson=0.833421; Spearman=0.847304
- Units Ordered: Pearson=0.511963; Spearman=0.426569
- Price: Pearson=-0.023461; Spearman=-0.024321
- Discount: Pearson=0.224723; Spearman=0.204341
- Weather Condition: Top group means: Sunny=115.167189; Cloudy=105.404064; Rainy=95.106743; Snowy=94.045789
- Promotion: Pearson=0.282537; Spearman=0.261325
- Competitor Pricing: Pearson=-0.023036; Spearman=-0.024071
- Seasonality: Top group means: Summer=112.855380; Autumn=103.422967; Winter=103.406190; Spring=97.703098
- Epidemic: Pearson=-0.363661; Spearman=-0.376951

## Train/Test Split

- Split method:
- Random state:
- Training row count:
- Test row count:
- Stratification used:
- Stratification target:

## Preprocessing and Feature Engineering

Summarize preprocessing and feature-engineering decisions here.

## Baseline Model

Record the baseline model, validation method, and executed metrics here.

## Model Comparison

| Model | Validation Method | Metric(s) | Result | Notes |
|---|---|---|---|---|

## Best Model

- Selected model:
- Selection reason:
- Cross-validation results:
- Test-set results:

## Final Conclusions

The deployed LightGBM artifact is integrated with a FastAPI backend and a responsive single-page dashboard. The application preserves the existing `predict.py` and `preprocessing.add_features()` feature path, loads the deployment artifacts and history during API startup, and uses one batch model call for a historical range backtest.

### Deployment Notes and Final Status

- The API exposes health, metadata, series, defaults, prediction, history, and batched backtest endpoints. Swagger documentation is available at `http://localhost:8000/docs`; the dashboard is served at `http://localhost:5500`.
- The focused API suite completed with 8 passing tests, including exact agreement with `predict_demand()` for three historical predictions.
- HTTP smoke checks returned 200 for health, series, defaults, forecast, backtest, and history. The invalid-discount request returned the expected 422 validation response. Both the frontend and Swagger page loaded in the browser.
- The frontend does not expose a Price input. The deployed model metadata includes Price in the feature list, so the latest observed Price (and its observed competitor-price context) is supplied automatically. Do not describe this artifact as a no-Price model; the no-Price metrics in the metadata refer to a separate evaluation.
- Continue monitoring performance on new data, especially post-epidemic recovery, and retrain/update the model and metadata together when the deployment is refreshed.

## Data Understanding (Observation Only)

- Dataset shape: 76000 rows and 16 columns.
- Target variable: `Demand`.
- Target type: regression.
- Fully duplicated rows: 0.
- Highly skewed numerical columns: `Inventory Level`, `Units Ordered`, `Epidemic`.
- Serious outlier columns: `Inventory Level`, `Units Ordered`, `Discount`.
- Multicollinearity candidates: 2 Pearson pairs with absolute correlation greater than 0.7.
- Data characteristics and all executed values were recorded in `data_characteristics.md`.

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
| LightGBM | 16.724148 +/- 1.884772 | 23.200741 +/- 2.595610 | 0.160077 +/- 0.023107 | 0.754698 +/- 0.052180 |
| XGBoost | 18.529101 +/- 1.532920 | 25.410153 +/- 2.155867 | 0.177015 +/- 0.018651 | 0.706390 +/- 0.051493 |
| HistGradientBoosting | 19.733529 +/- 1.298067 | 26.842348 +/- 1.911151 | 0.188518 +/- 0.017214 | 0.672675 +/- 0.052005 |
| Random Forest | 25.142407 +/- 1.462021 | 33.508579 +/- 2.034672 | 0.239965 +/- 0.017727 | 0.489544 +/- 0.079877 |
| Ridge | 26.417526 +/- 0.840093 | 34.423489 +/- 1.207646 | 0.252011 +/- 0.010831 | 0.462482 +/- 0.064059 |
| Linear Regression | 26.478861 +/- 0.753621 | 34.520934 +/- 1.031372 | 0.252604 +/- 0.010404 | 0.459069 +/- 0.065529 |
| Lasso | 27.763370 +/- 1.009476 | 36.162501 +/- 1.520428 | 0.264867 +/- 0.012827 | 0.407511 +/- 0.067277 |

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

- Moving Average 7 baseline test MAE: 31.639726.
- Final LightGBM test MAE: 3.710015.
- MAE improvement: 88.274186%.
- Test RMSE: 4.828937; WAPE: 0.037698; R2: 0.988430.
- No-Price test MAE: 7.592269; RMSE: 8.971564; WAPE: 0.077146; R2: 0.960063.

### Error Analysis and Feature Importance

- Overall mean residual: -0.120038.
- Epidemic mean residual: 0.189069; non-epidemic mean residual: -0.227705.
- Error plots: [11_actual_vs_predicted_1.png](figures/11_actual_vs_predicted_1.png), [11_actual_vs_predicted_2.png](figures/11_actual_vs_predicted_2.png), [11_actual_vs_predicted_3.png](figures/11_actual_vs_predicted_3.png).
- Residual, scatter and daily plots: [12_residual_distribution.png](figures/12_residual_distribution.png), [13_predicted_vs_actual.png](figures/13_predicted_vs_actual.png), [14_daily_actual_vs_predicted.png](figures/14_daily_actual_vs_predicted.png).
- Combined importance plot: [15_feature_importance.png](figures/15_feature_importance.png).
- Top permutation features: `price_relative_series_28`, `Discount`, `demand_roll_mean_28`.
- The deployment consistency check used 5 random test rows; maximum absolute difference: 0.000000000000.
