# Data Characteristics

## Dataset Overview

- Source: `demand_forecasting.csv`
- Row count: 76000
- Column count: 16
- Target variable: `Demand`
- Target type: regression
- Observation period: Date available

## Column Names

- `Date`
- `Store ID`
- `Product ID`
- `Category`
- `Region`
- `Inventory Level`
- `Units Sold`
- `Units Ordered`
- `Price`
- `Discount`
- `Weather Condition`
- `Promotion`
- `Competitor Pricing`
- `Seasonality`
- `Epidemic`
- `Demand`

## Basic Structure

### Shape

- Rows: 76000
- Columns: 16

### Data Types

| Column             | Data Type   |
|:-------------------|:------------|
| Date               | str         |
| Store ID           | str         |
| Product ID         | str         |
| Category           | str         |
| Region             | str         |
| Inventory Level    | int64       |
| Units Sold         | int64       |
| Units Ordered      | int64       |
| Price              | float64     |
| Discount           | int64       |
| Weather Condition  | str         |
| Promotion          | int64       |
| Competitor Pricing | float64     |
| Seasonality        | str         |
| Epidemic           | int64       |
| Demand             | int64       |

### Column Separation

- Numerical: `Inventory Level`, `Units Sold`, `Units Ordered`, `Price`, `Discount`, `Promotion`, `Competitor Pricing`, `Epidemic`, `Demand`
- Categorical: `Date`, `Store ID`, `Product ID`, `Category`, `Region`, `Weather Condition`, `Seasonality`
- Datetime: `Date`
- Possible ID: `Store ID`, `Product ID`

### Suspicious or Wrong Types

- `Date`: stored as object/string but parseable as datetime

## Statistical Summary

### Numerical Describe

| Column             |   count |       mean |        std |   min |      25% |   50% |      75% |     max |
|:-------------------|--------:|-----------:|-----------:|------:|---------:|------:|---------:|--------:|
| Inventory Level    |   76000 | 301.063    | 226.51     |  0    | 136      | 227   | 408      | 2267    |
| Units Sold         |   76000 |  88.8273   |  43.9945   |  0    |  58      |  84   | 114      |  426    |
| Units Ordered      |   76000 |  89.0906   | 162.405    |  0    |   0      |   0   | 121      | 1616    |
| Price              |   76000 |  67.726    |  39.3779   |  4.74 |  31.9975 |  64.5 |  95.83   |  228.03 |
| Discount           |   76000 |   9.08704  |   7.47578  |  0    |   5      |  10   |  10      |   25    |
| Promotion          |   76000 |   0.328947 |   0.469834 |  0    |   0      |   0   |   1      |    1    |
| Competitor Pricing |   76000 |  69.454    |  40.9438   |  4.29 |  32.62   |  65.7 |  97.9325 |  261.22 |
| Epidemic           |   76000 |   0.2      |   0.400003 |  0    |   0      |   0   |   0      |    1    |
| Demand             |   76000 | 104.317    |  46.9648   |  4    |  71      | 100   | 133      |  430    |

### Categorical Describe

| Column            |   count |   unique | top        |   freq |
|:------------------|--------:|---------:|:-----------|-------:|
| Date              |   76000 |      760 | 2022-01-01 |    100 |
| Store ID          |   76000 |        5 | S001       |  15200 |
| Product ID        |   76000 |       20 | P0001      |   3800 |
| Category          |   76000 |        5 | Groceries  |  30400 |
| Region            |   76000 |        4 | North      |  30400 |
| Weather Condition |   76000 |        4 | Cloudy     |  24360 |
| Seasonality       |   76000 |        4 | Winter     |  21000 |

### Skewness and Kurtosis

| Column             |   Skewness |   Kurtosis | Highly Skewed (|skew| > 1)   |
|:-------------------|-----------:|-----------:|:-----------------------------|
| Inventory Level    |   1.59194  |   3.37171  | True                         |
| Units Sold         |   0.790429 |   1.24907  | False                        |
| Units Ordered      |   2.51588  |   7.3058   | True                         |
| Price              |   0.485025 |  -0.483944 | False                        |
| Discount           |   0.643149 |  -0.405212 | False                        |
| Promotion          |   0.72816  |  -1.46982  | False                        |
| Competitor Pricing |   0.549003 |  -0.309692 | False                        |
| Epidemic           |   1.50003  |   0.250095 | True                         |
| Demand             |   0.607778 |   0.674937 | False                        |

Highly skewed columns (|skew| > 1): `Inventory Level`, `Units Ordered`, `Epidemic`

### Unique Values

| Column             |   Unique Count |   Unique Percentage | Constant   | Near-Constant   | High-Cardinality Categorical   |
|:-------------------|---------------:|--------------------:|:-----------|:----------------|:-------------------------------|
| Date               |            760 |            1        | False      | False           | False                          |
| Store ID           |              5 |            0.006579 | False      | False           | False                          |
| Product ID         |             20 |            0.026316 | False      | False           | False                          |
| Category           |              5 |            0.006579 | False      | False           | False                          |
| Region             |              4 |            0.005263 | False      | False           | False                          |
| Inventory Level    |           1426 |            1.87632  | False      | False           | False                          |
| Units Sold         |            330 |            0.434211 | False      | False           | False                          |
| Units Ordered      |            996 |            1.31053  | False      | False           | False                          |
| Price              |          15396 |           20.2579   | False      | False           | False                          |
| Discount           |              6 |            0.007895 | False      | False           | False                          |
| Weather Condition  |              4 |            0.005263 | False      | False           | False                          |
| Promotion          |              2 |            0.002632 | False      | False           | False                          |
| Competitor Pricing |          15963 |           21.0039   | False      | False           | False                          |
| Seasonality        |              4 |            0.005263 | False      | False           | False                          |
| Epidemic           |              2 |            0.002632 | False      | False           | False                          |
| Demand             |            340 |            0.447368 | False      | False           | False                          |

## Data Quality

### Missing Values

| Column             |   Missing Count |   Missing Percentage |
|:-------------------|----------------:|---------------------:|
| Category           |               0 |                    0 |
| Competitor Pricing |               0 |                    0 |
| Date               |               0 |                    0 |
| Demand             |               0 |                    0 |
| Discount           |               0 |                    0 |
| Epidemic           |               0 |                    0 |
| Inventory Level    |               0 |                    0 |
| Price              |               0 |                    0 |
| Product ID         |               0 |                    0 |
| Promotion          |               0 |                    0 |
| Region             |               0 |                    0 |
| Seasonality        |               0 |                    0 |
| Store ID           |               0 |                    0 |
| Units Ordered      |               0 |                    0 |
| Units Sold         |               0 |                    0 |
| Weather Condition  |               0 |                    0 |

### Missingness Together

- Missingness pairs with absolute correlation >= 0.5: None

### Duplicate Rows

- Fully duplicated row count: 0

### Negative Numeric Values

- None

### Impossible Ranges

- None

### Placeholder Values

- `Inventory Level`: 0 (406), 999 (7)
- `Units Sold`: 0 (406)
- `Units Ordered`: 0 (47107)
- `Discount`: 0 (17126)
- `Promotion`: 0 (51000)
- `Epidemic`: 0 (60800)

## Outlier Analysis

| Column             |   IQR Outlier Count |   IQR Outlier Percentage |   Z-score Outlier Count |   Z-score Outlier Percentage | Serious Outlier Problem   |
|:-------------------|--------------------:|-------------------------:|------------------------:|-----------------------------:|:--------------------------|
| Inventory Level    |                2759 |                 3.63026  |                    1141 |                     1.50132  | True                      |
| Units Sold         |                1411 |                 1.85658  |                     686 |                     0.902632 | False                     |
| Units Ordered      |                7524 |                 9.9      |                    2022 |                     2.66053  | True                      |
| Price              |                  70 |                 0.092105 |                     147 |                     0.193421 | False                     |
| Discount           |               12413 |                16.3329   |                       0 |                     0        | True                      |
| Promotion          |                   0 |                 0        |                       0 |                     0        | False                     |
| Competitor Pricing |                 185 |                 0.243421 |                     244 |                     0.321053 | False                     |
| Epidemic           |                   0 |                 0        |                       0 |                     0        | False                     |
| Demand             |                 986 |                 1.29737  |                     517 |                     0.680263 | False                     |

Serious outlier columns: `Inventory Level`, `Units Ordered`, `Discount`

## Target Variable Analysis

### Regression Distribution

|   count |    mean |     std |   min |   25% |   50% |   75% |   max |
|--------:|--------:|--------:|------:|------:|------:|------:|------:|
|   76000 | 104.317 | 46.9648 |     4 |    71 |   100 |   133 |   430 |

- Skewness: 0.607778
- Log transform may be needed: False

### Relationship with Features

| Feature            | Type        | Target Relationship                                                                                                                                                                                                                                   |
|:-------------------|:------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Date               | Categorical | Top group means: 2023-07-08=140.930000; 2023-08-10=139.360000; 2023-06-11=137.730000; 2022-07-26=137.720000; 2023-07-28=137.060000; 2023-06-20=136.840000; 2022-08-25=134.940000; 2023-07-03=134.520000; 2023-07-26=134.110000; 2022-06-26=133.490000 |
| Category           | Categorical | Top group means: Groceries=120.976447; Clothing=112.619737; Electronics=97.482018; Toys=92.606955; Furniture=73.581140                                                                                                                                |
| Region             | Categorical | Top group means: South=106.938882; East=106.468684; North=103.771645; West=100.634934                                                                                                                                                                 |
| Inventory Level    | Numerical   | Pearson=0.126618; Spearman=0.117184                                                                                                                                                                                                                   |
| Units Sold         | Numerical   | Pearson=0.833421; Spearman=0.847304                                                                                                                                                                                                                   |
| Units Ordered      | Numerical   | Pearson=0.511963; Spearman=0.426569                                                                                                                                                                                                                   |
| Price              | Numerical   | Pearson=-0.023461; Spearman=-0.024321                                                                                                                                                                                                                 |
| Discount           | Numerical   | Pearson=0.224723; Spearman=0.204341                                                                                                                                                                                                                   |
| Weather Condition  | Categorical | Top group means: Sunny=115.167189; Cloudy=105.404064; Rainy=95.106743; Snowy=94.045789                                                                                                                                                                |
| Promotion          | Numerical   | Pearson=0.282537; Spearman=0.261325                                                                                                                                                                                                                   |
| Competitor Pricing | Numerical   | Pearson=-0.023036; Spearman=-0.024071                                                                                                                                                                                                                 |
| Seasonality        | Categorical | Top group means: Summer=112.855380; Autumn=103.422967; Winter=103.406190; Spring=97.703098                                                                                                                                                            |
| Epidemic           | Numerical   | Pearson=-0.363661; Spearman=-0.376951                                                                                                                                                                                                                 |

## Relationship Analysis

### Pearson Correlation Matrix

| Column             |   Inventory Level |   Units Sold |   Units Ordered |     Price |   Discount |   Promotion |   Competitor Pricing |   Epidemic |    Demand |
|:-------------------|------------------:|-------------:|----------------:|----------:|-----------:|------------:|---------------------:|-----------:|----------:|
| Inventory Level    |          1        |     0.23459  |       -0.034698 | -0.036673 |  -0.000678 |   -0.006082 |            -0.034754 |  -0.09755  |  0.126618 |
| Units Sold         |          0.23459  |     1        |        0.433943 | -0.014506 |   0.183523 |    0.227062 |            -0.013989 |  -0.317855 |  0.833421 |
| Units Ordered      |         -0.034698 |     0.433943 |        1        |  0.04498  |   0.172972 |    0.214743 |             0.045235 |  -0.100546 |  0.511963 |
| Price              |         -0.036673 |    -0.014506 |        0.04498  |  1        |  -0.094136 |   -0.04319  |             0.976648 |  -0.089658 | -0.023461 |
| Discount           |         -0.000678 |     0.183523 |        0.172972 | -0.094136 |   1        |    0.784039 |            -0.092066 |  -0.003885 |  0.224723 |
| Promotion          |         -0.006082 |     0.227062 |        0.214743 | -0.04319  |   0.784039 |    1        |            -0.041385 |   0.0007   |  0.282537 |
| Competitor Pricing |         -0.034754 |    -0.013989 |        0.045235 |  0.976648 |  -0.092066 |   -0.041385 |             1        |  -0.089466 | -0.023036 |
| Epidemic           |         -0.09755  |    -0.317855 |       -0.100546 | -0.089658 |  -0.003885 |    0.0007   |            -0.089466 |   1        | -0.363661 |
| Demand             |          0.126618 |     0.833421 |        0.511963 | -0.023461 |   0.224723 |    0.282537 |            -0.023036 |  -0.363661 |  1        |

### Spearman Correlation Matrix

| Column             |   Inventory Level |   Units Sold |   Units Ordered |     Price |   Discount |   Promotion |   Competitor Pricing |   Epidemic |    Demand |
|:-------------------|------------------:|-------------:|----------------:|----------:|-----------:|------------:|---------------------:|-----------:|----------:|
| Inventory Level    |          1        |     0.252028 |       -0.27797  |  0.00798  |  -0.000829 |   -0.005777 |             0.010174 |  -0.106334 |  0.117184 |
| Units Sold         |          0.252028 |     1        |        0.337778 | -0.013064 |   0.166772 |    0.211554 |            -0.012589 |  -0.341171 |  0.847304 |
| Units Ordered      |         -0.27797  |     0.337778 |        1        |  0.024945 |   0.146276 |    0.189399 |             0.024087 |  -0.075258 |  0.426569 |
| Price              |          0.00798  |    -0.013064 |        0.024945 |  1        |  -0.083128 |   -0.039252 |             0.983696 |  -0.081835 | -0.024321 |
| Discount           |         -0.000829 |     0.166772 |        0.146276 | -0.083128 |   1        |    0.765674 |            -0.080804 |  -0.003444 |  0.204341 |
| Promotion          |         -0.005777 |     0.211554 |        0.189399 | -0.039252 |   0.765674 |    1        |            -0.037759 |   0.0007   |  0.261325 |
| Competitor Pricing |          0.010174 |    -0.012589 |        0.024087 |  0.983696 |  -0.080804 |   -0.037759 |             1        |  -0.080965 | -0.024071 |
| Epidemic           |         -0.106334 |    -0.341171 |       -0.075258 | -0.081835 |  -0.003444 |    0.0007   |            -0.080965 |   1        | -0.376951 |
| Demand             |          0.117184 |     0.847304 |        0.426569 | -0.024321 |   0.204341 |    0.261325 |            -0.024071 |  -0.376951 |  1        |

### Pearson-Spearman Large Differences (>0.2)

| Feature 1       | Feature 2     |   Pearson |   Spearman |   Absolute Difference |
|:----------------|:--------------|----------:|-----------:|----------------------:|
| Inventory Level | Units Ordered | -0.034698 |   -0.27797 |              0.243272 |

### Multicollinearity Candidates (|Pearson| > 0.7)

| Feature 1   | Feature 2          |   Pearson Correlation |
|:------------|:-------------------|----------------------:|
| Price       | Competitor Pricing |              0.976648 |
| Discount    | Promotion          |              0.784039 |

### VIF

| Feature            |      VIF |
|:-------------------|---------:|
| Inventory Level    |  1.08783 |
| Units Sold         |  1.50602 |
| Units Ordered      |  1.28886 |
| Price              | 21.6813  |
| Discount           |  2.62138 |
| Promotion          |  2.67779 |
| Competitor Pricing | 21.6692  |
| Epidemic           |  1.13242 |

### Cramer's V (Categorical Pairs > 0.3)

| Feature 1   | Feature 2         |   Cramer's V |
|:------------|:------------------|-------------:|
| Date        | Weather Condition |     0.571862 |
| Date        | Seasonality       |     0.995014 |

### Mutual Information with Target

| Feature            |   Mutual Information |
|:-------------------|---------------------:|
| Units Sold         |             0.974012 |
| Units Ordered      |             0.38853  |
| Price              |             0.340756 |
| Date               |             0.100673 |
| Category           |             0.09947  |
| Epidemic           |             0.089013 |
| Promotion          |             0.039727 |
| Discount           |             0.030519 |
| Competitor Pricing |             0.02274  |
| Product ID         |             0.01964  |
| Weather Condition  |             0.015666 |
| Seasonality        |             0.008324 |
| Inventory Level    |             0.008291 |
| Store ID           |             0.004356 |
| Region             |             0.004055 |

## Key Observations & Risks

- No missing values detected
- Highly skewed numerical columns: `Inventory Level`, `Units Ordered`, `Epidemic`
- Serious IQR/Z-score outlier columns: `Inventory Level`, `Units Ordered`, `Discount`
- High-cardinality categorical columns: None
- Near-constant columns: None
- Multicollinearity candidates: `Price` / `Competitor Pricing`, `Discount` / `Promotion`
- Target imbalance risk: Not detected
- Leakage suspects: Candidate IDs and columns unavailable at prediction time require business-time validation; no conclusion is made at observation stage.
- Data cleaning items: negative values, impossible ranges, placeholders, duplicate rows, missing values, and outliers should be handled only in the later cleaning stage.

## 10. Change Log

| Change | Rows Before | Rows After | Columns Before | Columns After | Reason |
|---|---:|---:|---:|---:|---|
| Date conversion and sorting | 76000 | 76000 | 16 | 17 | Prepared observations for chronological per-series lag calculations. |
| Added `series_id` | 76000 | 76000 | 16 | 17 | Created the Store/Product series key. |
| Engineered lag, rolling and calendar features | 76000 | 76000 | 17 | 40 | Created backward-looking forecasting features. |
| Removed first 28 days of every series | 76000 | 73200 | 40 | 35 | Removed rows where the 28-day lag or 28-day rolling feature was unavailable; 2800 rows removed. |

## 11. Train/Test Split Summary

- Split date: `2023-10-03`
- Training date range: `2022-01-29` to `2023-10-02`
- Test date range: `2023-10-03` to `2024-01-30`
- Training rows: 61200
- Test rows: 12000
- Training Demand mean: 105.167239
- Test Demand mean: 98.414167
- Test epidemic states: 2
- Test contains both epidemic states: True
- Split adjustment: No adjustment: the final 120-day window contains both epidemic states.

### Final Feature Set

**Numerical features (27):**
`year`, `month`, `day_of_month`, `day_of_week`, `week_of_year`, `is_weekend`, `month_sin`, `month_cos`, `demand_lag_1`, `demand_lag_7`, `demand_lag_14`, `demand_lag_28`, `demand_roll_mean_7`, `demand_roll_mean_14`, `demand_roll_mean_28`, `demand_roll_std_7`, `Units Sold lag 1`, `Units Sold lag 7`, `Units Ordered lag 1`, `Inventory Level lag 1`, `Price`, `Discount`, `Promotion`, `price_gap`, `price_ratio`, `Epidemic`, `epidemic_days_past_7`

**Categorical features (4):**
`Store ID`, `Product ID`, `Category`, `Weather Condition`

**Final feature count: 31**

## 12. Modeling Stage

- Added optional feature `price_relative_series_28` to the feature set because it improved CV MAE.
- Final feature count: 32.
- Price feature ablation CV MAE changed from 16.724148 with Price to 27.499129 without Price.
- Final chronological test range: `2023-10-03` to `2024-01-30`.
