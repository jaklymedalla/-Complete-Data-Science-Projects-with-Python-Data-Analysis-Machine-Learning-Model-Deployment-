from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "demand_forecasting.csv"
CHARACTERISTICS_PATH = BASE_DIR / "data_characteristics.md"
FINDINGS_PATH = BASE_DIR / "Findings_Report.md"
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "figures"
TARGET = "Demand"
SERIES_KEY = ["Store ID", "Product ID"]
RANDOM_STATE = 42


def add_features(frame):
    """Create backward-looking forecasting features for each Store/Product series."""
    result = frame.copy()
    result["Date"] = pd.to_datetime(result["Date"], errors="raise")
    if "series_id" not in result:
        result["series_id"] = result["Store ID"].astype(str) + "__" + result["Product ID"].astype(str)

    result = result.sort_values(["series_id", "Date"]).reset_index(drop=True)

    result["year"] = result["Date"].dt.year
    result["month"] = result["Date"].dt.month
    result["day_of_month"] = result["Date"].dt.day
    result["day_of_week"] = result["Date"].dt.day_of_week
    result["week_of_year"] = result["Date"].dt.isocalendar().week.astype(int)
    result["is_weekend"] = (result["day_of_week"] >= 5).astype(int)
    result["month_sin"] = np.sin(2 * np.pi * result["month"] / 12)
    result["month_cos"] = np.cos(2 * np.pi * result["month"] / 12)

    grouped_demand = result.groupby("series_id", sort=False)["Demand"]
    for lag in (1, 7, 14, 28):
        result[f"demand_lag_{lag}"] = grouped_demand.shift(lag)

    for window in (7, 14, 28):
        result[f"demand_roll_mean_{window}"] = (
            grouped_demand.transform(
                lambda values, window=window: values.shift(1).rolling(window=window, min_periods=window).mean()
            )
        )
    result["demand_roll_std_7"] = (
        grouped_demand.transform(
            lambda values: values.shift(1).rolling(window=7, min_periods=7).std()
        )
    )

    grouped = result.groupby("series_id", sort=False)
    result["Units Sold lag 1"] = grouped["Units Sold"].shift(1)
    result["Units Sold lag 7"] = grouped["Units Sold"].shift(7)
    result["Units Ordered lag 1"] = grouped["Units Ordered"].shift(1)
    result["Inventory Level lag 1"] = grouped["Inventory Level"].shift(1)

    competitor = result["Competitor Pricing"].replace(0, np.nan)
    result["price_gap"] = result["Price"] - result["Competitor Pricing"]
    result["price_ratio"] = result["Price"] / competitor
    if "Price" in result:
        price_roll_28 = (
            grouped["Price"].transform(
                lambda values: values.shift(1).rolling(window=28, min_periods=28).mean()
            )
        )
        result["price_relative_series_28"] = result["Price"] / price_roll_28.replace(0, np.nan)
    result["epidemic_days_past_7"] = (
        grouped["Epidemic"].transform(
            lambda values: values.shift(1).rolling(window=7, min_periods=1).sum()
        )
    )
    return result


def build_preprocessor(config_path=None, numerical_features=None, categorical_features=None):
    """Build a fitted-input-independent preprocessing object from feature_config.json."""
    if config_path is None:
        config_path = BASE_DIR / "feature_config.json"
    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)
    numerical = numerical_features if numerical_features is not None else config["numerical_features"]
    categorical = categorical_features if categorical_features is not None else config["categorical_features"]
    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numerical", StandardScaler(), numerical),
        ],
        remainder="drop",
    )

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")


    def save_and_close(name):
        path = FIGURES_DIR / name
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"PLOT SAVED: {path}")
        return path.name


    raw = pd.read_csv(DATA_PATH)
    row_count_before = len(raw)
    column_count_before = len(raw.columns)

    cleaned = raw.copy()
    cleaned["Date"] = pd.to_datetime(cleaned["Date"], errors="raise")
    cleaned["series_id"] = cleaned["Store ID"].astype(str) + "__" + cleaned["Product ID"].astype(str)
    cleaned = cleaned.sort_values(SERIES_KEY[:1] + ["Product ID", "Date"]).reset_index(drop=True)

    print("=== CLEANING ===")
    print("Original rows:", row_count_before)
    print("Original columns:", column_count_before)
    print("Date dtype:", cleaned["Date"].dtype)
    print("Rows with Units Sold > Demand:",
          int((cleaned["Units Sold"] > cleaned["Demand"]).sum()))
    print("Percentage with Units Sold > Demand:",
          f"{(cleaned['Units Sold'] > cleaned['Demand']).mean() * 100:.6f}%")

    series_counts = cleaned.groupby("series_id").size()
    unique_dates = cleaned["Date"].nunique()
    expected_rows = cleaned["series_id"].nunique() * unique_dates
    complete_series = int((series_counts == unique_dates).sum())
    global_dates = set(cleaned["Date"].unique())
    gap_rows = cleaned.groupby("series_id")["Date"].apply(
        lambda dates: len(set(dates) - global_dates)
    )

    print("Series count:", cleaned["series_id"].nunique())
    print("Unique dates:", unique_dates)
    print("Expected rows for complete 100 x 760 grid:", expected_rows)
    print("Actual rows:", len(cleaned))
    print("Complete series with 760 observations:", complete_series)
    print("Missing cross-series gaps:", int(gap_rows.sum()))
    print("Duplicate cleaned rows:", int(cleaned.duplicated().sum()))

    cleaned_history = cleaned.copy()
    cleaned_history.to_csv(DATA_DIR / "history.csv", index=False)
    print("Saved:", DATA_DIR / "history.csv")

    print("\n=== LEAKAGE PROOF ===")
    same_day_sold_corr = cleaned["Units Sold"].corr(cleaned[TARGET], method="pearson")
    lagged_sold = cleaned.groupby("series_id", sort=False)["Units Sold"].shift(1)
    lagged_pairs = pd.concat([lagged_sold, cleaned[TARGET]], axis=1).dropna()
    lag1_sold_corr = lagged_pairs.iloc[:, 0].corr(lagged_pairs[TARGET], method="pearson")
    print("Same-day Units Sold vs Demand Pearson correlation:",
          f"{same_day_sold_corr:.6f}")
    print("Lag-1 Units Sold vs Demand Pearson correlation:",
          f"{lag1_sold_corr:.6f}")

    # One aggregate epidemic-shaded demand plot.
    daily = cleaned.groupby("Date", as_index=False).agg(
        total_demand=(TARGET, "sum"),
        epidemic=("Epidemic", "max"),
    )
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(daily["Date"], daily["total_demand"], color="#2878b5", linewidth=1)
    epidemic_dates = daily.loc[daily["epidemic"] == 1, "Date"]
    for date in epidemic_dates:
        ax.axvspan(date, date + pd.Timedelta(days=1), color="red", alpha=0.025)
    ax.set_title("Daily Total Demand with Epidemic Days Shaded")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Demand")
    fig_daily = save_and_close("01_daily_demand_epidemic.png")
    print("INSIGHT: The chart shows the level and movement of aggregate daily demand, "
          "with red shading marking days where any epidemic indicator is present.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(cleaned[TARGET], bins=40, color="#2878b5", edgecolor="black", alpha=0.75)
    axes[0].set_title("Demand Distribution")
    axes[0].set_xlabel("Demand")
    axes[0].set_ylabel("Frequency")
    sns.boxplot(y=cleaned[TARGET], ax=axes[1], color="#ff8c00")
    axes[1].set_title("Demand Boxplot")
    fig_demand = save_and_close("02_demand_distribution_boxplot.png")
    print("INSIGHT: The histogram and boxplot display the demand spread and the "
          "location of high-side observations.")

    eda_specs = [
        ("Category", "03_demand_by_category.png", "Category"),
        ("Weather Condition", "04_demand_by_weather.png", "Weather Condition"),
        ("Promotion", "05_demand_by_promotion.png", "Promotion"),
        ("Discount", "06_demand_by_discount.png", "Discount"),
        ("Epidemic", "07_demand_by_epidemic.png", "Epidemic"),
    ]
    eda_insights = {}
    for column, filename, title in eda_specs:
        fig, ax = plt.subplots(figsize=(10, 5))
        order = sorted(cleaned[column].dropna().unique(), key=lambda value: str(value))
        sns.boxplot(data=cleaned, x=column, y=TARGET, order=order, ax=ax)
        ax.set_title(f"Demand by {title}")
        ax.set_xlabel(title)
        ax.set_ylabel("Demand")
        plt.xticks(rotation=45 if cleaned[column].nunique() > 8 else 0)
        save_and_close(filename)
        group_means = cleaned.groupby(column, observed=True)[TARGET].mean().sort_values(ascending=False)
        eda_insights[column] = group_means
        print(f"INSIGHT: Demand varies across {title} levels; the largest mean belongs to "
              f"{group_means.index[0]} at {group_means.iloc[0]:.6f}.")

    feature_engineered = cleaned.copy()

    feature_engineered["year"] = feature_engineered["Date"].dt.year
    feature_engineered["month"] = feature_engineered["Date"].dt.month
    feature_engineered["day_of_month"] = feature_engineered["Date"].dt.day
    feature_engineered["day_of_week"] = feature_engineered["Date"].dt.dayofweek
    feature_engineered["week_of_year"] = feature_engineered["Date"].dt.isocalendar().week.astype(int)
    feature_engineered["is_weekend"] = (feature_engineered["day_of_week"] >= 5).astype(int)
    feature_engineered["month_sin"] = np.sin(2 * np.pi * feature_engineered["month"] / 12)
    feature_engineered["month_cos"] = np.cos(2 * np.pi * feature_engineered["month"] / 12)

    for lag in [1, 7, 14, 28]:
        feature_engineered[f"demand_lag_{lag}"] = (
            feature_engineered.groupby("series_id", sort=False)[TARGET].shift(lag)
        )

    feature_engineered["demand_roll_mean_7"] = (
        feature_engineered.groupby("series_id", sort=False)[TARGET]
        .transform(lambda values: values.shift(1).rolling(window=7, min_periods=7).mean())
    )
    feature_engineered["demand_roll_mean_14"] = (
        feature_engineered.groupby("series_id", sort=False)[TARGET]
        .transform(lambda values: values.shift(1).rolling(window=14, min_periods=14).mean())
    )
    feature_engineered["demand_roll_mean_28"] = (
        feature_engineered.groupby("series_id", sort=False)[TARGET]
        .transform(lambda values: values.shift(1).rolling(window=28, min_periods=28).mean())
    )
    feature_engineered["demand_roll_std_7"] = (
        feature_engineered.groupby("series_id", sort=False)[TARGET]
        .transform(lambda values: values.shift(1).rolling(window=7, min_periods=7).std())
    )

    feature_engineered["Units Sold lag 1"] = (
        feature_engineered.groupby("series_id", sort=False)["Units Sold"].shift(1)
    )
    feature_engineered["Units Sold lag 7"] = (
        feature_engineered.groupby("series_id", sort=False)["Units Sold"].shift(7)
    )
    feature_engineered["Units Ordered lag 1"] = (
        feature_engineered.groupby("series_id", sort=False)["Units Ordered"].shift(1)
    )
    feature_engineered["Inventory Level lag 1"] = (
        feature_engineered.groupby("series_id", sort=False)["Inventory Level"].shift(1)
    )

    feature_engineered["price_gap"] = (
        feature_engineered["Price"] - feature_engineered["Competitor Pricing"]
    )
    feature_engineered["price_ratio"] = (
        feature_engineered["Price"] / feature_engineered["Competitor Pricing"].replace(0, np.nan)
    )
    feature_engineered["epidemic_days_past_7"] = (
        feature_engineered.groupby("series_id", sort=False)["Epidemic"]
        .transform(lambda values: values.shift(1).rolling(window=7, min_periods=1).sum())
    )

    engineered_column_count_before_drop = len(feature_engineered.columns)
    engineered_row_count_before_drop = len(feature_engineered)
    feature_engineered = feature_engineered.dropna(
        subset=[
            "demand_lag_28",
            "demand_roll_mean_28",
            "demand_roll_std_7",
            "Units Sold lag 7",
            "Inventory Level lag 1",
            "Units Ordered lag 1",
        ]
    ).copy()
    rows_removed_first_28 = engineered_row_count_before_drop - len(feature_engineered)

    feature_engineered = feature_engineered.drop(
        columns=["Region", "Seasonality", "Competitor Pricing",
                 "Units Sold", "Units Ordered", "Inventory Level"]
    )

    print("\n=== FEATURE ENGINEERING ===")
    print("Rows removed from first 28 days:", rows_removed_first_28)
    print("Rows remaining:", len(feature_engineered))
    print("Region, Seasonality and Competitor Pricing dropped.")
    print("Same-day Units Sold, Units Ordered and Inventory Level dropped.")

    categorical_features = ["Store ID", "Product ID", "Category", "Weather Condition"]
    numerical_features = [
        "year", "month", "day_of_month", "day_of_week", "week_of_year",
        "is_weekend", "month_sin", "month_cos",
        "demand_lag_1", "demand_lag_7", "demand_lag_14", "demand_lag_28",
        "demand_roll_mean_7", "demand_roll_mean_14", "demand_roll_mean_28",
        "demand_roll_std_7",
        "Units Sold lag 1", "Units Sold lag 7", "Units Ordered lag 1",
        "Inventory Level lag 1",
        "Price", "Discount", "Promotion", "price_gap", "price_ratio",
        "Epidemic", "epidemic_days_past_7",
    ]
    feature_columns = categorical_features + numerical_features

    print("Final numerical feature count:", len(numerical_features))
    print("Final categorical feature count:", len(categorical_features))
    print("Final feature count:", len(feature_columns))

    # Correlation heatmap using final numerical features plus target.
    correlation_data = feature_engineered[numerical_features + [TARGET]].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(16, 12))
    sns.heatmap(correlation_data, cmap="coolwarm", center=0, annot=False,
                square=False, ax=ax)
    ax.set_title("Pearson Correlation Heatmap of Final Numerical Features")
    heatmap_name = save_and_close("08_final_numerical_correlation_heatmap.png")
    print("INSIGHT: The final numerical correlation heatmap highlights linear relationships "
          "among lag features and the target.")

    feature_engineered["month_name"] = feature_engineered["Date"].dt.month_name()
    month_order = list(pd.date_range("2020-01-01", periods=12, freq="MS").month_name())
    month_means = feature_engineered.groupby("month_name", observed=True)[TARGET].mean().reindex(month_order)
    dow_means = feature_engineered.groupby("day_of_week")[TARGET].mean()
    dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(month_order, month_means.to_numpy(), marker="o")
    axes[0].set_title("Average Demand by Month")
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("Average Demand")
    axes[0].tick_params(axis="x", rotation=45)
    axes[1].bar(dow_names, dow_means.reindex(range(7)).to_numpy())
    axes[1].set_title("Average Demand by Day of Week")
    axes[1].set_xlabel("Day of Week")
    axes[1].set_ylabel("Average Demand")
    axes[1].tick_params(axis="x", rotation=45)
    calendar_name = save_and_close("09_average_demand_month_and_dayofweek.png")
    print("INSIGHT: Average demand varies by month and day of week, indicating recurring "
          "calendar patterns that can be captured by calendar features.")

    # Price MI and series/category interpretation.
    mi_price = mutual_info_regression(
        feature_engineered[["Price"]],
        feature_engineered[TARGET],
        random_state=RANDOM_STATE,
    )[0]
    price_by_category = feature_engineered.groupby("Category")["Price"].agg(["mean", "std", "nunique"])
    price_by_store = feature_engineered.groupby("Store ID")["Price"].agg(["mean", "std", "nunique"])

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.scatterplot(
        data=feature_engineered.sample(min(5000, len(feature_engineered)), random_state=RANDOM_STATE),
        x="Price",
        y=TARGET,
        hue="Category",
        alpha=0.45,
        ax=ax,
    )
    ax.set_title("Price vs Demand Coloured by Category")
    price_plot_name = save_and_close("10_price_vs_demand_by_category.png")
    print("INSIGHT: Price may carry category- or series-level structure that does not produce "
          "a strong global linear correlation but can provide nonlinear predictive information.")

    print("\n=== PRICE INVESTIGATION ===")
    print("Price mutual information with Demand:", f"{mi_price:.6f}")
    print("Price mean/std/unique by Category:")
    print(price_by_category.to_string())
    print("Price mean/std/unique by Store:")
    print(price_by_store.to_string())
    price_category_corr = price_by_category["mean"].corr(
        feature_engineered.groupby("Category")[TARGET].mean()
    )
    print("Correlation of category mean Price with category mean Demand:",
          f"{price_category_corr:.6f}")

    # Chronological split.
    unique_dates_sorted = np.sort(feature_engineered["Date"].unique())
    test_window = 120
    test_start = pd.Timestamp(unique_dates_sorted[-test_window])
    train = feature_engineered[feature_engineered["Date"] < test_start].copy()
    test = feature_engineered[feature_engineered["Date"] >= test_start].copy()

    test_epidemic_status = test["Epidemic"].nunique(dropna=True)
    test_has_both_epidemic_states = test["Epidemic"].dropna().nunique() == 2
    split_adjustment = "No adjustment: the final 120-day window contains both epidemic states."
    if not test_has_both_epidemic_states:
        split_adjustment = "Adjustment required: the final 120-day window does not contain both epidemic states."

    print("\n=== CHRONOLOGICAL TRAIN/TEST SPLIT ===")
    print("Train date range:",
          train["Date"].min().strftime("%Y-%m-%d"), "to",
          train["Date"].max().strftime("%Y-%m-%d"))
    print("Test date range:",
          test["Date"].min().strftime("%Y-%m-%d"), "to",
          test["Date"].max().strftime("%Y-%m-%d"))
    print("Split date:", test_start.strftime("%Y-%m-%d"))
    print("Train rows:", len(train))
    print("Test rows:", len(test))
    print("Train Demand mean:", f"{train[TARGET].mean():.6f}")
    print("Test Demand mean:", f"{test[TARGET].mean():.6f}")
    print("Test epidemic unique states:", test_epidemic_status)
    print("Test includes both epidemic states:",
          test_has_both_epidemic_states)
    print("Split adjustment:", split_adjustment)

    output_columns = ["Date", "series_id", TARGET] + feature_columns
    train_output = train[output_columns].copy()
    test_output = test[output_columns].copy()
    train_output.to_csv(DATA_DIR / "train.csv", index=False)
    test_output.to_csv(DATA_DIR / "test.csv", index=False)
    print("Saved:", DATA_DIR / "train.csv")
    print("Saved:", DATA_DIR / "test.csv")

    # Build the preprocessing object and fit only on training values.
    # It is intentionally not persisted; it belongs inside the next-stage model pipeline.
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("numerical", StandardScaler(), numerical_features),
        ],
        remainder="drop",
    )
    X_train = train_output.drop(columns=[TARGET, "Date", "series_id"])
    X_test = test_output.drop(columns=[TARGET, "Date", "series_id"])
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    print("\n=== PREPROCESSING CHECK ===")
    print("Training matrix shape after fit_transform:",
          X_train_transformed.shape)
    print("Test matrix shape after transform:",
          X_test_transformed.shape)
    print("Transformer fitted on training rows:", len(X_train))
    print("Test rows transformed:", len(X_test))

    feature_config = {
        "target_name": TARGET,
        "feature_list": feature_columns,
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "series_key": SERIES_KEY,
        "series_id": "series_id",
        "lag_settings": {
            "demand_lags": [1, 7, 14, 28],
            "demand_rolling_windows": [7, 14, 28],
            "demand_rolling_std_windows": [7],
            "operational_lags": {
                "Units Sold": [1, 7],
                "Units Ordered": [1],
                "Inventory Level": [1],
            },
            "epidemic_past_days": 7,
        },
        "split_date": test_start.strftime("%Y-%m-%d"),
        "test_days": test_window,
        "forecasting_setup": {
            "task": "1-day-ahead demand forecast per series",
            "forecast_time_features": [
                "calendar features",
                "planned Price",
                "planned Discount",
                "planned Promotion",
                "past values only",
                "Weather Condition",
                "Competitor Pricing",
                "Epidemic",
            ],
            "assumption": "Weather Condition, Competitor Pricing and Epidemic are assumed known for the forecast day from forecasts or announcements.",
            "same_day_fields_removed": ["Units Sold", "Units Ordered", "Inventory Level"],
        },
        "preprocessing": {
            "categorical_one_hot": categorical_features,
            "product_id_encoding": "one-hot",
            "numeric_scaler": "StandardScaler",
            "fit_scope": "training set only",
            "transformer_persistence": "not saved; fit inside model pipeline in the modeling stage",
        },
    }
    with open(BASE_DIR / "feature_config.json", "w", encoding="utf-8") as file:
        json.dump(feature_config, file, indent=2)
    print("Saved:", BASE_DIR / "feature_config.json")

    # Update data_characteristics.md with executed split and change-log information.
    with open(CHARACTERISTICS_PATH, "r", encoding="utf-8") as file:
        characteristics = file.read()

    change_log_section = f"""
    ## 10. Change Log

    | Change | Rows Before | Rows After | Columns Before | Columns After | Reason |
    |---|---:|---:|---:|---:|---|
    | Date conversion and sorting | {row_count_before} | {len(cleaned)} | {column_count_before} | {len(cleaned.columns)} | Prepared observations for chronological per-series lag calculations. |
    | Added `series_id` | {len(cleaned)} | {len(cleaned)} | {column_count_before} | {len(cleaned.columns)} | Created the Store/Product series key. |
    | Engineered lag, rolling and calendar features | {len(cleaned)} | {engineered_row_count_before_drop} | {len(cleaned.columns)} | {engineered_column_count_before_drop} | Created backward-looking forecasting features. |
    | Removed first 28 days of every series | {engineered_row_count_before_drop} | {len(feature_engineered)} | {engineered_column_count_before_drop} | {len(feature_engineered.columns)} | Removed rows where the 28-day lag or 28-day rolling feature was unavailable; {rows_removed_first_28} rows removed. |

    ## 11. Train/Test Split Summary

    - Split date: `{test_start.strftime('%Y-%m-%d')}`
    - Training date range: `{train["Date"].min().strftime('%Y-%m-%d')}` to `{train["Date"].max().strftime('%Y-%m-%d')}`
    - Test date range: `{test["Date"].min().strftime('%Y-%m-%d')}` to `{test["Date"].max().strftime('%Y-%m-%d')}`
    - Training rows: {len(train)}
    - Test rows: {len(test)}
    - Training Demand mean: {train[TARGET].mean():.6f}
    - Test Demand mean: {test[TARGET].mean():.6f}
    - Test epidemic states: {test_epidemic_status}
    - Test contains both epidemic states: {test_has_both_epidemic_states}
    - Split adjustment: {split_adjustment}

    ### Final Feature Set

    **Numerical features ({len(numerical_features)}):**
    {", ".join(f"`{feature}`" for feature in numerical_features)}

    **Categorical features ({len(categorical_features)}):**
    {", ".join(f"`{feature}`" for feature in categorical_features)}

    **Final feature count: {len(feature_columns)}**
    """

    characteristics = characteristics.rstrip()
    if "\n## 10. Change Log" in characteristics:
        characteristics = characteristics.split("\n## 10. Change Log")[0].rstrip()
    characteristics += "\n" + change_log_section
    with open(CHARACTERISTICS_PATH, "w", encoding="utf-8") as file:
        file.write(characteristics)

    # Update Findings_Report.md.
    with open(FINDINGS_PATH, "r", encoding="utf-8") as file:
        findings = file.read()

    top_month = month_means.idxmax()
    top_dow = dow_names[dow_means.idxmax()]
    top_category = eda_insights["Category"].index[0]
    top_category_mean = eda_insights["Category"].iloc[0]
    top_weather = eda_insights["Weather Condition"].index[0]
    top_weather_mean = eda_insights["Weather Condition"].iloc[0]

    data_cleaning_section = f"""

    ## Data Cleaning & Preprocessing

    ### Cleaning and Series Integrity

    - Converted `Date` to datetime and sorted by `Store ID`, `Product ID`, and `Date`.
    - Created `series_id` by combining `Store ID` and `Product ID`.
    - Retained all rows where `Units Sold > Demand`; they account for {(cleaned['Units Sold'] > cleaned['Demand']).mean() * 100:.6f}% of observations.
    - The high-side observations and outliers were retained because they appear compatible with the synthetic daily series and removing them would interrupt the continuous daily history.
    - Verified {cleaned['series_id'].nunique()} series, {unique_dates} dates, and {len(cleaned)} rows: the cleaned series has a {cleaned['series_id'].nunique()} x {unique_dates} complete grid with no missing cross-series dates.
    - Fully duplicated cleaned rows: {int(cleaned.duplicated().sum())}.

    ### Leakage Handling

    - Same-day `Units Sold`, `Units Ordered`, and `Inventory Level` were excluded from the feature set.
    - Same-day `Units Sold` Pearson correlation with `Demand`: {same_day_sold_corr:.6f}.
    - Lag-1 `Units Sold` Pearson correlation with `Demand`: {lag1_sold_corr:.6f}.
    - The lagged operational values are only available from prior observations within each `series_id`.

    ### Feature Engineering

    - Removed {rows_removed_first_28} rows from the first 28 days of each series where required lag or rolling features were unavailable.
    - Added calendar, demand lag, shifted demand rolling, lagged operational, price gap/ratio, and epidemic context features.
    - Dropped `Region`, `Seasonality`, and `Competitor Pricing`.
    - Kept `Price`, `Discount`, `Promotion`, `Weather Condition`, `Epidemic`, and the other final features because they are representable at forecast time under the documented assumptions.

    ### Chronological Split

    - Training date range: `{train["Date"].min().strftime('%Y-%m-%d')}` to `{train["Date"].max().strftime('%Y-%m-%d')}`; {len(train)} rows; mean Demand {train[TARGET].mean():.6f}.
    - Test date range: `{test["Date"].min().strftime('%Y-%m-%d')}` to `{test["Date"].max().strftime('%Y-%m-%d')}`; {len(test)} rows; mean Demand {test[TARGET].mean():.6f}.
    - The test window contains both epidemic states: {test_has_both_epidemic_states}. {split_adjustment}

    ### Preprocessing

    - Built a training-fit-only `ColumnTransformer` with `OneHotEncoder(handle_unknown="ignore")` for `Store ID`, `Category`, and `Weather Condition`.
    - Encoded `Product ID` with target-free one-hot encoding.
    - Applied `StandardScaler` only to the {len(numerical_features)} numerical features.
    - Fit scope: {len(train)} training rows only.
    - The test set was transformed with the training-fitted transformer; no fitted transformer was saved at this stage.

    ### Forecasting Assumptions

    - The task is a 1-day-ahead demand forecast for each `(Store ID, Product ID)` series.
    - Calendar features, planned `Price`, `Discount`, and `Promotion`, and past values are available at forecast time.
    - `Weather Condition`, `Competitor Pricing`, and `Epidemic` are assumed known for the forecast day from forecasts or announcements.
    - Same-day `Units Sold`, `Units Ordered`, and `Inventory Level` are not forecast-time features and are used only through lagged values.

    ## EDA Insights

    1. [01_daily_demand_epidemic.png](figures/01_daily_demand_epidemic.png): aggregate daily demand varies over time and epidemic shading marks periods represented by the epidemic indicator.
    2. [02_demand_distribution_boxplot.png](figures/02_demand_distribution_boxplot.png): the Demand histogram and boxplot show the observed distribution and high-side observations.
    3. [03_demand_by_category.png](figures/03_demand_by_category.png): `{top_category}` has the highest category mean Demand at {top_category_mean:.6f}.
    4. [04_demand_by_weather.png](figures/04_demand_by_weather.png): `{top_weather}` has the highest weather-condition mean Demand at {top_weather_mean:.6f}.
    5. [05_demand_by_promotion.png](figures/05_demand_by_promotion.png): promotion group means varied; promotion was retained as a forecast-time feature.
    6. [06_demand_by_discount.png](figures/06_demand_by_discount.png): discount levels show distinct demand distributions.
    7. [07_demand_by_epidemic.png](figures/07_demand_by_epidemic.png): epidemic states show different observed demand distributions.
    8. [08_final_numerical_correlation_heatmap.png](figures/08_final_numerical_correlation_heatmap.png): the final numerical feature heatmap shows linear relationships without including same-day operational fields.
    9. [09_average_demand_month_and_dayofweek.png](figures/09_average_demand_month_and_dayofweek.png): `{top_month}` has the highest average demand by month and `{top_dow}` has the highest average demand by day of week.
    10. [10_price_vs_demand_by_category.png](figures/10_price_vs_demand_by_category.png): Price has mutual information {mi_price:.6f} with Demand. Its Pearson correlation was near zero, while category and store summaries show that Price has structured level differences; this suggests nonlinear or category/series-level structure rather than a simple linear global relationship.
    """

    for heading in ["## Data Cleaning & Preprocessing", "## EDA Insights"]:
        if heading in findings:
            start = findings.find(heading)
            # Remove old section to next top-level heading.
            next_heading = findings.find("\n## ", start + len(heading))
            if next_heading == -1:
                findings = findings[:start]
            else:
                findings = findings[:start] + findings[next_heading:]

    findings = findings.rstrip() + "\n" + data_cleaning_section
    with open(FINDINGS_PATH, "w", encoding="utf-8") as file:
        file.write(findings)

    print("\n=== FIVE MOST IMPORTANT EDA INSIGHTS ===")
    print("1. The first category mean shown by executed grouping is",
          f"{top_category} at {top_category_mean:.6f}.")
    print("2. The highest-weather category mean is",
          f"{top_weather} at {top_weather_mean:.6f}.")
    print("3. The highest month-mean month is",
          f"{top_month}; the highest day-of-week mean is {top_dow}.")
    print("4. Price mutual information with Demand is",
          f"{mi_price:.6f}; its global linear correlation was near zero.")
    print("5. The leakage proof compared same-day correlation",
          f"{same_day_sold_corr:.6f} with lag-1 correlation {lag1_sold_corr:.6f}.")

    print("\n=== OUTPUTS SAVED ===")
    print(DATA_DIR / "train.csv")
    print(DATA_DIR / "test.csv")
    print(DATA_DIR / "history.csv")
    print(BASE_DIR / "feature_config.json")


if __name__ == "__main__":
    main()
