from pathlib import Path
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import TransformedTargetRegressor

from preprocessing import build_preprocessor

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except Exception:
    LIGHTGBM_AVAILABLE = False

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
FIGURES_DIR = BASE_DIR / "figures"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = BASE_DIR / "feature_config.json"
FINDINGS_PATH = BASE_DIR / "Findings_Report.md"
CHARACTERISTICS_PATH = BASE_DIR / "data_characteristics.md"
DOCUMENTATION_PATH = BASE_DIR / "project_documentation.md"
TARGET = "Demand"
RANDOM_STATE = 42
sns.set_theme(style="whitegrid")

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = json.load(file)

train = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["Date"])
test = pd.read_csv(DATA_DIR / "test.csv", parse_dates=["Date"])
history = pd.read_csv(DATA_DIR / "history.csv", parse_dates=["Date"])


def save_plot(name):
    path = FIGURES_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("PLOT SAVED:", path)
    return name


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": float(mae), "RMSE": float(rmse), "WAPE": float(wape), "R2": float(r2)}


def metric_table(rows):
    result = []
    for name, values in rows.items():
        if not values:
            continue
        keys = ["MAE", "RMSE", "WAPE", "R2"]
        record = {"Model": name}
        for key in keys:
            vals = np.array([value[key] for value in values])
            record[key] = f"{vals.mean():.6f} +/- {vals.std():.6f}"
        record["Training Time Mean"] = f"{np.mean([value['Training Time'] for value in values]):.6f}s"
        result.append(record)
    return pd.DataFrame(result).sort_values("MAE")


train_dates = np.sort(train["Date"].unique())
unique_splitter = TimeSeriesSplit(n_splits=5)
date_splits = []
for fold, (train_positions, valid_positions) in enumerate(unique_splitter.split(train_dates), start=1):
    train_date_values = set(train_dates[train_positions])
    valid_date_values = set(train_dates[valid_positions])
    train_idx = train["Date"].isin(train_date_values)
    valid_idx = train["Date"].isin(valid_date_values)
    date_splits.append((np.flatnonzero(train_idx.to_numpy()), np.flatnonzero(valid_idx.to_numpy())))
    print(f"Fold {fold}: train {pd.Timestamp(train_dates[train_positions].min()).strftime('%Y-%m-%d')} to "
          f"{pd.Timestamp(train_dates[train_positions].max()).strftime('%Y-%m-%d')} | "
          f"validation {pd.Timestamp(train_dates[valid_positions].min()).strftime('%Y-%m-%d')} to "
          f"{pd.Timestamp(train_dates[valid_positions].max()).strftime('%Y-%m-%d')} "
          f"({train_idx.sum()} + {valid_idx.sum()} rows)")

feature_cols = config["feature_list"]
numerical = config["numerical_features"]
categorical = config["categorical_features"]
X = train[feature_cols].copy()
y = train[TARGET].copy()

print("\n=== BASELINES ===")
baseline_specs = {
    "Naive Lag-1": "demand_lag_1",
    "Seasonal Naive Lag-7": "demand_lag_7",
    "Moving Average 7": "demand_roll_mean_7",
}
baseline_cv = {}
for name, column in baseline_specs.items():
    fold_results = []
    for train_idx, valid_idx in date_splits:
        true = y.iloc[valid_idx].to_numpy()
        pred = X.iloc[valid_idx][column].to_numpy()
        result = metrics(true, pred)
        result["Training Time"] = 0.0
        fold_results.append(result)
    baseline_cv[name] = fold_results
baseline_table = metric_table(baseline_cv)
print(baseline_table.to_string(index=False))
best_baseline_name = baseline_table.iloc[0]["Model"]
best_baseline_mae = float(baseline_table.iloc[0]["MAE"].split()[0])

print("\n=== MODEL COMPARISON ===")
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(random_state=RANDOM_STATE),
    "Lasso": Lasso(random_state=RANDOM_STATE, max_iter=3000),
    "Random Forest": RandomForestRegressor(
        n_estimators=80, max_depth=12, min_samples_leaf=3,
        n_jobs=-1, random_state=RANDOM_STATE
    ),
    "HistGradientBoosting": HistGradientBoostingRegressor(
        max_iter=150, learning_rate=0.08, max_depth=8,
        l2_regularization=1.0, random_state=RANDOM_STATE
    ),
}
if LIGHTGBM_AVAILABLE:
    models["LightGBM"] = LGBMRegressor(
        n_estimators=300, learning_rate=0.06, num_leaves=31,
        max_depth=-1, n_jobs=-1, random_state=RANDOM_STATE, verbosity=-1
    )
if XGBOOST_AVAILABLE:
    models["XGBoost"] = XGBRegressor(
        n_estimators=300, learning_rate=0.06, max_depth=6,
        subsample=0.9, colsample_bytree=0.9, n_jobs=-1,
        random_state=RANDOM_STATE, objective="reg:squarederror"
    )

model_cv = {}
for name, estimator in models.items():
    fold_results = []
    print("Running:", name)
    for fold, (train_idx, valid_idx) in enumerate(date_splits, start=1):
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor(CONFIG_PATH)),
            ("model", estimator),
        ])
        start = time.perf_counter()
        pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
        training_time = time.perf_counter() - start
        pred = pipeline.predict(X.iloc[valid_idx])
        result = metrics(y.iloc[valid_idx], pred)
        result["Training Time"] = training_time
        fold_results.append(result)
    model_cv[name] = fold_results

model_table = metric_table(model_cv)
print(model_table.to_string(index=False))
best_model_name = model_table.iloc[0]["Model"]
best_model_cv_mae = float(model_table.iloc[0]["MAE"].split()[0])
print("BEST UNTUNED MODEL:", best_model_name)

untuned = models[best_model_name]

# Raw vs sqrt target experiment.
sqrt_pipeline = Pipeline([
    ("preprocessor", build_preprocessor(CONFIG_PATH)),
    ("model", TransformedTargetRegressor(
        regressor=clone(untuned),
        func=np.sqrt,
        inverse_func=np.square,
    )),
])
raw_pipeline = Pipeline([
    ("preprocessor", build_preprocessor(CONFIG_PATH)),
    ("model", clone(untuned)),
])
transform_results = {}
for target_name, pipeline in [("raw", raw_pipeline), ("sqrt", sqrt_pipeline)]:
    fold_metrics = []
    for train_idx, valid_idx in date_splits:
        start = time.perf_counter()
        pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
        elapsed = time.perf_counter() - start
        pred = pipeline.predict(X.iloc[valid_idx])
        result = metrics(y.iloc[valid_idx], pred)
        result["Training Time"] = elapsed
        fold_metrics.append(result)
    transform_results[target_name] = fold_metrics

raw_mae = np.mean([result["MAE"] for result in transform_results["raw"]])
sqrt_mae = np.mean([result["MAE"] for result in transform_results["sqrt"]])
use_sqrt = sqrt_mae < raw_mae
print("\n=== TARGET TRANSFORM EXPERIMENT ===")
print("Raw MAE:", f"{raw_mae:.6f}")
print("Sqrt MAE:", f"{sqrt_mae:.6f}")
print("Selected target transform:", "sqrt" if use_sqrt else "raw")

# Price ablation.
price_features = ["Price", "price_gap", "price_ratio"]
with_price_results = []
without_price_results = []
for train_idx, valid_idx in date_splits:
    for price_mode, feature_set, collection in [
        ("with", feature_cols, with_price_results),
        ("without", [c for c in feature_cols if c not in price_features], without_price_results),
    ]:
        active_numerical = [c for c in numerical if c in feature_set]
        active_categorical = [c for c in categorical if c in feature_set]
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor(CONFIG_PATH, active_numerical, active_categorical)),
            ("model", clone(untuned)),
        ])
        # build_preprocessor would include Price unless the feature is absent.
        start = time.perf_counter()
        pipeline.fit(train[feature_set].iloc[train_idx], y.iloc[train_idx])
        elapsed = time.perf_counter() - start
        pred = pipeline.predict(train[feature_set].iloc[valid_idx])
        result = metrics(y.iloc[valid_idx], pred)
        result["Training Time"] = elapsed
        collection.append(result)

price_cv_mae = np.mean([result["MAE"] for result in with_price_results])
no_price_cv_mae = np.mean([result["MAE"] for result in without_price_results])
print("\n=== PRICE ABLATION ===")
print("CV MAE with price features:", f"{price_cv_mae:.6f}")
print("CV MAE without price features:", f"{no_price_cv_mae:.6f}")

# Optional relative-price feature experiment.
relative_feature = "price_relative_series_28"
history_full = add_features(history)
relative_values = history_full[["Date", "Store ID", "Product ID", relative_feature]].copy()
for frame in (train, test):
    frame.drop(columns=[relative_feature], errors="ignore", inplace=True)
    frame_with_relative = frame.merge(
        relative_values,
        on=["Date", "Store ID", "Product ID"],
        how="left",
    )
    frame.loc[:, relative_feature] = frame_with_relative[relative_feature].to_numpy()

with_relative_results = []
without_relative_results = []
for train_idx, valid_idx in date_splits:
    for feature_set, collection in [
        (feature_cols, without_relative_results),
        (feature_cols + [relative_feature], with_relative_results),
    ]:
        if relative_feature not in train.columns and relative_feature in feature_set:
            continue
        active_numerical = [c for c in numerical if c in feature_set]
        active_categorical = [c for c in categorical if c in feature_set]
        if relative_feature in feature_set:
            active_numerical = active_numerical + [relative_feature]
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor(CONFIG_PATH, active_numerical, active_categorical)),
            ("model", clone(untuned)),
        ])
        start = time.perf_counter()
        pipeline.fit(train[feature_set].iloc[train_idx], y.iloc[train_idx])
        elapsed = time.perf_counter() - start
        pred = pipeline.predict(train[feature_set].iloc[valid_idx])
        result = metrics(y.iloc[valid_idx], pred)
        result["Training Time"] = elapsed
        collection.append(result)

relative_cv_mae = np.mean([result["MAE"] for result in with_relative_results])
base_relative_mae = np.mean([result["MAE"] for result in without_relative_results])
use_relative_feature = len(with_relative_results) > 0 and relative_cv_mae < base_relative_mae
print("\n=== OPTIONAL RELATIVE PRICE FEATURE ===")
print("CV MAE without relative price:", f"{base_relative_mae:.6f}")
print("CV MAE with relative price:", f"{relative_cv_mae:.6f}")
print("Added to feature config:", use_relative_feature)

if use_relative_feature:
    config["feature_list"].append(relative_feature)
    config["numerical_features"].append(relative_feature)
    config["optional_feature_experiment"] = {
        "feature": relative_feature,
        "without_cv_mae": base_relative_mae,
        "with_cv_mae": relative_cv_mae,
        "added": True,
    }
else:
    config["optional_feature_experiment"] = {
        "feature": relative_feature,
        "without_cv_mae": base_relative_mae,
        "with_cv_mae": relative_cv_mae,
        "added": False,
    }

with open(CONFIG_PATH, "w", encoding="utf-8") as file:
    json.dump(config, file, indent=2)

feature_cols = config["feature_list"]
numerical = config["numerical_features"]
categorical = config["categorical_features"]
X = train[feature_cols].copy()

# Hyperparameter tuning for the untuned best model.
search_spaces = {
    "Linear Regression": {},
    "Ridge": {"model__alpha": np.logspace(-3, 3, 100)},
    "Lasso": {"model__alpha": np.logspace(-4, 1, 100)},
    "Random Forest": {
        "model__n_estimators": [60, 80, 120, 160],
        "model__max_depth": [6, 10, 14, 18],
        "model__min_samples_leaf": [1, 2, 3, 5],
        "model__max_features": ["sqrt", 0.7, 1.0],
    },
    "HistGradientBoosting": {
        "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
        "model__max_iter": [100, 150, 220],
        "model__max_depth": [4, 6, 8, 10],
        "model__l2_regularization": [0, 0.1, 1.0, 5.0],
    },
    "LightGBM": {
        "model__n_estimators": [150, 250, 350, 500],
        "model__learning_rate": [0.03, 0.05, 0.08],
        "model__num_leaves": [15, 31, 63],
        "model__max_depth": [-1, 6, 10],
    },
    "XGBoost": {
        "model__n_estimators": [150, 250, 350, 500],
        "model__learning_rate": [0.03, 0.05, 0.08],
        "model__max_depth": [3, 5, 7],
        "model__subsample": [0.7, 0.9, 1.0],
        "model__colsample_bytree": [0.7, 0.9, 1.0],
    },
}

search_pipeline = Pipeline([
    ("preprocessor", build_preprocessor(CONFIG_PATH)),
    ("model", (
        TransformedTargetRegressor(
            regressor=clone(untuned),
            func=np.sqrt,
            inverse_func=np.square,
        ) if use_sqrt else clone(untuned)
    )),
])
search_space = search_spaces[best_model_name]
if use_sqrt and search_space:
    search_space = {
        key.replace("model__", "model__regressor__", 1): value
        for key, value in search_space.items()
    }
if search_space:
    search = RandomizedSearchCV(
        search_pipeline,
        search_space,
        n_iter=30,
        scoring="neg_mean_absolute_error",
        cv=date_splits,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X, y)
    tuned_pipeline = search.best_estimator_
    tuned_cv_mae = -search.best_score_
    best_params = search.best_params_
else:
    tuned_pipeline = clone(search_pipeline)
    tuned_pipeline.fit(X, y)
    tuned_cv_mae = best_model_cv_mae
    best_params = {}

improvement_cv = best_model_cv_mae - tuned_cv_mae
print("\n=== HYPERPARAMETER TUNING ===")
print("Search space:", search_space)
print("Best parameters:", best_params)
print("Untuned CV MAE:", f"{best_model_cv_mae:.6f}")
print("Tuned CV MAE:", f"{tuned_cv_mae:.6f}")
print("CV MAE improvement:", f"{improvement_cv:.6f}")

# Final test evaluation (test used once).
final_pipeline = clone(tuned_pipeline)
final_pipeline.fit(X, y)
test_pred = final_pipeline.predict(test[feature_cols])

test_metrics = metrics(test[TARGET], test_pred)
baseline_test_pred = test["demand_lag_1"].to_numpy()
baseline_test_metrics = metrics(test[TARGET], baseline_test_pred)
mae_improvement_pct = (
    (baseline_test_metrics["MAE"] - test_metrics["MAE"])
    / baseline_test_metrics["MAE"] * 100
)

no_price_features = [c for c in feature_cols if c not in price_features]
no_price_pipeline = clone(tuned_pipeline)
active_no_price_numerical = [c for c in numerical if c in no_price_features]
active_no_price_categorical = [c for c in categorical if c in no_price_features]
no_price_pipeline = Pipeline([
    ("preprocessor", build_preprocessor(CONFIG_PATH, active_no_price_numerical, active_no_price_categorical)),
    ("model", clone(tuned_pipeline.named_steps["model"])),
])
no_price_pipeline.fit(train[no_price_features], y)
no_price_test_pred = no_price_pipeline.predict(test[no_price_features])
no_price_test_metrics = metrics(test[TARGET], no_price_test_pred)

print("\n=== FINAL TEST EVALUATION ===")
print("Best baseline on test (Lag-1):", baseline_test_metrics)
print("Final model test:", test_metrics)
print("MAE improvement over best baseline:", f"{mae_improvement_pct:.6f}%")
print("No-Price test:", no_price_test_metrics)

# Save evaluation and deployment models.
joblib.dump(final_pipeline, MODELS_DIR / "demand_model_eval.joblib")
full_train_test = pd.concat([train, test], ignore_index=True)
deployment_pipeline = clone(tuned_pipeline)
deployment_pipeline.fit(full_train_test[feature_cols], full_train_test[TARGET])
joblib.dump(deployment_pipeline, MODELS_DIR / "demand_model.joblib")
print("Saved:", MODELS_DIR / "demand_model_eval.joblib")
print("Saved:", MODELS_DIR / "demand_model.joblib")

# Error analysis.
test_analysis = test.copy()
test_analysis["Actual"] = test[TARGET]
test_analysis["Predicted"] = test_pred
test_analysis["Residual"] = test_analysis["Actual"] - test_analysis["Predicted"]

def grouped_error(column):
    return test_analysis.groupby(column).apply(
        lambda frame: pd.Series({
            "Rows": len(frame),
            "MAE": mean_absolute_error(frame["Actual"], frame["Predicted"]),
            "Mean Residual": frame["Residual"].mean(),
        }),
        include_groups=False,
    ).sort_values("MAE", ascending=False)

print("\n=== ERROR ANALYSIS ===")
epidemic_error = grouped_error("Epidemic")
category_error = grouped_error("Category")
store_error = grouped_error("Store ID")
print("EPidemic:")
print(epidemic_error)
print("Category:")
print(category_error)
print("Store:")
print(store_error)

series_error = test_analysis.groupby(["Store ID", "Product ID"]).apply(
    lambda frame: pd.Series({
        "MAE": mean_absolute_error(frame["Actual"], frame["Predicted"]),
        "Rows": len(frame),
    }),
    include_groups=False,
)
best_series = series_error.nsmallest(5, "MAE")
worst_series = series_error.nlargest(5, "MAE")
print("5 best series:")
print(best_series)
print("5 worst series:")
print(worst_series)

print("Bias overall:", f"{test_analysis['Residual'].mean():.6f}")
epidemic_bias = test_analysis.groupby("Epidemic")["Residual"].mean()
print("Bias by Epidemic:", epidemic_bias.to_dict())

# Error plots.
example_series = test_analysis.groupby(["Store ID", "Product ID"]).size().index[:3]
for index, (store, product) in enumerate(example_series, start=1):
    frame = test_analysis[
        (test_analysis["Store ID"] == store)
        & (test_analysis["Product ID"] == product)
    ].sort_values("Date")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(frame["Date"], frame["Actual"], label="Actual", linewidth=1.5)
    ax.plot(frame["Date"], frame["Predicted"], label="Predicted", linewidth=1.5)
    for date in frame.loc[frame["Epidemic"] == 1, "Date"]:
        ax.axvspan(date, date + pd.Timedelta(days=1), color="red", alpha=0.025)
    ax.set_title(f"Actual vs Predicted: {store} / {product}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Demand")
    ax.legend()
    save_plot(f"11_actual_vs_predicted_{index}.png")

fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(test_analysis["Residual"], bins=50, kde=True, ax=ax)
ax.set_title("Residual Distribution")
ax.set_xlabel("Actual - Predicted")
save_plot("12_residual_distribution.png")

fig, ax = plt.subplots(figsize=(7, 7))
sample = test_analysis.sample(min(5000, len(test_analysis)), random_state=RANDOM_STATE)
ax.scatter(sample["Actual"], sample["Predicted"], alpha=0.25)
max_value = max(sample["Actual"].max(), sample["Predicted"].max())
ax.plot([0, max_value], [0, max_value], color="red", linestyle="--")
ax.set_xlabel("Actual Demand")
ax.set_ylabel("Predicted Demand")
ax.set_title("Predicted vs Actual")
save_plot("13_predicted_vs_actual.png")

daily = test_analysis.groupby("Date").agg(
    Actual=("Actual", "sum"),
    Predicted=("Predicted", "sum"),
)
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(daily.index, daily["Actual"], label="Actual")
ax.plot(daily.index, daily["Predicted"], label="Predicted")
ax.set_title("Daily Total Actual vs Predicted Demand")
ax.set_xlabel("Date")
ax.set_ylabel("Total Demand")
ax.legend()
save_plot("14_daily_actual_vs_predicted.png")

# Built-in and permutation feature importance.
built_in = None
model_step = final_pipeline.named_steps["model"]
if hasattr(model_step, "feature_importances_"):
    # Map importances through transformed feature names.
    try:
        names = final_pipeline.named_steps["preprocessor"].get_feature_names_out()
    except Exception:
        names = np.array(feature_cols)
    built_in = pd.DataFrame({"Feature": names, "Importance": model_step.feature_importances_})
elif hasattr(model_step, "coef_"):
    try:
        names = final_pipeline.named_steps["preprocessor"].get_feature_names_out()
    except Exception:
        names = np.array(feature_cols)
    values = np.abs(model_step.coef_)
    built_in = pd.DataFrame({"Feature": names, "Importance": values})

perm = permutation_importance(
    final_pipeline,
    test[feature_cols],
    test[TARGET],
    n_repeats=5,
    random_state=RANDOM_STATE,
    scoring="neg_mean_absolute_error",
)
perm_importance = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": perm.importances_mean,
}).sort_values("Importance", ascending=False)

print("\n=== FEATURE IMPORTANCE ===")
if built_in is not None:
    print("Built-in top 20:")
    print(built_in.sort_values("Importance", ascending=False).head(20).to_string(index=False))
print("Permutation top 20:")
print(perm_importance.head(20).to_string(index=False))

top_builtin = (
    built_in.sort_values("Importance", ascending=False).head(20)
    if built_in is not None else pd.DataFrame(columns=["Feature", "Importance"])
)
top_perm = perm_importance.head(20)
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
if not top_builtin.empty:
    sns.barplot(data=top_builtin.sort_values("Importance"), x="Importance", y="Feature", ax=axes[0])
else:
    axes[0].text(0.5, 0.5, "No built-in importance", ha="center")
axes[0].set_title("Built-in Feature Importance (Top 20)")
sns.barplot(data=top_perm.sort_values("Importance"), x="Importance", y="Feature", ax=axes[1])
axes[1].set_title("Permutation Importance on Test (Top 20)")
save_plot("15_feature_importance.png")

top_three = perm_importance.head(3)["Feature"].tolist()

# Create deployment prediction module.
predict_code = '''from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
from preprocessing import add_features


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "demand_model.joblib"
CONFIG_PATH = BASE_DIR / "feature_config.json"
HISTORY_PATH = BASE_DIR / "data" / "history.csv"


def predict_demand(store_id, product_id, date, price, discount, promotion,
                   weather_condition=None, competitor_pricing=None, epidemic=None):
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = json.load(file)
    model = joblib.load(MODEL_PATH)
    history = pd.read_csv(HISTORY_PATH, parse_dates=["Date"])
    history["series_id"] = history["Store ID"].astype(str) + "__" + history["Product ID"].astype(str)

    requested_store = str(store_id)
    requested_product = str(product_id)
    valid_stores = sorted(history["Store ID"].astype(str).unique())
    valid_pairs = set(zip(history["Store ID"].astype(str), history["Product ID"].astype(str)))
    if requested_store not recognized if False else False:
        pass
    if requested_store not in valid_stores or (requested_store, requested_product) not in valid_pairs:
        raise ValueError("Unknown store/product pair.")

    requested_date = pd.Timestamp(date)
    last_history_date = history["Date"].max()
    next_forecast_date = last_history_date + pd.Timedelta(days=1)
    backtest_mode = requested_date <= last_history_date and requested_date in set(history["Date"].unique())
    forecast_mode = requested_date == next_forecast_date
    if not backtest_mode and not forecast_mode:
        raise ValueError("Unsupported date. Use a date in history or the next day after the last history date.")

    series = history[
        (history["Store ID"].astype(str) == requested_store)
        & (history["Product ID"].astype(str) == requested_product)
    ].sort_values("Date").copy()
    if len(series) == 0:
        raise ValueError("Unknown store/product pair.")

    last_row = series.iloc[-1]
    if weather_condition is None:
        region = last_row["Region"]
        candidates = history[
            (history["Region"] == region)
            & (history["Date"].dt.month == requested_date.month)
        ]
        weather_condition = candidates["Weather Condition"].mode().iloc[0]
    if competitor_pricing is None:
        last_ratio = last_row["Price"] / last_row["Competitor Pricing"] if last_row["Competitor Pricing"] != 0 else 1.0
        competitor_pricing = float(price) / float(last_ratio)
    if epidemic is None:
        epidemic = int(last_row["Epidemic"])

    if forecast_mode:
        planning_row = last_row.copy()
        planning_row["Date"] = requested_date
        planning_row["Price"] = float(price)
        planning_row["Discount"] = float(discount)
        planning_row["Promotion"] = int(promotion)
        planning_row["Weather Condition"] = str(weather_condition)
        planning_row["Competitor Pricing"] = float(competitor_pricing)
        planning_row["Epidemic"] = int(epidemic)
    else:
        planning_row = history[
            (history["Store ID"].astype(str) == requested_store)
            & (history["Product ID"].astype(str) == requested_product)
            & (history["Date"] == requested_date)
        ].iloc[0].copy()

    future_history = pd.concat([history, pd.DataFrame([planning_row])], ignore_index=True)
    future_history = future_history.sort_values(["Store ID", "Product ID", "Date"]).reset_index(drop=True)
    engineered = add_features(future_history)
    match = engineered[
        (engineered["Store ID"].astype(str) == requested_store)
        & (engineered["Product ID"].astype(str) == requested_product)
        & (engineered["Date"] == requested_date)
    ].iloc[0]

    feature_row = pd.DataFrame([match])[config["feature_list"]]
    prediction = float(model.predict(feature_row)[0])
    result = {"prediction": prediction}
    if backtest_mode:
        result["actual"] = float(match["Demand"])
    return result


def get_default_inputs(store_id, product_id):
    history = pd.read_csv(HISTORY_PATH, parse_dates=["Date"])
    series = history[
        (history["Store ID"].astype(str) == str(store_id))
        & (history["Product ID"].astype(str) == str(product_id))
    ].sort_values("Date")
    if len(series) == 0:
        raise ValueError("Unknown store/product pair.")
    last = series.iloc[-1]
    return {
        "price": float(last["Price"]),
        "discount": float(last["Discount"]),
        "promotion": int(last["Promotion"]),
    }
'''
# Remove accidental invalid placeholder by replacing line cleanly.
predict_code = predict_code.replace(
    '    if requested_store not recognized if False else False:\n        pass\n',
    ''
)
(BASE_DIR / "predict.py").write_text(predict_code, encoding="utf-8")
print("Saved:", BASE_DIR / "predict.py")

# 5-row consistency check using eval model.
from predict import predict_demand
random_test_rows = test.sample(5, random_state=RANDOM_STATE)
consistency = []
for _, row in random_test_rows.iterrows():
    raw_history_match = history[
        (history["Store ID"].astype(str) == str(row["Store ID"]))
        & (history["Product ID"].astype(str) == str(row["Product ID"]))
        & (history["Date"] == row["Date"])
    ]
    competitor_from_history = (
        float(raw_history_match.iloc[0]["Competitor Pricing"])
        if len(raw_history_match) else None
    )
    result = predict_demand(
        row["Store ID"],
        row["Product ID"],
        row["Date"].strftime("%Y-%m-%d"),
        row["Price"],
        row["Discount"],
        row["Promotion"],
        weather_condition=row["Weather Condition"],
        competitor_pricing=competitor_from_history,
        epidemic=row["Epidemic"],
        model_path=str(MODELS_DIR / "demand_model_eval.joblib"),
    )
    consistency.append(result["prediction"])
direct_eval_pred = final_pipeline.predict(random_test_rows[feature_cols])
max_consistency_error = float(np.max(np.abs(np.array(consistency) - direct_eval_pred)))
print("\n=== PREDICT CONSISTENCY ===")
print("5 random test predictions:", consistency)
print("Direct eval-pipeline predictions:", direct_eval_pred.tolist())
print("Maximum absolute difference:", f"{max_consistency_error:.12f}")

# Metadata for deployment.
valid_store_pairs = (
    history[["Store ID", "Product ID", "Category"]]
    .drop_duplicates()
    .sort_values(["Store ID", "Product ID"])
)
metadata = {
    "model_name": best_model_name,
    "best_parameters": best_params,
    "feature_list": feature_cols,
    "target_name": TARGET,
    "cv_metrics": model_table.to_dict(orient="records"),
    "baseline_cv_metrics": baseline_table.to_dict(orient="records"),
    "tuned_cv_mae": tuned_cv_mae,
    "test_metrics": test_metrics,
    "test_metrics_no_price": no_price_test_metrics,
    "baseline_test_metrics": baseline_test_metrics,
    "best_baseline_name": best_baseline_name,
    "training_date_range": [
        train["Date"].min().strftime("%Y-%m-%d"),
        train["Date"].max().strftime("%Y-%m-%d"),
    ],
    "test_date_range": [
        test["Date"].min().strftime("%Y-%m-%d"),
        test["Date"].max().strftime("%Y-%m-%d"),
    ],
    "last_history_date": history["Date"].max().strftime("%Y-%m-%d"),
    "valid_stores": sorted(history["Store ID"].astype(str).unique()),
    "valid_store_product_pairs": [
        {"Store ID": str(row["Store ID"]), "Product ID": str(row["Product ID"]), "Category": str(row["Category"])}
        for _, row in valid_store_pairs.iterrows()
    ],
    "valid_weather_values": sorted(history["Weather Condition"].dropna().astype(str).unique()),
    "default_rules": {
        "weather": "Most frequent weather for the store's Region in the requested calendar month",
        "competitor_pricing": "Given price divided by the series' last observed Price / Competitor Pricing ratio",
        "epidemic": "Last known epidemic value for the series",
        "price_discount_promotion": "Available from get_default_inputs() using each series' last known values",
    },
}
with open(MODELS_DIR / "model_metadata.json", "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=2)
print("Saved:", MODELS_DIR / "model_metadata.json")
