from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
from preprocessing import add_features


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "demand_model.joblib"
CONFIG_PATH = BASE_DIR / "feature_config.json"
HISTORY_PATH = BASE_DIR / "data" / "history.csv"


class PredictionError(ValueError):
    """Raised when a prediction request is outside the supported inputs."""


def load_prediction_data():
    """Load the prediction artifacts and source history."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = json.load(file)
    model = joblib.load(MODEL_PATH)
    history = pd.read_csv(HISTORY_PATH, parse_dates=["Date"])
    history["series_id"] = history["Store ID"].astype(str) + "__" + history["Product ID"].astype(str)
    return model, config, history


def _validate_series(history, store_id, product_id):
    store = str(store_id)
    product = str(product_id)
    series = history[
        (history["Store ID"].astype(str) == store)
        & (history["Product ID"].astype(str) == product)
    ].sort_values("Date")
    if series.empty:
        raise PredictionError("Unknown store/product pair.")
    return series


def get_input_defaults(store_id, product_id, date, history=None, metadata=None):
    """Return user-facing defaults plus the hidden price context used by the model."""
    if history is None:
        _, _, history = load_prediction_data()
    series = _validate_series(history, store_id, product_id)
    requested_date = pd.Timestamp(date)
    last = series.iloc[-1]

    month_candidates = history[
        (history["Region"] == last["Region"])
        & (history["Date"].dt.month == requested_date.month)
    ]["Weather Condition"]
    if month_candidates.empty:
        raise PredictionError("No weather history is available for the requested month.")
    weather = str(month_candidates.mode().iloc[0])

    rules = (metadata or {}).get("default_rules", {})
    return {
        "values": {
            "discount": float(last["Discount"]),
            "promotion": int(last["Promotion"]),
            "weather_condition": weather,
            "epidemic": int(last["Epidemic"]),
        },
        "rules": {
            "discount": rules.get("price_discount_promotion", "Last observed value for the series"),
            "promotion": rules.get("price_discount_promotion", "Last observed value for the series"),
            "weather_condition": rules.get("weather", "Most frequent weather for the store's Region in the requested calendar month"),
            "epidemic": rules.get("epidemic", "Last known epidemic value for the series"),
        },
        "model_context": {
            "price": float(last["Price"]),
            "competitor_pricing": float(last["Competitor Pricing"]),
        },
    }


def predict_demand(store_id, product_id, date, price, discount, promotion,
                   weather_condition=None, competitor_pricing=None, epidemic=None,
                   model_path=None, model=None, config=None, history=None):
    if config is None:
        with open(CONFIG_PATH, "r", encoding="utf-8") as file:
            config = json.load(file)
    if model is None:
        model = joblib.load(MODEL_PATH if model_path is None else model_path)
    if history is None:
        history = pd.read_csv(HISTORY_PATH, parse_dates=["Date"])
        history["series_id"] = history["Store ID"].astype(str) + "__" + history["Product ID"].astype(str)

    requested_store = str(store_id)
    requested_product = str(product_id)
    valid_stores = sorted(history["Store ID"].astype(str).unique())
    valid_pairs = set(zip(history["Store ID"].astype(str), history["Product ID"].astype(str)))
    if requested_store not in valid_stores or (requested_store, requested_product) not in valid_pairs:
        raise PredictionError("Unknown store/product pair.")

    try:
        requested_date = pd.Timestamp(date)
    except (TypeError, ValueError) as error:
        raise PredictionError("Invalid date. Use YYYY-MM-DD.") from error
    last_history_date = history["Date"].max()
    next_forecast_date = last_history_date + pd.Timedelta(days=1)
    backtest_mode = requested_date <= last_history_date and requested_date in set(history["Date"].unique())
    forecast_mode = requested_date == next_forecast_date
    if not backtest_mode and not forecast_mode:
        raise PredictionError("Unsupported date. Use a date in history or the next day after the last history date.")

    series = _validate_series(history, requested_store, requested_product).copy()

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

    future_history = history if backtest_mode else pd.concat(
        [history, pd.DataFrame([planning_row])], ignore_index=True
    )
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


def predict_backtest(store_id, product_id, start, end, model, config, history):
    """Predict all dates in a historical range in one model batch."""
    series = _validate_series(history, store_id, product_id)
    start_date = pd.Timestamp(start)
    end_date = pd.Timestamp(end)
    if start_date > end_date:
        raise PredictionError("Start date must be on or before end date.")
    rows = series[(series["Date"] >= start_date) & (series["Date"] <= end_date)]
    if rows.empty:
        raise PredictionError("The requested range contains no history for this series.")

    engineered = add_features(history)
    selected = engineered[
        (engineered["Store ID"].astype(str) == str(store_id))
        & (engineered["Product ID"].astype(str) == str(product_id))
        & (engineered["Date"] >= start_date)
        & (engineered["Date"] <= end_date)
    ].sort_values("Date").copy()
    feature_rows = selected[config["feature_list"]]
    predictions = model.predict(feature_rows)
    return [
        {
            "date": row.Date.strftime("%Y-%m-%d"),
            "prediction": float(prediction),
            "actual": float(row.Demand),
            "epidemic": int(row.Epidemic),
        }
        for row, prediction in zip(selected.itertuples(index=False), predictions)
    ]


def get_default_inputs(store_id, product_id, history=None):
    if history is None:
        _, _, history = load_prediction_data()
    series = _validate_series(history, store_id, product_id)
    last = series.iloc[-1]
    return {
        "price": float(last["Price"]),
        "discount": float(last["Discount"]),
        "promotion": int(last["Promotion"]),
    }
