from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationInfo, field_validator

from predict import (
    PredictionError,
    get_input_defaults,
    load_prediction_data,
    predict_backtest,
    predict_demand,
)


BASE_DIR = Path(__file__).resolve().parents[1]
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"
FRONTEND_ORIGINS = ["http://localhost:5500", "http://127.0.0.1:5500"]
_VALID_DISCOUNTS = set()
_VALID_PROMOTION = set()
_VALID_EPIDEMIC = set()
_VALID_WEATHER = set()


def validate_date_text(value):
    if not isinstance(value, str) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None:
        raise ValueError("Date must use YYYY-MM-DD format.")
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError as error:
        raise ValueError("Date must use YYYY-MM-DD format and be a valid calendar date.") from error
    return value


class PredictRequest(BaseModel):
    store_id: str
    product_id: str
    date: str
    discount: float | None = None
    promotion: int | None = None
    weather_condition: str | None = None
    epidemic: int | None = None

    @field_validator("date")
    @classmethod
    def check_date_format(cls, value):
        return validate_date_text(value)

    @field_validator("discount")
    @classmethod
    def check_discount(cls, value):
        if value is not None and value not in _VALID_DISCOUNTS:
            allowed = ", ".join(str(int(item)) if item.is_integer() else str(item) for item in sorted(_VALID_DISCOUNTS))
            raise ValueError(f"Discount must be one of the values in history: {allowed}.")
        return value

    @field_validator("promotion", "epidemic")
    @classmethod
    def check_binary_input(cls, value, info: ValidationInfo):
        allowed = _VALID_PROMOTION if info.field_name == "promotion" else _VALID_EPIDEMIC
        if value is not None and value not in allowed:
            choices = ", ".join(str(item) for item in sorted(allowed))
            raise ValueError(f"Value must be one of: {choices}.")
        return value

    @field_validator("weather_condition")
    @classmethod
    def check_weather(cls, value):
        if value is not None and value not in _VALID_WEATHER:
            allowed = ", ".join(sorted(_VALID_WEATHER))
            raise ValueError(f"Weather must be one of: {allowed}.")
        return value


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _VALID_DISCOUNTS, _VALID_PROMOTION, _VALID_EPIDEMIC, _VALID_WEATHER
    model, config, history = load_prediction_data()
    with METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    history_values = {
        "valid_discount_values": {float(value) for value in history["Discount"].dropna().unique()},
        "valid_promotion_values": {int(value) for value in history["Promotion"].dropna().unique()},
        "valid_epidemic_values": {int(value) for value in history["Epidemic"].dropna().unique()},
        "valid_weather_values": set(history["Weather Condition"].dropna().unique()),
    }
    for key, observed_values in history_values.items():
        if observed_values != set(metadata[key]):
            raise RuntimeError(f"Metadata field '{key}' does not match data/history.csv.")
    _VALID_DISCOUNTS = set(float(value) for value in metadata["valid_discount_values"])
    _VALID_PROMOTION = set(metadata["valid_promotion_values"])
    _VALID_EPIDEMIC = set(metadata["valid_epidemic_values"])
    _VALID_WEATHER = set(metadata["valid_weather_values"])
    app.state.model = model
    app.state.config = config
    app.state.history = history
    app.state.metadata = metadata
    yield
    _VALID_DISCOUNTS = set()
    _VALID_PROMOTION = set()
    _VALID_EPIDEMIC = set()
    _VALID_WEATHER = set()


app = FastAPI(
    title="Demand Forecasting API",
    description="One-day-ahead demand predictions for each store-product series.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _validate_pair(request, store_id, product_id):
    pairs = request.app.state.metadata["valid_store_product_pairs"]
    if not any(
        pair["Store ID"] == str(store_id) and pair["Product ID"] == str(product_id)
        for pair in pairs
    ):
        raise HTTPException(status_code=400, detail="Unknown store/product pair.")


def _mode_for_date(history, value):
    date = pd.Timestamp(validate_date_text(value))
    last_history_date = history["Date"].max()
    forecast_date = last_history_date + pd.Timedelta(days=1)
    if date <= last_history_date and date in set(history["Date"].unique()):
        return "backtest"
    if date == forecast_date:
        return "forecast"
    raise PredictionError(
        "Unsupported date. Use a date in history or the next day after the last history date."
    )


@app.get("/health", description="Check API readiness and the model's current history and forecast dates.")
def health(request: Request):
    metadata = request.app.state.metadata
    last_history_date = request.app.state.history["Date"].max()
    return {
        "status": "ok",
        "model_name": metadata["model_name"],
        "last_history_date": last_history_date.strftime("%Y-%m-%d"),
        "forecast_date": (last_history_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
    }


@app.get("/metadata", description="Get model metrics, improvement, drivers, assumptions, and supported dates.")
def get_metadata(request: Request):
    metadata = request.app.state.metadata
    test_mae = float(metadata["test_metrics"]["MAE"])
    baseline_mae = float(metadata["baseline_test_metrics"]["MAE"])
    improvement = (baseline_mae - test_mae) / baseline_mae * 100
    return {
        "model_name": metadata["model_name"],
        "cv_metrics": metadata["cv_metrics"][metadata["model_name"]],
        "test_metrics": metadata["test_metrics"],
        "baseline_test_metrics": metadata["baseline_test_metrics"],
        "improvement_vs_baseline": {
            "baseline_model": metadata["best_baseline_name"],
            "baseline_mae": baseline_mae,
            "percentage": improvement,
        },
        "top_drivers": metadata["top_drivers"],
        "assumptions": metadata["assumptions"],
        "known_weak_spots": metadata["known_weak_spots"],
        "supported_date_range": metadata["supported_date_range"],
        "test_date_range": metadata["test_date_range"],
        "last_history_date": metadata["last_history_date"],
        "forecast_date": (pd.Timestamp(metadata["last_history_date"]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        "valid_weather_values": metadata["valid_weather_values"],
        "valid_discount_values": metadata["valid_discount_values"],
        "valid_promotion_values": metadata["valid_promotion_values"],
        "valid_epidemic_values": metadata["valid_epidemic_values"],
        "price_feature_used": "Price" in metadata["feature_list"],
    }


@app.get("/series", description="List available stores and their valid products and categories.")
def get_series(request: Request):
    metadata = request.app.state.metadata
    stores = []
    for store_id in metadata["valid_stores"]:
        products = [
            {
                "product_id": pair["Product ID"],
                "category": pair["Category"],
            }
            for pair in metadata["valid_store_product_pairs"]
            if pair["Store ID"] == store_id
        ]
        stores.append({"store_id": store_id, "products": products})
    return {"stores": stores}


@app.get("/defaults", description="Return date-aware prefill values and the rule used for each field.")
def defaults(
    request: Request,
    store_id: str = Query(...),
    product_id: str = Query(...),
    date: str = Query(...),
):
    _validate_pair(request, store_id, product_id)
    try:
        validate_date_text(date)
        _mode_for_date(request.app.state.history, date)
        return get_input_defaults(
            store_id,
            product_id,
            date,
            history=request.app.state.history,
            metadata=request.app.state.metadata,
        )
    except PredictionError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/predict", description="Predict demand for a historical backtest date or the next forecast date.")
def predict(request_body: PredictRequest, request: Request):
    _validate_pair(request, request_body.store_id, request_body.product_id)
    try:
        mode = _mode_for_date(request.app.state.history, request_body.date)
        defaults_data = get_input_defaults(
            request_body.store_id,
            request_body.product_id,
            request_body.date,
            history=request.app.state.history,
            metadata=request.app.state.metadata,
        )
        history = request.app.state.history
        if mode == "forecast":
            used = {
                "discount": request_body.discount if request_body.discount is not None else defaults_data["values"]["discount"],
                "promotion": request_body.promotion if request_body.promotion is not None else defaults_data["values"]["promotion"],
                "weather_condition": request_body.weather_condition or defaults_data["values"]["weather_condition"],
                "epidemic": request_body.epidemic if request_body.epidemic is not None else defaults_data["values"]["epidemic"],
                "price": defaults_data["model_context"]["price"],
                "competitor_pricing": defaults_data["model_context"]["competitor_pricing"],
            }
            defaulted = [
                name
                for name in ("discount", "promotion", "weather_condition", "epidemic")
                if getattr(request_body, name) is None
            ]
            result = predict_demand(
                request_body.store_id,
                request_body.product_id,
                request_body.date,
                price=used["price"],
                discount=used["discount"],
                promotion=used["promotion"],
                weather_condition=used["weather_condition"],
                competitor_pricing=used["competitor_pricing"],
                epidemic=used["epidemic"],
                model=request.app.state.model,
                config=request.app.state.config,
                history=history,
            )
            input_sources = {
                name: ("default" if name in defaulted else "provided")
                for name in ("discount", "promotion", "weather_condition", "epidemic")
            }
            input_sources.update({"price": "last_observed", "competitor_pricing": "last_observed"})
        else:
            row = history[
                (history["Store ID"].astype(str) == request_body.store_id)
                & (history["Product ID"].astype(str) == request_body.product_id)
                & (history["Date"] == pd.Timestamp(request_body.date))
            ].iloc[0]
            used = {
                "discount": float(row["Discount"]),
                "promotion": int(row["Promotion"]),
                "weather_condition": str(row["Weather Condition"]),
                "epidemic": int(row["Epidemic"]),
                "price": float(row["Price"]),
                "competitor_pricing": float(row["Competitor Pricing"]),
            }
            defaulted = []
            input_sources = {name: "history" for name in used}
            result = predict_demand(
                request_body.store_id,
                request_body.product_id,
                request_body.date,
                price=used["price"],
                discount=used["discount"],
                promotion=used["promotion"],
                weather_condition=used["weather_condition"],
                competitor_pricing=used["competitor_pricing"],
                epidemic=used["epidemic"],
                model=request.app.state.model,
                config=request.app.state.config,
                history=history,
            )
    except PredictionError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    raw_prediction = result["prediction"]
    test_mae = float(request.app.state.metadata["test_metrics"]["MAE"])
    response = {
        "date": request_body.date,
        "mode": mode,
        "prediction": round(raw_prediction, 2),
        "raw_prediction": raw_prediction,
        "typical_error_range": {
            "low": raw_prediction - test_mae,
            "high": raw_prediction + test_mae,
            "test_mae": test_mae,
        },
        "inputs_used": used,
        "input_sources": input_sources,
        "defaulted_inputs": defaulted,
    }
    if mode == "backtest":
        actual = result["actual"]
        response.update({
            "actual_demand": actual,
            "error": raw_prediction - actual,
            "absolute_error": abs(raw_prediction - actual),
        })
    return response


@app.get("/history", description="Get actual daily demand and epidemic flags for a store-product series.")
def get_history(
    request: Request,
    store_id: str = Query(...),
    product_id: str = Query(...),
    end_date: str | None = Query(None),
    days: int = Query(60, ge=1),
):
    _validate_pair(request, store_id, product_id)
    history = request.app.state.history
    if end_date is None:
        end = history["Date"].max()
    else:
        try:
            validate_date_text(end_date)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        end = pd.Timestamp(end_date)
        if end > history["Date"].max():
            raise HTTPException(status_code=400, detail="History is not available after the last history date.")
    start = end - pd.Timedelta(days=days - 1)
    rows = history[
        (history["Store ID"].astype(str) == store_id)
        & (history["Product ID"].astype(str) == product_id)
        & (history["Date"] >= start)
        & (history["Date"] <= end)
    ].sort_values("Date")
    return {
        "store_id": store_id,
        "product_id": product_id,
        "series": [
            {
                "date": row.Date.strftime("%Y-%m-%d"),
                "demand": float(row.Demand),
                "epidemic": int(row.Epidemic),
            }
            for row in rows.itertuples(index=False)
        ],
    }


@app.get("/backtest", description="Batch-predict a historical date range and report its mean absolute error.")
def backtest(
    request: Request,
    store_id: str = Query(...),
    product_id: str = Query(...),
    start: str = Query(...),
    end: str = Query(...),
):
    _validate_pair(request, store_id, product_id)
    try:
        validate_date_text(start)
        validate_date_text(end)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    start_date = pd.Timestamp(start)
    end_date = pd.Timestamp(end)
    if start_date > end_date:
        raise HTTPException(status_code=400, detail="Start date must be on or before end date.")
    if (end_date - start_date).days + 1 > 120:
        raise HTTPException(status_code=400, detail="Backtest ranges may not exceed 120 days.")
    if end_date > request.app.state.history["Date"].max():
        raise HTTPException(status_code=400, detail="Backtest dates must be within the history range.")
    try:
        predictions = predict_backtest(
            store_id,
            product_id,
            start,
            end,
            request.app.state.model,
            request.app.state.config,
            request.app.state.history,
        )
    except PredictionError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    mae = float(np.mean([abs(row["prediction"] - row["actual"]) for row in predictions]))
    return {
        "store_id": store_id,
        "product_id": product_id,
        "start": start,
        "end": end,
        "mae": mae,
        "predictions": predictions,
    }
