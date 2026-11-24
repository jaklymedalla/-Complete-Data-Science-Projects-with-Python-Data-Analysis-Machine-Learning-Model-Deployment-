import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from predict import predict_demand


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "ok"
    assert result["model_name"] == client.app.state.metadata["model_name"]
    assert result["forecast_date"] == "2024-01-31"


def test_series(client):
    response = client.get("/series")
    assert response.status_code == 200
    stores = response.json()["stores"]
    assert stores
    assert all(store["products"] for store in stores)
    assert all("category" in product for store in stores for product in store["products"])


def test_defaults(client):
    response = client.get(
        "/defaults",
        params={"store_id": "S001", "product_id": "P0001", "date": "2024-01-31"},
    )
    assert response.status_code == 200
    result = response.json()
    assert set(result["values"]) == {"discount", "promotion", "weather_condition", "epidemic"}
    assert set(result["rules"]) == set(result["values"])


def test_forecast_prediction(client):
    response = client.post(
        "/predict",
        json={"store_id": "S001", "product_id": "P0001", "date": "2024-01-31"},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["mode"] == "forecast"
    assert isinstance(result["prediction"], float)
    assert result["defaulted_inputs"] == [
        "discount",
        "promotion",
        "weather_condition",
        "epidemic",
    ]


def test_backtest_prediction(client):
    response = client.post(
        "/predict",
        json={"store_id": "S001", "product_id": "P0001", "date": "2023-12-01"},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["mode"] == "backtest"
    assert "actual_demand" in result
    assert "error" in result


def test_backtest_range_returns_batch_metrics(client):
    response = client.get(
        "/backtest",
        params={
            "store_id": "S001",
            "product_id": "P0001",
            "start": "2023-12-01",
            "end": "2023-12-03",
        },
    )
    assert response.status_code == 200
    result = response.json()
    assert len(result["predictions"]) == 3
    assert result["mae"] == pytest.approx(
        sum(abs(row["prediction"] - row["actual"]) for row in result["predictions"]) / 3
    )
    too_long = client.get(
        "/backtest",
        params={
            "store_id": "S001",
            "product_id": "P0001",
            "start": "2023-10-02",
            "end": "2024-01-30",
        },
    )
    assert too_long.status_code == 400


def test_invalid_inputs(client):
    unknown_store = client.post(
        "/predict",
        json={"store_id": "unknown", "product_id": "P0001", "date": "2024-01-31"},
    )
    invalid_date = client.post(
        "/predict",
        json={"store_id": "S001", "product_id": "P0001", "date": "2024-02-31"},
    )
    invalid_discount = client.post(
        "/predict",
        json={
            "store_id": "S001",
            "product_id": "P0001",
            "date": "2024-01-31",
            "discount": 6,
        },
    )
    invalid_promotion = client.post(
        "/predict",
        json={
            "store_id": "S001",
            "product_id": "P0001",
            "date": "2024-01-31",
            "promotion": 2,
        },
    )
    invalid_weather = client.post(
        "/predict",
        json={
            "store_id": "S001",
            "product_id": "P0001",
            "date": "2024-01-31",
            "weather_condition": "Hail",
        },
    )
    assert unknown_store.status_code == 400
    assert invalid_date.status_code == 422
    assert invalid_discount.status_code == 422
    assert invalid_promotion.status_code == 422
    assert invalid_weather.status_code == 422
    assert "Unknown store/product pair" in unknown_store.json()["detail"]
    assert "YYYY-MM-DD" in invalid_date.json()["detail"][0]["msg"]


def test_predict_matches_direct_predictor_for_three_backtest_rows(client):
    history = client.app.state.history
    selected = history[
        (history["Store ID"].astype(str) == "S001")
        & (history["Product ID"].astype(str) == "P0001")
        & (history["Date"] >= pd.Timestamp("2023-12-01"))
        & (history["Date"] <= pd.Timestamp("2023-12-03"))
    ].sort_values("Date")
    assert len(selected) == 3

    for _, row in selected.iterrows():
        response = client.post(
            "/predict",
            json={
                "store_id": "S001",
                "product_id": "P0001",
                "date": row["Date"].strftime("%Y-%m-%d"),
            },
        )
        assert response.status_code == 200
        expected = predict_demand(
            "S001",
            "P0001",
            row["Date"],
            price=float(row["Price"]),
            discount=float(row["Discount"]),
            promotion=int(row["Promotion"]),
            weather_condition=str(row["Weather Condition"]),
            competitor_pricing=float(row["Competitor Pricing"]),
            epidemic=int(row["Epidemic"]),
            model=client.app.state.model,
            config=client.app.state.config,
            history=history,
        )
        assert response.json()["raw_prediction"] == expected["prediction"]
