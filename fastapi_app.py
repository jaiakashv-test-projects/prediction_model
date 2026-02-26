from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from datetime import timedelta, date

from db import engine
from config import SEATS_MODEL_PATH, PRICE_MODEL_PATH
from feature_engineering import create_features, FEATURE_COLUMNS_SEATS, FEATURE_COLUMNS_PRICE


app = FastAPI(
    title="TravelFlux ML API",
    description="AI-powered demand and price prediction service",
    version="1.1"
)


# Load models once at startup with fallback
try:
    seats_model = joblib.load(SEATS_MODEL_PATH)
except:
    old_path = "model/travelflux_xgb.pkl"
    if os.path.exists(old_path):
        seats_model = joblib.load(old_path)
    else:
        seats_model = None

price_model = None
if os.path.exists(PRICE_MODEL_PATH):
    price_model = joblib.load(PRICE_MODEL_PATH)


# -----------------------------
# Health Check
# -----------------------------
@app.get("/")
def home():
    return {
        "status": "TravelFlux ML API running",
        "seats_model_loaded": seats_model is not None,
        "price_model_loaded": price_model is not None
    }


# -----------------------------
# Get all saved predictions
# -----------------------------
@app.get("/predictions")
def get_predictions():

    query = """
    SELECT *
    FROM predictions
    ORDER BY travel_date ASC;
    """

    df = pd.read_sql(query, engine)

    return df.to_dict(orient="records")


# -----------------------------
# Get prediction for specific route
# -----------------------------
@app.get("/predictions/{route_name}")
def get_route_predictions(route_name: str):

    query = f"""
    SELECT *
    FROM predictions
    WHERE route_name = '{route_name}'
    ORDER BY travel_date ASC;
    """

    df = pd.read_sql(query, engine)

    if df.empty:
        raise HTTPException(status_code=404, detail="No predictions found")

    return df.to_dict(orient="records")


# -----------------------------
# Generate fresh prediction (LIVE)
# -----------------------------
@app.post("/predict-live")
def predict_live():
    if not seats_model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    query = """
    SELECT *
    FROM redbus_fill_rates
    ORDER BY route_name, travel_date;
    """

    df = pd.read_sql(query, engine)
    df = create_features(df)

    latest_rows = df.groupby("route_name").tail(1)

    predictions = []
    today = date.today()

    for _, row in latest_rows.iterrows():
        current_features = row.copy()
        
        # We only predict for tomorrow in live mode for simplicity
        next_date = today + timedelta(days=1)
        
        current_features['day_of_week'] = next_date.weekday()
        current_features['day_of_month'] = next_date.day
        current_features['month'] = next_date.month
        current_features['is_weekend'] = 1 if next_date.weekday() in [5, 6] else 0
        
        # Lag 1 is the row we just got
        current_features['lag_1_seats'] = row['filled_seats']
        current_features['lag_1_price'] = row.get('average_price', 1000)

        # Predict seats
        feat_seats = current_features[FEATURE_COLUMNS_SEATS].values.reshape(1, -1)
        predicted_seats = int(seats_model.predict(feat_seats)[0])

        # Predict price
        if price_model:
            feat_price = current_features[FEATURE_COLUMNS_PRICE].values.reshape(1, -1)
            suggested_price = float(price_model.predict(feat_price)[0])
        else:
            # Heuristic
            capacity = row['total_capacity'] if row['total_capacity'] > 0 else 2000
            fill_rate = predicted_seats / capacity
            base_price = row.get('average_price', 1000)
            surge = 1.2 if fill_rate > 0.7 else (1.1 if fill_rate > 0.4 else 1.0)
            suggested_price = base_price * surge

        predictions.append({
            "route_name": row["route_name"],
            "travel_date": next_date,
            "predicted_filled_seats": predicted_seats,
            "suggested_price": round(suggested_price, 2)
        })

    return predictions
