from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from datetime import timedelta

from db import engine
from config import MODEL_PATH
from feature_engineering import create_features, FEATURE_COLUMNS


app = FastAPI(
    title="TravelFlux ML API",
    description="AI-powered demand prediction service",
    version="1.0"
)


# Load model once at startup
model = joblib.load(MODEL_PATH)


# -----------------------------
# Health Check
# -----------------------------
@app.get("/")
def home():
    return {
        "status": "TravelFlux ML API running",
        "model_loaded": True
    }


# -----------------------------
# Get all saved predictions
# -----------------------------
@app.get("/predictions")
def get_predictions():

    query = """
    SELECT *
    FROM predictions
    ORDER BY created_at DESC;
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
    ORDER BY created_at DESC;
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

    query = """
    SELECT *
    FROM redbus_fill_rates
    ORDER BY route_name, travel_date;
    """

    df = pd.read_sql(query, engine)

    df = create_features(df)

    latest_rows = df.groupby("route_name").tail(1)

    predictions = []

    for _, row in latest_rows.iterrows():

        features = row[FEATURE_COLUMNS].values.reshape(1, -1)

        predicted_seats = int(model.predict(features)[0])

        next_date = row['travel_date'] + timedelta(days=1)

        predictions.append({

            "route_name": row["route_name"],
            "travel_date": next_date,
            "predicted_filled_seats": predicted_seats

        })

    return predictions
