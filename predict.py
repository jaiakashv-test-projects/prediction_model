import pandas as pd
import joblib
import os
from datetime import timedelta, date
from sqlalchemy import text

from db import engine
from config import SEATS_MODEL_PATH, PRICE_MODEL_PATH
from feature_engineering import create_features, FEATURE_COLUMNS_SEATS, FEATURE_COLUMNS_PRICE


print("Loading TravelFlux ML models...")

# Load models with fallback
try:
    seats_model = joblib.load(SEATS_MODEL_PATH)
    print(f"Loaded seats model from {SEATS_MODEL_PATH}")
except:
    # Fallback to old path if new one doesn't exist yet
    old_path = "model/travelflux_xgb.pkl"
    if os.path.exists(old_path):
        seats_model = joblib.load(old_path)
        print(f"Loaded seats model from {old_path} (fallback)")
    else:
        print("Error: No seats model found!")
        exit()

price_model = None
if os.path.exists(PRICE_MODEL_PATH):
    price_model = joblib.load(PRICE_MODEL_PATH)
    print(f"Loaded price model from {PRICE_MODEL_PATH}")
else:
    print("Warning: No price model found, will use heuristic.")


print("Loading historical data from Neon...")

query = """
SELECT
    route_name,
    travel_date,
    bus_count,
    total_capacity,
    available_seats,
    filled_seats,
    fill_rate_percent,
    average_price,
    scraped_at
FROM redbus_fill_rates
ORDER BY route_name, travel_date;
"""

df = pd.read_sql(query, engine)

print("Creating features...")
df = create_features(df)

print("Getting latest row per route for anchor...")
latest_rows = df.groupby("route_name").tail(1)

predictions = []
FORECAST_DAYS = 7

# Anchor date should be today to avoid the "predicts after 3 days" issue
today = date.today()
print(f"Generating predictions starting from tomorrow ({today + timedelta(days=1)}) for next {FORECAST_DAYS} days...")

for _, row in latest_rows.iterrows():
    route_name = row["route_name"]
    current_features = row.copy()
    
    # Initialize lag values from the latest real data
    last_seats = row['filled_seats']
    last_price = row.get('average_price', 1000) # Default if missing
    
    for day in range(1, FORECAST_DAYS + 1):
        future_date = today + timedelta(days=day)
        
        # Update temporal features for the future date
        current_features['day_of_week'] = future_date.weekday()
        current_features['day_of_month'] = future_date.day
        current_features['month'] = future_date.month
        current_features['is_weekend'] = 1 if future_date.weekday() in [5, 6] else 0
        
        # Update lag features (recursive)
        current_features['lag_1_seats'] = last_seats
        # Rolling mean is harder recursive, so we'll just use lag_1 or update it simply
        if 'rolling_mean_2_seats' in current_features:
            current_features['rolling_mean_2_seats'] = (current_features['rolling_mean_2_seats'] + last_seats) / 2
            
        current_features['lag_1_price'] = last_price
        if 'rolling_mean_2_price' in current_features:
            current_features['rolling_mean_2_price'] = (current_features['rolling_mean_2_price'] + last_price) / 2

        # Predict seats
        feat_seats = current_features[FEATURE_COLUMNS_SEATS].values.reshape(1, -1)
        predicted_seats = int(seats_model.predict(feat_seats)[0])
        
        # Predict price
        if price_model:
            feat_price = current_features[FEATURE_COLUMNS_PRICE].values.reshape(1, -1)
            suggested_price = float(price_model.predict(feat_price)[0])
        else:
            # Heuristic: base price + surge
            fill_rate = (predicted_seats / current_features['total_capacity']) if current_features['total_capacity'] > 0 else 0.5
            base_price = last_price
            surge = 1.2 if fill_rate > 0.7 else (1.1 if fill_rate > 0.4 else 1.0)
            suggested_price = base_price * surge

        predictions.append({
            "route_name": route_name,
            "travel_date": future_date,
            "predicted_filled_seats": predicted_seats,
            "suggested_price": round(suggested_price, 2)
        })

        # Update lags for next day
        last_seats = predicted_seats
        last_price = suggested_price

pred_df = pd.DataFrame(predictions)

print("\nGenerated Predictions:")
print(pred_df)

print("\nDeleting legacy predictions from Neon...")
with engine.connect() as conn:
    conn.execute(text("DELETE FROM predictions"))
    conn.commit()

print(f"Saving new {FORECAST_DAYS}-day predictions to Neon...")
pred_df.to_sql(
    "predictions",
    engine,
    if_exists="append",
    index=False
)

print("\nTravelFlux predictions updated successfully!")
