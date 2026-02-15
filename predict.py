import pandas as pd
import joblib
from datetime import timedelta
from sqlalchemy import text

from db import engine
from config import MODEL_PATH
from feature_engineering import create_features, FEATURE_COLUMNS


print("Loading TravelFlux ML model...")

model = joblib.load(MODEL_PATH)


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
    scraped_at
FROM redbus_fill_rates
ORDER BY route_name, travel_date;
"""

df = pd.read_sql(query, engine)


print("Creating features...")

df = create_features(df)


print("Getting latest row per route...")

latest_rows = df.groupby("route_name").tail(1)


predictions = []


FORECAST_DAYS = 7


print(f"Generating predictions for next {FORECAST_DAYS} days...")


for _, row in latest_rows.iterrows():

    route_name = row["route_name"]

    current_features = row.copy()

    current_date = row["travel_date"]


    for day in range(1, FORECAST_DAYS + 1):

        future_date = current_date + timedelta(days=day)

        feature_values = current_features[FEATURE_COLUMNS].values.reshape(1, -1)

        predicted_seats = int(model.predict(feature_values)[0])

        predictions.append({

            "route_name": route_name,
            "travel_date": future_date,
            "predicted_filled_seats": predicted_seats

        })


        # update lag features for next iteration (recursive prediction)

        current_features["lag_1"] = predicted_seats

        if "rolling_mean_2" in current_features:
            current_features["rolling_mean_2"] = (
                current_features["rolling_mean_2"] + predicted_seats
            ) / 2


pred_df = pd.DataFrame(predictions)


print("\nGenerated Predictions:")
print(pred_df)


print("\nDeleting old predictions from Neon...")


with engine.connect() as conn:
    conn.execute(text("DELETE FROM predictions"))
    conn.commit()


print("Saving new 7-day predictions to Neon...")


pred_df.to_sql(

    "predictions",
    engine,
    if_exists="append",
    index=False

)


print("\nTravelFlux 7-day predictions saved successfully!")
