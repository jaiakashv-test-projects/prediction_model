import pandas as pd
import joblib
from datetime import timedelta
from sqlalchemy import text

from db import engine
from config import MODEL_PATH
from feature_engineering import create_features, FEATURE_COLUMNS


print("Loading TravelFlux ML model...")

model = joblib.load(MODEL_PATH)


print("Loading latest data from Neon...")

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


print("Generating new predictions...")

for _, row in latest_rows.iterrows():

    features = row[FEATURE_COLUMNS].values.reshape(1, -1)

    predicted_seats = int(model.predict(features)[0])

    next_date = row['travel_date'] + timedelta(days=1)

    predictions.append({

        "route_name": row["route_name"],
        "travel_date": next_date,
        "predicted_filled_seats": predicted_seats

    })


pred_df = pd.DataFrame(predictions)


print("\nNew predictions:")
print(pred_df)


# -------------------------
# DELETE OLD PREDICTIONS
# -------------------------

print("\nDeleting old predictions...")

with engine.connect() as conn:
    conn.execute(text("DELETE FROM predictions"))
    conn.commit()


# -------------------------
# INSERT NEW PREDICTIONS
# -------------------------

print("Saving new predictions to Neon...")

pred_df.to_sql(

    "predictions",
    engine,
    if_exists="append",
    index=False

)


print("\nTravelFlux predictions updated successfully!")
