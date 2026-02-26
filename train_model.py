import pandas as pd
import joblib

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from db import engine
from config import SEATS_MODEL_PATH, PRICE_MODEL_PATH
from feature_engineering import create_features, FEATURE_COLUMNS_SEATS, FEATURE_COLUMNS_PRICE


print("Loading data from Neon...")

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

print("Total raw rows:", len(df))

print("Creating features...")

df = create_features(df)

print("Rows after feature engineering:", len(df))

if len(df) < 5:
    print("Not enough data to train model yet.")
    exit()

def train_and_save(X_cols, y_col, model_path, name):
    print(f"\nTraining {name} model...")
    X = df[X_cols]
    y = df[y_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"{name} Model MAE: {mae:.2f}")

    joblib.dump(model, model_path)
    print(f"{name} model saved to {model_path}")

# Train Seats Model
train_and_save(FEATURE_COLUMNS_SEATS, 'filled_seats', SEATS_MODEL_PATH, "Seats")

# Train Price Model
if 'average_price' in df.columns:
    train_and_save(FEATURE_COLUMNS_PRICE, 'average_price', PRICE_MODEL_PATH, "Price")

print("\nTravelFlux ML training complete!")
