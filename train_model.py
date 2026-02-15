import pandas as pd
import joblib

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from db import engine
from config import MODEL_PATH
from feature_engineering import create_features, FEATURE_COLUMNS


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
    scraped_at
FROM redbus_fill_rates
ORDER BY route_name, travel_date;
"""

df = pd.read_sql(query, engine)

print("Total raw rows:", len(df))

print("Creating features...")

df = create_features(df)

print("Rows after feature engineering:", len(df))

if len(df) < 2:
    print("Not enough data to train model yet.")
    exit()

X = df[FEATURE_COLUMNS]
y = df['filled_seats']


print("Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)


print("Training XGBoost model...")

model = XGBRegressor(

    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42

)

model.fit(X_train, y_train)


print("Evaluating model...")

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

print(f"Model MAE: {mae:.2f} seats")


print("Saving model...")

joblib.dump(model, MODEL_PATH)

print("Model saved successfully!")

print("TravelFlux ML training complete!")
