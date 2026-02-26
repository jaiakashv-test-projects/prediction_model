import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL")

SEATS_MODEL_PATH = "model/travelflux_seats_xgb.pkl"
PRICE_MODEL_PATH = "model/travelflux_price_xgb.pkl"

# Keep for backward compatibility if needed, but we'll transition
MODEL_PATH = SEATS_MODEL_PATH
