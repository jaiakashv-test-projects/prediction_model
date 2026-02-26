
from sqlalchemy import text
from db import engine

def check_insights_schema():
    with engine.connect() as conn:
        res = conn.execute(text("SELECT column_name, data_type FROM information_schema.columns WHERE table_name='insights'"))
        for row in res:
            print(f"Column: {row[0]}, Type: {row[1]}")

if __name__ == "__main__":
    check_insights_schema()
