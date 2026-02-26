
import pandas as pd
from db import engine

def check_tables():
    try:
        df_fill = pd.read_sql("SELECT * FROM redbus_fill_rates LIMIT 1", engine)
        print("Columns in redbus_fill_rates:", df_fill.columns.tolist())
        
        df_pred = pd.read_sql("SELECT * FROM predictions LIMIT 1", engine)
        print("Columns in predictions:", df_pred.columns.tolist())
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    check_tables()
