
from sqlalchemy import text
from db import engine

def update_schema():
    with engine.connect() as conn:
        try:
            # Check if suggested_price already exists
            result = conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name='predictions' AND column_name='suggested_price'"))
            if not result.fetchone():
                print("Adding suggested_price column to predictions table...")
                conn.execute(text("ALTER TABLE predictions ADD COLUMN suggested_price FLOAT"))
                conn.commit()
                print("Column added.")
            else:
                print("suggested_price column already exists.")
        except Exception as e:
            print("Error:", e)

if __name__ == "__main__":
    update_schema()
