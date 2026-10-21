
import os
import psycopg
from dotenv import load_dotenv

# Load database settings from the .env file
load_dotenv()

try:
    with psycopg.connect(
        host=os.getenv("PGHOST"),
        port=os.getenv("PGPORT"),
        dbname=os.getenv("PGDATABASE"),
        user=os.getenv("PGUSER"),
        password=os.getenv("PGPASSWORD"),
        connect_timeout=5,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT current_user, current_database();")
            user, database = cur.fetchone()

            print("Connection successful!")
            print("User:", user)
            print("Database:", database)

except Exception as error:
    print("Connection failed.")
    print("Error:", error)
