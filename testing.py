import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT")),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD")
}

try:
    conn = psycopg2.connect(**DB_CONFIG)
    print("Connection berhasil")
    conn.close()
except Exception as e:
    print(f"Koneksi Gagal: {e}")