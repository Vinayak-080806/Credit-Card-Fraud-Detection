import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "fraud_detection.db"

def get_connection():
    return sqlite3.connect(DB_PATH)

def initialize_database():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_amount REAL,
                prediction INTEGER,
                fraud_probability REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

def save_prediction(amount, prediction, probability):
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO prediction_history
            (transaction_amount, prediction, fraud_probability)
            VALUES (?, ?, ?)
            """,
            (float(amount), int(prediction), float(probability))
        )

def get_prediction_history(limit=100):
    with get_connection() as connection:
        cursor = connection.execute(
            """
            SELECT id, transaction_amount, prediction,
                   fraud_probability, created_at
            FROM prediction_history
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),)
        )
        return cursor.fetchall()
