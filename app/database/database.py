"""SQLite persistence for local assessment history."""
from datetime import datetime
from pathlib import Path
import sqlite3

DB_PATH = Path(__file__).resolve().parents[2] / "asha_ai.db"

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_name TEXT NOT NULL,
        age INTEGER,
        symptoms TEXT NOT NULL,
        risk_level TEXT NOT NULL,
        risk_score INTEGER NOT NULL,
        recommendation TEXT NOT NULL,
        created_at TEXT NOT NULL
    )""")
    conn.commit()
    return conn

def save_patient(name: str, age: int, symptoms: str, result) -> None:
    conn = get_connection()
    try:
        conn.execute("""INSERT INTO patients
        (patient_name, age, symptoms, risk_level, risk_score, recommendation, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)""", (
            name.strip() or "Anonymous", age, symptoms,
            result.risk_level, result.risk_score, result.recommendation,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ))
        conn.commit()
    finally:
        conn.close()

def get_patients():
    conn = get_connection()
    try:
        return conn.execute("SELECT * FROM patients ORDER BY id DESC").fetchall()
    finally:
        conn.close()
