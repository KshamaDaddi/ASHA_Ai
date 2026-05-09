import sqlite3
from datetime import datetime

# =====================================================
# CONNECT DATABASE
# =====================================================

conn = sqlite3.connect(
    "asha_ai.db",
    check_same_thread=False
)

cursor = conn.cursor()

# =====================================================
# CREATE TABLE
# =====================================================

cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS patients (

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        patient_name TEXT,

        age TEXT,

        symptoms TEXT,

        risk_level TEXT,

        recommendation TEXT,

        created_at TEXT
    )
"""
)

conn.commit()


# =====================================================
# SAVE PATIENT
# =====================================================

def save_patient(

    patient_name,
    age,
    symptoms,
    risk_level,
    recommendation

):

    cursor.execute(

        """
        INSERT INTO patients (

            patient_name,
            age,
            symptoms,
            risk_level,
            recommendation,
            created_at

        )

        VALUES (?, ?, ?, ?, ?, ?)
        """,

        (
            patient_name,
            age,
            symptoms,
            risk_level,
            recommendation,
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )
    )

    conn.commit()


# =====================================================
# GET ALL PATIENTS
# =====================================================

def get_patients():

    cursor.execute(

        """
        SELECT
        patient_name,
        age,
        symptoms,
        risk_level,
        recommendation,
        created_at

        FROM patients

        ORDER BY id DESC
        """
    )

    return cursor.fetchall()