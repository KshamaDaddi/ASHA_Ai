"""Assessment report generation."""
from datetime import datetime

def build_report(name: str, age: int, symptoms: str, result, ai_text: str) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""ASHA AI GUARDIAN - ASSESSMENT REPORT
====================================
Generated: {timestamp}
Patient: {name.strip() or 'Anonymous'}
Age: {age}

TRIAGE RESULT
-------------
Risk level: {result.risk_level}
Risk score: {result.risk_score}
Detected symptoms: {', '.join(result.detected_symptoms) or 'None'}
Recommendation: {result.recommendation}

SYMPTOMS / REPORT TEXT
----------------------
{symptoms}

AI EXPLANATION
--------------
{ai_text}

SAFETY NOTICE
-------------
This report is for preliminary decision support only. It is not a medical
 diagnosis and does not replace evaluation by a qualified healthcare professional.
"""
