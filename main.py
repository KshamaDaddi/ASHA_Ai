"""ASHA AI Guardian - single-file offline healthcare assistant.

Run with:
    streamlit run main.py

The deterministic triage engine is the safety authority. Ollama is used for
explanation and conversational support only; it must never override an
emergency classification.
"""

from __future__ import annotations

import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

try:
    import ollama
except ImportError:
    ollama = None

try:
    from deep_translator import GoogleTranslator
except ImportError:
    GoogleTranslator = None

try:
    import easyocr
except ImportError:
    easyocr = None

try:
    import speech_recognition as sr
except ImportError:
    sr = None

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

from PIL import Image


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DB_PATH = Path(__file__).with_name("asha_ai.db")
OLLAMA_MODEL = os.getenv("ASHA_AI_MODEL", "phi3")

LANGUAGES = {"English": "en", "Kannada": "kn", "Hindi": "hi"}

SYMPTOM_WEIGHTS = {
    "chest pain": 5,
    "breathing difficulty": 5,
    "shortness of breath": 5,
    "unconscious": 5,
    "seizure": 5,
    "heavy bleeding": 5,
    "high fever": 3,
    "vomiting": 2,
    "dehydration": 3,
    "cough": 1,
    "fever": 2,
    "headache": 1,
    "dizziness": 2,
    "weakness": 2,
}

NEGATION_PATTERNS = (
    r"\bno\s+(?:severe\s+)?{symptom}\b",
    r"\bwithout\s+(?:any\s+)?{symptom}\b",
    r"\bdo\s+not\s+have\s+(?:any\s+)?{symptom}\b",
    r"\bdon't\s+have\s+(?:any\s+)?{symptom}\b",
)

SYSTEM_PROMPT = """You are ASHA AI Guardian, an offline healthcare support assistant.

You are NOT a doctor and must not diagnose disease.
The application's deterministic triage engine is the safety authority.
Never contradict or downgrade a CRITICAL or HIGH risk result.
For emergency symptoms, advise immediate professional medical care.
Do not prescribe medicines or recommend unsafe treatment changes.
Use simple, calm language and keep responses concise.
"""


# -----------------------------------------------------------------------------
# Database
# -----------------------------------------------------------------------------
@st.cache_resource

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT NOT NULL,
            age INTEGER,
            symptoms TEXT NOT NULL,
            risk_level TEXT NOT NULL,
            risk_score INTEGER NOT NULL,
            recommendation TEXT NOT NULL,
            created_at TEXT NOT NULL
        )"""
    )
    conn.commit()
    return conn


def save_patient(
    name: str,
    age: int | None,
    symptoms: str,
    risk_level: str,
    risk_score: int,
    recommendation: str,
) -> None:
    get_connection().execute(
        """INSERT INTO patients
        (patient_name, age, symptoms, risk_level, risk_score, recommendation, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            name or "Anonymous",
            age,
            symptoms,
            risk_level,
            risk_score,
            recommendation,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ),
    )
    get_connection().commit()


def get_patients() -> list[sqlite3.Row]:
    return get_connection().execute(
        "SELECT * FROM patients ORDER BY id DESC"
    ).fetchall()


# -----------------------------------------------------------------------------
# Deterministic safety-first triage
# -----------------------------------------------------------------------------
def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().strip())


def is_negated(text: str, symptom: str) -> bool:
    escaped = re.escape(symptom)
    return any(
        re.search(pattern.format(symptom=escaped), text)
        for pattern in NEGATION_PATTERNS
    )


def triage(symptoms: str, age: int | None) -> dict[str, Any]:
    text = normalise(symptoms)
    patient_age = max(0, int(age or 0))
    score = 0
    detected: list[str] = []
    active: set[str] = set()

    for symptom, weight in SYMPTOM_WEIGHTS.items():
        if symptom in text and not is_negated(text, symptom):
            score += weight
            detected.append(symptom)
            active.add(symptom)

    if patient_age >= 60:
        score += 2
    if patient_age <= 5 and {"fever", "high fever", "dehydration"} & active:
        score += 3

    if {"chest pain", "breathing difficulty"} <= active or {
        "chest pain", "shortness of breath"
    } <= active:
        score += 5
    if {"high fever", "dehydration"} <= active:
        score += 4
    if {"vomiting", "weakness"} <= active:
        score += 2

    if score >= 10:
        level = "CRITICAL EMERGENCY"
        recommendation = "Seek emergency medical care immediately."
    elif score >= 6:
        level = "HIGH RISK"
        recommendation = "Seek urgent medical evaluation as soon as possible."
    elif score >= 3:
        level = "MODERATE RISK"
        recommendation = "Monitor symptoms closely and seek medical advice."
    else:
        level = "LOW RISK"
        recommendation = "Continue basic precautions and seek medical advice if symptoms worsen."

    return {
        "risk_level": level,
        "risk_score": score,
        "detected_symptoms": detected,
        "recommendation": recommendation,
    }


# -----------------------------------------------------------------------------
# Translation
# -----------------------------------------------------------------------------
def translate_text(text: str, source: str, target: str) -> str:
    if not text or source == target or GoogleTranslator is None:
        return text
    try:
        return GoogleTranslator(source=source, target=target).translate(text)
    except Exception:
        return text


# -----------------------------------------------------------------------------
# Ollama
# -----------------------------------------------------------------------------
def ask_ai(prompt: str, triage_result: dict[str, Any] | None = None) -> str:
    if not prompt.strip():
        return "Please describe the patient's symptoms."
    if ollama is None:
        return "Ollama is not installed. Install it and make sure the selected local model is available."

    context = ""
    if triage_result:
        context = f"""
Deterministic triage result:
Risk level: {triage_result['risk_level']}
Risk score: {triage_result['risk_score']}
Detected symptoms: {', '.join(triage_result['detected_symptoms']) or 'None'}
Recommendation: {triage_result['recommendation']}
"""

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT + context},
                {"role": "user", "content": prompt.strip()},
            ],
        )
        return response["message"]["content"].strip()
    except Exception as exc:
        return (
            f"Local AI service is unavailable. Make sure Ollama is running and "
            f"the model '{OLLAMA_MODEL}' is installed. ({exc})"
        )


# -----------------------------------------------------------------------------
# OCR / Voice
# -----------------------------------------------------------------------------
@st.cache_resource

def get_ocr_reader():
    if easyocr is None:
        return None
    return easyocr.Reader(["en"], gpu=False)


def extract_text(image: Image.Image) -> str:
    reader = get_ocr_reader()
    if reader is None:
        return "OCR is unavailable because EasyOCR is not installed."
    results = reader.readtext(np.array(image))
    return " ".join(result[1] for result in results).strip()


def listen() -> str | None:
    if sr is None:
        return None
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.info("Listening...")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=15)
        return recognizer.recognize_google(audio)
    except Exception:
        return None


def speak(text: str) -> None:
    if pyttsx3 is None:
        return
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 160)
        engine.say(text)
        engine.runAndWait()
    except Exception:
        pass


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ASHA AI Guardian",
    page_icon="🩺",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #0f172a; color: white; }
    h1, h2, h3, h4 { color: white; }
    .block-container { padding-top: 2rem; }
    .disclaimer { padding: 12px; border-radius: 10px; background: #334155; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("🩺 ASHA AI Guardian")
    st.caption("Offline healthcare support platform")
    language_name = st.selectbox("🌐 Language", list(LANGUAGES))
    st.divider()
    st.markdown(
        f"**Local model:** `{OLLAMA_MODEL}`\n\n"
        "**Safety:** deterministic triage + local LLM explanation"
    )
    st.divider()
    st.subheader("Recent patients")
    for patient in get_patients()[:5]:
        st.markdown(
            f"**{patient['patient_name']}** ({patient['age'] or 'N/A'})  \n"
            f"Risk: **{patient['risk_level']}**  \n"
            f"{patient['created_at']}"
        )

st.title("🩺 ASHA AI Guardian")
st.subheader("Offline Multilingual Healthcare Intelligence")
st.markdown(
    '<div class="disclaimer">⚠️ <b>Safety notice:</b> ASHA AI provides preliminary '
    'decision support and is not a diagnostic system or substitute for a qualified clinician.</div>',
    unsafe_allow_html=True,
)

patients = get_patients()
counts = {
    "Total": len(patients),
    "Critical": sum("CRITICAL" in p["risk_level"] for p in patients),
    "High": sum("HIGH" in p["risk_level"] for p in patients),
    "Moderate": sum("MODERATE" in p["risk_level"] for p in patients),
    "Low": sum("LOW" in p["risk_level"] for p in patients),
}

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("👥 Patients", counts["Total"])
c2.metric("🔴 Critical", counts["Critical"])
c3.metric("🟠 High", counts["High"])
c4.metric("🟡 Moderate", counts["Moderate"])
c5.metric("🟢 Low", counts["Low"])

with st.expander("📊 Risk analytics", expanded=False):
    chart = pd.DataFrame(
        {"Patients": [counts["Critical"], counts["High"], counts["Moderate"], counts["Low"]]},
        index=["Critical", "High", "Moderate", "Low"],
    )
    st.bar_chart(chart)

st.divider()
st.subheader("🧑 Patient Assessment")

c1, c2 = st.columns(2)
with c1:
    patient_name = st.text_input("Patient name", placeholder="Optional")
with c2:
    age_input = st.number_input("Patient age", min_value=0, max_value=120, value=25, step=1)

uploaded_file = st.file_uploader(
    "🖼 Upload medical report for OCR",
    type=["png", "jpg", "jpeg"],
)
report_text = ""
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded report", use_container_width=True)
    with st.spinner("Extracting text..."):
        report_text = extract_text(image)
    st.text_area("Extracted report text", report_text, height=120)

c1, c2 = st.columns([1, 3])
with c1:
    voice_clicked = st.button("🎤 Speak symptoms", use_container_width=True)
with c2:
    symptoms_input = st.text_area(
        "Symptoms",
        placeholder="Example: chest pain and breathing difficulty",
        height=120,
    )

if voice_clicked:
    voice_text = listen()
    if voice_text:
        st.session_state["voice_symptoms"] = voice_text
        st.success(f"Recognized: {voice_text}")
    else:
        st.warning("Voice input was not available. You can type the symptoms instead.")

if st.session_state.get("voice_symptoms"):
    symptoms_input = st.session_state["voice_symptoms"]
    st.caption(f"Voice symptoms: {symptoms_input}")

analyze = st.button("🔍 Analyze Patient", type="primary", use_container_width=True)

if analyze:
    if not symptoms_input.strip() and not report_text.strip():
        st.error("Please enter symptoms or upload a medical report.")
        st.stop()

    source_lang = LANGUAGES[language_name]
    english_symptoms = translate_text(symptoms_input, source_lang, "en")
    combined_symptoms = " ".join(
        part for part in [english_symptoms, report_text] if part.strip()
    )

    result = triage(combined_symptoms, int(age_input))
    save_patient(
        patient_name,
        int(age_input),
        combined_symptoms,
        result["risk_level"],
        result["risk_score"],
        result["recommendation"],
    )

    st.subheader("🚨 Triage Result")
    if result["risk_level"] == "CRITICAL EMERGENCY":
        st.error(f"🔴 {result['risk_level']}")
    elif result["risk_level"] == "HIGH RISK":
        st.warning(f"🟠 {result['risk_level']}")
    elif result["risk_level"] == "MODERATE RISK":
        st.info(f"🟡 {result['risk_level']}")
    else:
        st.success(f"🟢 {result['risk_level']}")

    m1, m2 = st.columns(2)
    m1.metric("Risk score", result["risk_score"])
    m2.metric("Detected symptoms", len(result["detected_symptoms"]))

    st.write("**Detected:**", ", ".join(result["detected_symptoms"]) or "None")
    st.write("**Recommendation:**", result["recommendation"])

    if "CRITICAL" in result["risk_level"] or "HIGH" in result["risk_level"]:
        st.error(
            "🚑 **Emergency action:** Do not rely on this application alone. "
            "Seek urgent professional medical care."
        )

    with st.spinner("Generating local AI explanation..."):
        ai_response = ask_ai(combined_symptoms, result)

    translated_response = translate_text(ai_response, "en", source_lang)
    st.subheader("🤖 ASHA AI Explanation")
    st.write(translated_response)

    if st.button("🔊 Read response aloud"):
        speak(translated_response)

st.divider()
st.caption(
    "ASHA AI Guardian • Local-first healthcare support • "
    "The deterministic triage engine is the safety authority."
)
