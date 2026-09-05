"""ASHA AI Guardian - local-first healthcare decision-support demo.

Run:
    streamlit run main.py

Safety architecture:
    Deterministic triage -> safety authority
    Ollama               -> explanation only

This application is for preliminary decision support, not diagnosis or treatment.
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
from PIL import Image

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

DB_PATH = Path(__file__).with_name("asha_ai.db")
OLLAMA_MODEL = os.getenv("ASHA_AI_MODEL", "phi3")
LANGUAGES = {"English": "en", "Kannada": "kn", "Hindi": "hi"}

# More specific phrases must be evaluated first. This prevents "high fever"
# from also being counted as the generic "fever" symptom.
SYMPTOM_WEIGHTS = {
    "breathing difficulty": 5,
    "shortness of breath": 5,
    "heavy bleeding": 5,
    "chest pain": 5,
    "unconscious": 5,
    "seizure": 5,
    "high fever": 3,
    "dehydration": 3,
    "vomiting": 2,
    "dizziness": 2,
    "weakness": 2,
    "fever": 2,
    "headache": 1,
    "cough": 1,
}

NEGATION_PATTERNS = (
    r"\bno\s+(?:severe\s+)?{symptom}\b",
    r"\bwithout\s+(?:any\s+)?{symptom}\b",
    r"\bdo\s+not\s+have\s+(?:any\s+)?{symptom}\b",
    r"\bdon't\s+have\s+(?:any\s+)?{symptom}\b",
)

SYSTEM_PROMPT = """You are ASHA AI Guardian, a healthcare decision-support assistant.
You are NOT a doctor and must not diagnose disease.
The deterministic triage result supplied by the application is the safety authority.
Never contradict, reduce, or downgrade a CRITICAL or HIGH risk result.
For emergency symptoms, advise immediate professional medical care.
Do not prescribe medicines or recommend unsafe treatment changes.
Explain the result in simple, calm language and keep the response concise.
"""


# ----------------------------- Database -------------------------------------
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


def save_patient(name: str, age: int, symptoms: str, result: dict[str, Any]) -> None:
    get_connection().execute(
        """INSERT INTO patients
        (patient_name, age, symptoms, risk_level, risk_score, recommendation, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            name.strip() or "Anonymous",
            age,
            symptoms,
            result["risk_level"],
            result["risk_score"],
            result["recommendation"],
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ),
    )
    get_connection().commit()


def get_patients() -> list[sqlite3.Row]:
    return get_connection().execute(
        "SELECT * FROM patients ORDER BY id DESC"
    ).fetchall()


# ----------------------------- Triage ---------------------------------------
def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().strip())


def is_negated(text: str, symptom: str) -> bool:
    escaped = re.escape(symptom)
    return any(
        re.search(pattern.format(symptom=escaped), text)
        for pattern in NEGATION_PATTERNS
    )


def contains_phrase(text: str, phrase: str) -> bool:
    return bool(re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text))


def triage(symptoms: str, age: int | None) -> dict[str, Any]:
    text = normalise(symptoms)
    patient_age = max(0, int(age or 0))
    score = 0
    detected: list[str] = []
    active: set[str] = set()

    # Longest-first + span masking prevents overlapping symptom phrases.
    occupied: list[tuple[int, int]] = []
    for symptom, weight in sorted(SYMPTOM_WEIGHTS.items(), key=lambda item: -len(item[0])):
        match = re.search(rf"(?<!\w){re.escape(symptom)}(?!\w)", text)
        if match and not is_negated(text, symptom):
            span = match.span()
            if any(span[0] < end and start < span[1] for start, end in occupied):
                continue
            occupied.append(span)
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


# ----------------------------- AI / translation -----------------------------
def translate_text(text: str, source: str, target: str) -> str:
    if not text or source == target or GoogleTranslator is None:
        return text
    try:
        return GoogleTranslator(source=source, target=target).translate(text)
    except Exception:
        return text


def ask_ai(prompt: str, triage_result: dict[str, Any] | None = None) -> str:
    if not prompt.strip():
        return "Please describe the patient's symptoms."
    if ollama is None:
        return "Ollama is not installed. Install Ollama and the selected local model."

    context = ""
    if triage_result:
        context = (
            "\nDeterministic triage result (SAFETY AUTHORITY):\n"
            f"Risk level: {triage_result['risk_level']}\n"
            f"Risk score: {triage_result['risk_score']}\n"
            f"Detected symptoms: {', '.join(triage_result['detected_symptoms']) or 'None'}\n"
            f"Recommendation: {triage_result['recommendation']}\n"
        )
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT + context},
                {"role": "user", "content": prompt.strip()},
            ],
        )
        return response["message"]["content"].strip()
    except Exception:
        return (
            f"Local AI is unavailable. Start Ollama and install '{OLLAMA_MODEL}'. "
            "The deterministic triage result remains available."
        )


# ----------------------------- OCR / voice ----------------------------------
@st.cache_resource

def get_ocr_reader():
    if easyocr is None:
        return None
    return easyocr.Reader(["en"], gpu=False)


def extract_text(image: Image.Image) -> str:
    reader = get_ocr_reader()
    if reader is None:
        return "OCR is unavailable because EasyOCR is not installed."
    try:
        results = reader.readtext(np.array(image))
        return " ".join(result[1] for result in results).strip()
    except Exception:
        return "OCR could not process this image."


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


# ----------------------------- Reports --------------------------------------
def build_report(name: str, age: int, symptoms: str, result: dict[str, Any], ai_text: str) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""ASHA AI GUARDIAN - ASSESSMENT REPORT
====================================
Generated: {timestamp}
Patient: {name.strip() or 'Anonymous'}
Age: {age}

TRIAGE RESULT
-------------
Risk level: {result['risk_level']}
Risk score: {result['risk_score']}
Detected symptoms: {', '.join(result['detected_symptoms']) or 'None'}
Recommendation: {result['recommendation']}

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


# ----------------------------- UI -------------------------------------------
st.set_page_config(page_title="ASHA AI Guardian", page_icon="🩺", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: #0b1220; }
    .block-container { max-width: 1400px; padding-top: 1.5rem; }
    .hero { padding: 1.4rem 1.6rem; border: 1px solid #26354d; border-radius: 16px;
            background: linear-gradient(135deg,#111c31,#16253d); margin-bottom: 1rem; }
    .hero h1 { margin: 0; color: #ffffff; }
    .hero p { margin: .35rem 0 0; color: #b9c7dc; }
    .notice { padding: .9rem 1rem; border-radius: 12px; background: #332b16;
              border: 1px solid #66551f; color: #f4df9a; }
    .result { padding: 1rem; border-radius: 14px; background: #121d30; border: 1px solid #2b3a52; }
    .small { color: #94a3b8; font-size: .85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("🩺 ASHA AI")
    st.caption("Local-first healthcare decision support")
    language_name = st.selectbox("🌐 Response language", list(LANGUAGES))
    st.divider()
    st.markdown(f"**Ollama model:** `{OLLAMA_MODEL}`")
    st.markdown("**Safety authority:** Deterministic triage")
    st.divider()
    st.subheader("Recent assessments")
    for patient in get_patients()[:5]:
        st.markdown(
            f"**{patient['patient_name']}** · {patient['risk_level']}  \n"
            f"<span class='small'>{patient['created_at']}</span>",
            unsafe_allow_html=True,
        )

st.markdown(
    "<div class='hero'><h1>🩺 ASHA AI Guardian</h1>"
    "<p>Multilingual preliminary triage, OCR, voice support and local LLM explanation.</p></div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='notice'>⚠️ <b>Safety notice:</b> This is a decision-support prototype, not a diagnostic system. "
    "Emergency classifications must be followed by professional medical evaluation.</div>",
    unsafe_allow_html=True,
)

patients = get_patients()
counts = {
    "Total": len(patients),
    "Critical": sum("CRITICAL" in p["risk_level"] for p in patients),
    "High": sum(p["risk_level"] == "HIGH RISK" for p in patients),
    "Moderate": sum(p["risk_level"] == "MODERATE RISK" for p in patients),
    "Low": sum(p["risk_level"] == "LOW RISK" for p in patients),
}

st.subheader("📊 Assessment Dashboard")
cards = st.columns(5)
cards[0].metric("👥 Assessments", counts["Total"])
cards[1].metric("🔴 Critical", counts["Critical"])
cards[2].metric("🟠 High", counts["High"])
cards[3].metric("🟡 Moderate", counts["Moderate"])
cards[4].metric("🟢 Low", counts["Low"])

if patients:
    with st.expander("Risk distribution", expanded=True):
        chart = pd.DataFrame(
            {"Assessments": [counts["Critical"], counts["High"], counts["Moderate"], counts["Low"]]},
            index=["Critical", "High", "Moderate", "Low"],
        )
        st.bar_chart(chart)

st.divider()
st.subheader("👤 New Patient Assessment")

c1, c2 = st.columns(2)
with c1:
    patient_name = st.text_input("Patient name", placeholder="Optional")
with c2:
    age_input = st.number_input("Patient age", min_value=0, max_value=120, value=25, step=1)

st.markdown("**Symptoms / clinical notes**")
quick = st.pills(
    "Quick symptom selection",
    ["Chest pain", "Breathing difficulty", "High fever", "Vomiting", "Dehydration", "Headache", "Cough", "Dizziness"],
    selection_mode="multi",
)
quick_text = ", ".join(quick) if quick else ""

uploaded_file = st.file_uploader("🖼 Upload report image for OCR", type=["png", "jpg", "jpeg"])
report_text = ""
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded report", use_container_width=True)
    with st.spinner("Extracting text from report..."):
        report_text = extract_text(image)
    if report_text:
        st.text_area("OCR result", report_text, height=100)

voice_col, text_col = st.columns([1, 3])
with voice_col:
    voice_clicked = st.button("🎤 Voice input", use_container_width=True)
with text_col:
    typed_symptoms = st.text_area(
        "Describe symptoms",
        placeholder="Example: chest pain and breathing difficulty for 20 minutes",
        height=110,
    )

if voice_clicked:
    voice_text = listen()
    if voice_text:
        st.session_state["voice_symptoms"] = voice_text
        st.success(f"Recognized: {voice_text}")
    else:
        st.warning("Voice input was unavailable. Please type the symptoms.")

voice_text = st.session_state.get("voice_symptoms", "")
parts = [quick_text, typed_symptoms, voice_text, report_text]
combined_input = " ".join(dict.fromkeys(part.strip() for part in parts if part.strip()))
if combined_input:
    st.caption(f"Assessment input: {combined_input}")

analyze = st.button("🔍 Analyze Patient", type="primary", use_container_width=True)

if analyze:
    if not combined_input:
        st.error("Enter symptoms, select a symptom, or upload a report.")
        st.stop()

    source_lang = LANGUAGES[language_name]
    english_input = translate_text(combined_input, source_lang, "en")
    result = triage(english_input, int(age_input))
    save_patient(patient_name, int(age_input), english_input, result)
    st.session_state["last_result"] = result
    st.session_state["last_input"] = english_input
    st.session_state["last_name"] = patient_name.strip() or "Anonymous"
    st.session_state["last_age"] = int(age_input)

    st.rerun()

# Persist the result across Streamlit reruns so the report/download controls work.
if st.session_state.get("last_result"):
    result = st.session_state["last_result"]
    assessment_input = st.session_state.get("last_input", "")
    saved_name = st.session_state.get("last_name", "Anonymous")
    saved_age = st.session_state.get("last_age", 0)

    st.divider()
    st.subheader("🚨 Latest Triage Result")

    if result["risk_level"] == "CRITICAL EMERGENCY":
        st.error("🔴 CRITICAL EMERGENCY")
    elif result["risk_level"] == "HIGH RISK":
        st.warning("🟠 HIGH RISK")
    elif result["risk_level"] == "MODERATE RISK":
        st.info("🟡 MODERATE RISK")
    else:
        st.success("🟢 LOW RISK")

    m1, m2, m3 = st.columns(3)
    m1.metric("Risk score", result["risk_score"])
    m2.metric("Detected symptoms", len(result["detected_symptoms"]))
    m3.metric("Patient age", saved_age)

    left, right = st.columns(2)
    with left:
        st.markdown("### Detected symptoms")
        if result["detected_symptoms"]:
            for symptom in result["detected_symptoms"]:
                st.markdown(f"- **{symptom.title()}**")
        else:
            st.write("No supported symptom keywords detected.")
    with right:
        st.markdown("### Recommendation")
        st.write(result["recommendation"])

    if result["risk_level"] in {"CRITICAL EMERGENCY", "HIGH RISK"}:
        st.error(
            "🚑 **Urgent action:** Do not rely on this application alone. "
            "Seek professional medical care immediately/as soon as possible."
        )

    with st.spinner("Generating local AI explanation..."):
        ai_response = ask_ai(assessment_input, result)
    display_response = translate_text(ai_response, "en", LANGUAGES[language_name])

    st.markdown("### 🤖 AI Explanation")
    st.markdown(f"<div class='result'>{display_response}</div>", unsafe_allow_html=True)

    b1, b2 = st.columns(2)
    with b1:
        if st.button("🔊 Read explanation aloud", use_container_width=True):
            speak(display_response)
    with b2:
        report = build_report(saved_name, saved_age, assessment_input, result, display_response)
        st.download_button(
            "📥 Download assessment report",
            data=report,
            file_name=f"asha_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True,
        )

st.divider()
st.subheader("📋 Patient Assessment History")

if patients:
    history_df = pd.DataFrame([dict(row) for row in patients])
    history_df = history_df.rename(
        columns={
            "patient_name": "Patient",
            "age": "Age",
            "risk_level": "Risk",
            "risk_score": "Score",
            "created_at": "Created",
            "recommendation": "Recommendation",
        }
    )
    st.dataframe(
        history_df[["Patient", "Age", "Risk", "Score", "Created", "Recommendation"]],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No assessments yet. Complete your first patient assessment above.")

st.caption(
    "ASHA AI Guardian • Local-first prototype • Deterministic triage is the safety authority. "
    "Translation and Google speech recognition may require internet access."
)
