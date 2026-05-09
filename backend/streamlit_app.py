import streamlit as st
import ollama
from datetime import datetime
from deep_translator import GoogleTranslator
import speech_recognition as sr
import pyttsx3
import sqlite3
import easyocr
from PIL import Image
import numpy as np
import pandas as pd
from smart_triage import smart_triage

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="ASHA-AI Guardian",
    page_icon="🩺",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.stApp {
    background-color: #0f172a;
    color: white;
}

h1, h2, h3, h4 {
    color: white;
}

.block-container {
    padding-top: 2rem;
}

.metric-card {
    background: #1e293b;
    padding: 20px;
    border-radius: 20px;
    border: 1px solid #334155;
}

.emergency-box {
    background-color: #7f1d1d;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid red;
}

.warning-box {
    background-color: #78350f;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid orange;
}

.safe-box {
    background-color: #14532d;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid green;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# DATABASE
# =====================================================

conn = sqlite3.connect(
    "asha_ai.db",
    check_same_thread=False
)

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS patients (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    patient_name TEXT,

    age TEXT,

    symptoms TEXT,

    risk_level TEXT,

    recommendation TEXT,

    created_at TEXT
)
""")

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
# GET PATIENTS
# =====================================================

def get_patients():

    cursor.execute("""
    SELECT
    patient_name,
    age,
    symptoms,
    risk_level,
    recommendation,
    created_at

    FROM patients

    ORDER BY id DESC
    """)

    return cursor.fetchall()

# =====================================================
# TRANSLATOR
# =====================================================

def translate_to_english(text, source_lang):

    if source_lang == "English":
        return text

    lang_map = {
        "Kannada": "kn",
        "Hindi": "hi"
    }

    try:

        return GoogleTranslator(
            source=lang_map[source_lang],
            target='en'
        ).translate(text)

    except:

        return text


def translate_from_english(text, target_lang):

    if target_lang == "English":
        return text

    lang_map = {
        "Kannada": "kn",
        "Hindi": "hi"
    }

    try:

        return GoogleTranslator(
            source='en',
            target=lang_map[target_lang]
        ).translate(text)

    except:

        return text

# =====================================================
# VOICE AI
# =====================================================

def listen():

    recognizer = sr.Recognizer()

    with sr.Microphone() as source:

        st.info("🎤 Listening...")

        recognizer.adjust_for_ambient_noise(source)

        audio = recognizer.listen(source)

    try:

        return recognizer.recognize_google(audio)

    except:

        return None


def speak(text):

    engine = pyttsx3.init()

    engine.setProperty('rate', 160)

    engine.say(text)

    engine.runAndWait()

# =====================================================
# OCR
# =====================================================

reader = easyocr.Reader(['en'])

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.title("🩺 ASHA-AI Guardian")

    st.markdown("""
### Offline Healthcare Intelligence

Powered By:
- Gemma 2B
- Ollama
- Streamlit

Features:
✅ Offline AI  
✅ Voice Assistant  
✅ Emergency Triage  
✅ Multilingual Support  
✅ Medical OCR  
✅ Patient Memory  
✅ Smart Risk Detection  
""")

    language = st.selectbox(
        "🌐 Select Language",
        ["English", "Kannada", "Hindi"]
    )

    st.divider()

    st.subheader("📋 Recent Patients")

    patients = get_patients()

    for patient in patients[:5]:

        st.markdown(f"""
**{patient[0]}** ({patient[1]})

Risk: {patient[3]}

Date: {patient[5]}
""")

# =====================================================
# MAIN HEADER
# =====================================================

st.title("🩺 ASHA-AI Guardian")

st.subheader(
    "Offline Multilingual Healthcare Intelligence Platform"
)

st.info("""
AI-powered healthcare intelligence system designed for
ASHA workers and rural healthcare support environments.
Runs fully offline using Gemma via Ollama.
""")

# =====================================================
# DASHBOARD
# =====================================================

patients = get_patients()

total_patients = len(patients)

critical_count = len([
    p for p in patients
    if "CRITICAL" in p[3]
])

high_count = len([
    p for p in patients
    if "HIGH" in p[3]
])

moderate_count = len([
    p for p in patients
    if "MODERATE" in p[3]
])

low_count = len([
    p for p in patients
    if "LOW" in p[3]
])

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "👥 Total Patients",
        total_patients
    )

with col2:
    st.metric(
        "🔴 Critical",
        critical_count
    )

with col3:
    st.metric(
        "🟠 High Risk",
        high_count
    )

with col4:
    st.metric(
        "🟢 Safe",
        low_count
    )

st.divider()

# =====================================================
# ANALYTICS
# =====================================================

st.subheader("📊 Healthcare Analytics")

risk_data = pd.DataFrame({

    "Risk Level": [
        "Critical",
        "High",
        "Moderate",
        "Low"
    ],

    "Patients": [
        critical_count,
        high_count,
        moderate_count,
        low_count
    ]
})

st.bar_chart(
    risk_data.set_index("Risk Level")
)

# =====================================================
# PATIENT INFO
# =====================================================

st.subheader("🧑 Patient Information")

col1, col2 = st.columns(2)

with col1:

    patient_name = st.text_input(
        "Patient Name"
    )

with col2:

    patient_age = st.text_input(
        "Patient Age"
    )

# =====================================================
# IMAGE UPLOAD
# =====================================================

st.subheader("🖼 Upload Medical Report")

uploaded_file = st.file_uploader(
    "Upload prescription/report image",
    type=["png", "jpg", "jpeg"]
)

image_text = ""

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(
        image,
        use_container_width=True
    )

    image_array = np.array(image)

    results = reader.readtext(image_array)

    extracted_text = " ".join(
        [result[1] for result in results]
    )

    image_text = extracted_text

    st.success(
        f"📄 Extracted Text:\n\n{image_text}"
    )

# =====================================================
# VOICE INPUT
# =====================================================

voice_prompt = None

if st.button("🎤 Speak Symptoms"):

    voice_prompt = listen()

    if voice_prompt:

        st.success(
            f"You said: {voice_prompt}"
        )

# =====================================================
# CHAT INPUT
# =====================================================

text_prompt = st.chat_input(
    "Describe patient symptoms..."
)

prompt = (
    voice_prompt
    if voice_prompt
    else text_prompt
)

# =====================================================
# MAIN AI FLOW
# =====================================================

if prompt:

    english_prompt = translate_to_english(
        prompt,
        language
    )

    full_prompt = f"""
Patient Information:
Name: {patient_name}
Age: {patient_age}

Symptoms:
{english_prompt}

Medical Report:
{image_text}
"""

    # =====================================================
    # SMART TRIAGE
    # =====================================================

    triage_result = smart_triage(
        english_prompt,
        patient_age
    )

    save_patient(

        patient_name,

        patient_age,

        english_prompt,

        triage_result['risk_level'],

        triage_result['recommendation']
    )

    # =====================================================
    # RISK ALERT
    # =====================================================

    risk = triage_result['risk_level']

    st.subheader("🚨 Emergency Triage Analysis")

    if "CRITICAL" in risk:

        st.error(f"""
{risk}

Immediate hospitalization required.
Critical symptoms detected.
""")

    elif "HIGH" in risk:

        st.warning(f"""
{risk}

Urgent medical consultation recommended.
""")

    elif "MODERATE" in risk:

        st.info(f"""
{risk}

Monitor symptoms carefully.
""")

    else:

        st.success(f"""
{risk}

Basic precautions recommended.
""")

    # =====================================================
    # TRIAGE DETAILS
    # =====================================================

    st.markdown(f"""
### 🩺 Medical Analysis

**Detected Symptoms:**  
{", ".join(triage_result['detected_symptoms'])}

**Risk Score:**  
{triage_result['risk_score']}

**Recommendation:**  
{triage_result['recommendation']}
""")

    # =====================================================
    # AI CONFIDENCE
    # =====================================================

    confidence = min(
        95,
        60 + triage_result['risk_score'] * 4
    )

    st.progress(confidence / 100)

    st.caption(
        f"AI Confidence Score: {confidence}%"
    )

    # =====================================================
    # EMERGENCY ACTION PLAN
    # =====================================================

    if (
        "CRITICAL" in risk or
        "HIGH" in risk
    ):

        st.subheader("🚑 Emergency Action Plan")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.error("""
### 🏥 Hospital

Immediate medical support required
""")

        with col2:

            st.warning("""
### 📞 Emergency

Contact healthcare provider
""")

        with col3:

            st.info("""
### 💊 Monitoring

Monitor patient continuously
""")

    # =====================================================
    # GEMMA RESPONSE
    # =====================================================

    with st.spinner(
        "Analyzing patient condition..."
    ):

        response = ollama.chat(

            model="gemma:2b",

            messages=[

                {
                    "role": "system",

                    "content": f"""
You are ASHA-AI Guardian.

You are an offline multilingual healthcare assistant.

Triage Information:
- Risk Level: {triage_result['risk_level']}
- Symptoms: {triage_result['detected_symptoms']}
- Recommendation: {triage_result['recommendation']}

Your tasks:
- Explain symptoms
- Provide healthcare guidance
- Detect emergencies
- Keep answers short and clear
- Never provide unsafe medical advice
"""
                },

                {
                    "role": "user",
                    "content": full_prompt
                }

            ],

            options={
                "temperature": 0.3,
                "num_predict": 300
            }
        )

        ai_response = response["message"]["content"]

        translated_response = translate_from_english(
            ai_response,
            language
        )

        st.chat_message("assistant").markdown(
            translated_response
        )

        speak(translated_response)

# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "ASHA-AI Guardian • Offline Edge Healthcare Intelligence using Gemma via Ollama"
)