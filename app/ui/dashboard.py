"""Streamlit dashboard: presentation and orchestration only."""
import pandas as pd
import streamlit as st
from PIL import Image

from app.database.database import get_patients, save_patient
from app.services.ocr_service import extract_text
from app.services.ollama_service import OLLAMA_MODEL, ask_ai
from app.services.report_service import build_report
from app.services.triage_engine import triage
from app.services.translation_service import translate_text
from app.services.voice_service import listen, speak

LANGUAGES = {"English": "en", "Kannada": "kn", "Hindi": "hi"}

def run_app() -> None:
    st.set_page_config(page_title="ASHA AI Guardian", page_icon="🩺", layout="wide")
    st.markdown("""<style>
    .stApp{background:#0b1220}.block-container{max-width:1400px;padding-top:1.5rem}
    .hero{padding:1.4rem 1.6rem;border:1px solid #26354d;border-radius:16px;background:#111c31;margin-bottom:1rem}
    .hero h1{margin:0;color:#fff}.hero p{margin:.35rem 0;color:#b9c7dc}
    .notice{padding:.9rem 1rem;border-radius:12px;background:#332b16;border:1px solid #66551f;color:#f4df9a}
    </style>""", unsafe_allow_html=True)

    with st.sidebar:
        st.title("🩺 ASHA AI")
        st.caption("Local-first healthcare decision support")
        input_language_name = st.selectbox("🌐 Input language", list(LANGUAGES))
        response_language_name = st.selectbox("🌐 Response language", list(LANGUAGES))
        st.divider()
        st.markdown(f"**Ollama model:** `{OLLAMA_MODEL}`")
        st.markdown("**Safety authority:** Deterministic triage")
        st.divider()
        st.subheader("Recent assessments")
        for patient in get_patients()[:5]:
            st.markdown(f"**{patient['patient_name']}** · {patient['risk_level']}  \n{patient['created_at']}")

    st.markdown("<div class='hero'><h1>🩺 ASHA AI Guardian</h1><p>Multilingual preliminary triage, OCR, voice support and local LLM explanation.</p></div>", unsafe_allow_html=True)
    st.markdown("<div class='notice'>⚠️ <b>Safety notice:</b> This is a decision-support prototype, not a diagnostic system. Emergency classifications require professional medical evaluation.</div>", unsafe_allow_html=True)

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
    for col, label, key in zip(cards, ["👥 Assessments", "🔴 Critical", "🟠 High", "🟡 Moderate", "🟢 Low"], ["Total", "Critical", "High", "Moderate", "Low"]):
        col.metric(label, counts[key])
    if patients:
        chart = pd.DataFrame({"Assessments": [counts["Critical"], counts["High"], counts["Moderate"], counts["Low"]]}, index=["Critical", "High", "Moderate", "Low"])
        with st.expander("Risk distribution", expanded=True):
            st.bar_chart(chart)

    st.divider()
    st.subheader("👤 New Patient Assessment")
    c1, c2 = st.columns(2)
    with c1: patient_name = st.text_input("Patient name", placeholder="Optional")
    with c2: age = st.number_input("Patient age", min_value=0, max_value=120, value=25, step=1)

    quick = st.pills("Quick symptom selection", ["Chest pain", "Breathing difficulty", "High fever", "Vomiting", "Dehydration", "Headache", "Cough", "Dizziness"], selection_mode="multi")
    typed = st.text_area("Symptoms / clinical notes", height=130, placeholder="Describe symptoms in the selected input language...")
    voice_col, _ = st.columns([1, 3])
    voice_text = ""
    with voice_col:
        if st.button("🎙 Speak symptoms"):
            voice_text, error = listen(LANGUAGES[input_language_name])
            if error: st.warning(error)
            elif voice_text: st.session_state["voice_text"] = voice_text
    voice_text = st.session_state.get("voice_text", voice_text)
    if voice_text: st.info(f"Voice input: {voice_text}")

    report_text = ""
    uploaded = st.file_uploader("🖼 Upload report image for OCR", type=["png", "jpg", "jpeg"])
    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded report", use_container_width=True)
        with st.spinner("Extracting text from report..."): report_text = extract_text(image)
        st.text_area("OCR result (English)", report_text, height=100)

    if st.button("🔎 Analyze patient", type="primary", use_container_width=True):
        raw_user_text = ", ".join(quick or []) + " " + typed + " " + voice_text
        english_user_text = translate_text(raw_user_text, LANGUAGES[input_language_name], "en")
        combined = " ".join(part for part in [english_user_text, report_text] if part.strip()).strip()
        if not combined:
            st.warning("Please enter symptoms, select a symptom, speak, or upload a report.")
            return
        result = triage(combined, int(age))
        ai_text = ask_ai(combined, result)
        display_ai = translate_text(ai_text, "en", LANGUAGES[response_language_name])
        display_recommendation = translate_text(result.recommendation, "en", LANGUAGES[response_language_name])
        st.session_state["latest"] = {"result": result, "ai": display_ai, "recommendation": display_recommendation, "symptoms": combined, "name": patient_name, "age": int(age)}
        save_patient(patient_name, int(age), combined, result)

    latest = st.session_state.get("latest")
    if latest:
        result = latest["result"]
        st.divider(); st.subheader("🧠 Assessment Result")
        st.metric("Risk level", result.risk_level)
        st.metric("Risk score", result.risk_score)
        if result.risk_level == "CRITICAL EMERGENCY": st.error("🚨 " + latest["recommendation"])
        elif result.risk_level == "HIGH RISK": st.warning("⚠️ " + latest["recommendation"])
        else: st.info("ℹ️ " + latest["recommendation"])
        st.write("**Detected symptoms:**", ", ".join(result.detected_symptoms) or "None")
        st.subheader("🤖 AI Explanation")
        st.write(latest["ai"])
        if st.button("🔊 Read explanation aloud"):
            ok, error = speak(latest["ai"])
            if not ok: st.warning(error)
        report = build_report(latest["name"], latest["age"], latest["symptoms"], result, latest["ai"])
        st.download_button("📄 Download assessment report", report, file_name="asha_ai_assessment.txt", mime="text/plain")

    st.divider()
    st.caption("Local-first prototype. Triage is rule-based and not clinically validated. OCR is English-only; translation and Google speech recognition may require internet access. Never use this application as a substitute for professional medical care.")
