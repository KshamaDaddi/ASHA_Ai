"""Streamlit presentation layer for ASHA AI Guardian."""

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
QUICK_SYMPTOMS = [
    "Chest pain",
    "Breathing difficulty",
    "High fever",
    "Vomiting",
    "Dehydration",
    "Headache",
    "Cough",
    "Dizziness",
]


def _inject_css(dark_mode: bool) -> None:
    """Inject the SaaS-style visual system without changing application logic."""
    if dark_mode:
        bg = "#0b1220"
        surface = "#111c31"
        surface2 = "#162238"
        border = "#26354d"
        text = "#f7f9fc"
        muted = "#aebbd0"
        input_bg = "#0f192b"
        shadow = "0 12px 35px rgba(0,0,0,.24)"
    else:
        bg = "#f6f8fc"
        surface = "#ffffff"
        surface2 = "#f8fafc"
        border = "#e4e9f1"
        text = "#14213d"
        muted = "#667085"
        input_bg = "#ffffff"
        shadow = "0 10px 30px rgba(20,33,61,.07)"

    st.markdown(
        f"""
        <style>
        :root {{
            --bg: {bg}; --surface: {surface}; --surface2: {surface2};
            --border: {border}; --text: {text}; --muted: {muted};
            --input: {input_bg}; --shadow: {shadow};
            --teal: #12b8a6; --teal-dark: #079f92;
            --red: #ef4444; --orange: #f97316; --yellow: #eab308; --green: #22c55e;
        }}
        .stApp {{ background: var(--bg); color: var(--text); }}
        .block-container {{ max-width: 1500px; padding: 1.25rem 2rem 2rem; }}
        [data-testid="stHeader"] {{ background: transparent; }}
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg,#101b31 0%,#0b1529 100%);
            border-right: 1px solid #22324b;
        }}
        [data-testid="stSidebar"] * {{ color: #eaf0f8; }}
        [data-testid="stSidebar"] .stButton button {{
            background: transparent; border: 0; color: #dbe5f2;
            text-align: left; border-radius: 10px; min-height: 42px;
        }}
        [data-testid="stSidebar"] .stButton button:hover {{ background: rgba(18,184,166,.13); }}
        .brand {{ padding: .35rem .25rem 1.25rem; }}
        .brand-row {{ display:flex; align-items:center; gap:.7rem; }}
        .brand-icon {{
            width:48px;height:48px;border-radius:15px;display:flex;align-items:center;justify-content:center;
            background:linear-gradient(135deg,#22c7b5,#0ea5a0);font-size:25px;
            box-shadow:0 8px 22px rgba(18,184,166,.25);
        }}
        .brand-title {{ font-size:1.32rem;font-weight:800;line-height:1.05;color:#fff; }}
        .brand-title span {{ color:#20c7b5; }}
        .brand-sub {{ color:#93a5be;font-size:.78rem;margin-top:.8rem;line-height:1.45; }}
        .nav-label {{ color:#72839d;text-transform:uppercase;letter-spacing:.09em;font-size:.68rem;font-weight:700;margin:1rem 0 .4rem; }}
        .system-box {{ border:1px solid #3a4c67;border-radius:12px;padding:.85rem;margin-top:1.2rem;background:rgba(255,255,255,.025); }}
        .system-title {{ font-weight:700;font-size:.82rem;margin-bottom:.55rem; }}
        .system-line {{ display:flex;justify-content:space-between;align-items:center;font-size:.72rem;color:#b8c5d7;padding:.18rem 0; }}
        .dot {{ width:8px;height:8px;border-radius:50%;background:#22c55e;display:inline-block;margin-right:6px;box-shadow:0 0 0 3px rgba(34,197,94,.12); }}
        .user-box {{ border-top:1px solid #25354e;margin-top:1.1rem;padding-top:.85rem;display:flex;align-items:center;gap:.65rem; }}
        .avatar {{ width:34px;height:34px;border-radius:50%;background:linear-gradient(135deg,#f472b6,#c084fc);display:flex;align-items:center;justify-content:center;font-weight:800;color:white;font-size:.78rem; }}
        .topbar {{ display:flex;align-items:center;justify-content:space-between;margin-bottom:1rem; }}
        .page-title {{ font-size:1.55rem;font-weight:800;color:var(--text);margin:0; }}
        .page-subtitle {{ color:var(--muted);font-size:.86rem;margin-top:.2rem; }}
        .online {{ border:1px solid #d7eee7;background:#f0fbf7;color:#145c4b;border-radius:10px;padding:.55rem .8rem;font-size:.78rem;font-weight:700; }}
        .hero-card {{ background:var(--surface);border:1px solid var(--border);border-radius:15px;padding:1rem 1.1rem;box-shadow:var(--shadow); }}
        .safety {{ background:#fff8e8;border:1px solid #f0d99b;color:#725815;border-radius:12px;padding:.72rem .9rem;font-size:.78rem;margin-bottom:1rem; }}
        .section-title {{ font-size:1rem;font-weight:800;color:var(--text);margin:0 0 .7rem; }}
        .section-caption {{ color:var(--muted);font-size:.75rem;margin:-.4rem 0 .8rem; }}
        .metric-card {{ background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:.95rem;box-shadow:var(--shadow);min-height:105px; }}
        .metric-label {{ color:var(--muted);font-size:.72rem;font-weight:700; }}
        .metric-value {{ color:var(--text);font-size:1.55rem;font-weight:800;margin-top:.3rem; }}
        .metric-icon {{ float:right;font-size:1.1rem; }}
        .input-card {{ background:var(--surface);border:1px solid var(--border);border-radius:15px;padding:1.05rem;box-shadow:var(--shadow);height:100%; }}
        .mode-card {{ background:var(--surface2);border:1px solid var(--border);border-radius:12px;padding:1rem .75rem;text-align:center;min-height:110px; }}
        .mode-icon {{ font-size:1.7rem;margin-bottom:.35rem; }}
        .mode-title {{ font-weight:800;color:var(--text);font-size:.84rem; }}
        .mode-text {{ color:var(--muted);font-size:.7rem;margin-top:.18rem; }}
        .result-card {{ background:var(--surface);border:1px solid var(--border);border-radius:15px;padding:1rem 1.1rem;box-shadow:var(--shadow);height:100%; }}
        .risk-pill {{ display:inline-flex;align-items:center;gap:.45rem;padding:.7rem .9rem;border-radius:11px;font-size:1.02rem;font-weight:850;width:100%;box-sizing:border-box; }}
        .risk-critical {{ background:#fff0f0;border:1px solid #fecaca;color:#dc2626; }}
        .risk-high {{ background:#fff5eb;border:1px solid #fed7aa;color:#ea580c; }}
        .risk-moderate {{ background:#fffbea;border:1px solid #fde68a;color:#a16207; }}
        .risk-low {{ background:#effcf4;border:1px solid #bbf7d0;color:#15803d; }}
        .score {{ font-size:1.8rem;font-weight:850;color:var(--text);margin:.8rem 0 .25rem; }}
        .score span {{ color:var(--muted);font-size:.75rem;font-weight:600; }}
        .score-track {{ height:8px;border-radius:99px;background:linear-gradient(90deg,#65c74f 0%,#e7d63d 38%,#f59e0b 65%,#ef4444 100%);position:relative;margin:.65rem 0 .25rem; }}
        .score-marker {{ position:absolute;top:50%;width:15px;height:15px;border-radius:50%;background:#fff;border:3px solid #ef4444;transform:translate(-50%,-50%);box-shadow:0 1px 4px rgba(0,0,0,.2); }}
        .scale {{ display:flex;justify-content:space-between;color:var(--muted);font-size:.62rem; }}
        .symptom-tag {{ display:inline-block;background:#fff0f0;color:#c2413a;border:1px solid #ffd4d4;border-radius:7px;padding:.38rem .55rem;font-size:.7rem;font-weight:700;margin:.18rem .18rem .18rem 0; }}
        .rec-card {{ background:#effbf7;border:1px solid #ccefe3;border-radius:12px;padding:.85rem;color:#164e42; }}
        .ai-card {{ background:var(--surface);border:1px solid var(--border);border-radius:15px;padding:1rem 1.1rem;box-shadow:var(--shadow); }}
        .ai-label {{ font-weight:800;color:var(--text);font-size:.9rem; }}
        .summary-row {{ display:flex;justify-content:space-between;border-bottom:1px solid var(--border);padding:.5rem 0;font-size:.72rem; }}
        .summary-row:last-child {{ border-bottom:0; }}
        .summary-key {{ color:var(--muted); }} .summary-value {{ color:var(--text);font-weight:700;text-align:right; }}
        .footer-note {{ background:#eef5ff;border:1px solid #d8e7fb;color:#244f85;border-radius:10px;padding:.7rem .85rem;font-size:.72rem;line-height:1.45; }}
        .stTextInput input,.stNumberInput input,.stTextArea textarea,.stSelectbox [data-baseweb="select"] > div {{
            background:var(--input);color:var(--text);border-color:var(--border);border-radius:9px;
        }}
        .stButton button {{ border-radius:9px;border:1px solid var(--border);font-weight:700; }}
        .stButton button[kind="primary"] {{ background:linear-gradient(135deg,#15b9a7,#08a397);border:0;color:#fff;box-shadow:0 6px 15px rgba(18,184,166,.2); }}
        .stDownloadButton button {{ border-radius:9px;color:#079f92;border:1px solid #9adfd4;background:var(--surface);font-weight:700; }}
        div[data-testid="stPills"] button {{ border-radius:8px !important; }}
        [data-testid="stMetric"] {{ background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:.7rem; }}
        .small-muted {{ color:var(--muted);font-size:.7rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _risk_class(level: str) -> str:
    if "CRITICAL" in level:
        return "risk-critical"
    if "HIGH" in level:
        return "risk-high"
    if "MODERATE" in level:
        return "risk-moderate"
    return "risk-low"


def _risk_icon(level: str) -> str:
    if "CRITICAL" in level:
        return "🚨"
    if "HIGH" in level:
        return "⚠️"
    if "MODERATE" in level:
        return "◐"
    return "✓"


def _render_sidebar(dark_mode: bool) -> tuple[str, str]:
    with st.sidebar:
        st.markdown(
            """
            <div class="brand">
              <div class="brand-row">
                <div class="brand-icon">✚</div>
                <div class="brand-title">ASHA AI<br><span>Guardian</span></div>
              </div>
              <div class="brand-sub">AI-Powered Triage<br>for Better Care</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div class='nav-label'>Workspace</div>", unsafe_allow_html=True)
        page = st.radio(
            "Navigation",
            ["Dashboard", "New Assessment", "History", "Patients", "Reports", "Analytics", "Settings"],
            label_visibility="collapsed",
        )

        st.markdown("<div class='nav-label'>System</div>", unsafe_allow_html=True)
        input_language_name = st.selectbox("Input language", list(LANGUAGES), index=0)
        response_language_name = st.selectbox("Response language", list(LANGUAGES), index=0)

        st.markdown(
            f"""
            <div class="system-box">
              <div class="system-title">System Status</div>
              <div style="font-size:.72rem;margin-bottom:.5rem;"><span class="dot"></span>All Systems Online</div>
              <div class="system-line"><span>Triage Engine</span><span>✓</span></div>
              <div class="system-line"><span>Ollama ({OLLAMA_MODEL})</span><span>✓</span></div>
              <div class="system-line"><span>OCR</span><span>✓</span></div>
              <div class="system-line"><span>Voice</span><span>✓</span></div>
              <div class="system-line"><span>Translation</span><span>✓</span></div>
              <div class="system-line"><span>Database</span><span>✓</span></div>
            </div>
            <div class="user-box">
              <div class="avatar">KD</div>
              <div><b style="font-size:.76rem;">Kshama D</b><br><span style="font-size:.65rem;color:#8ea0b9;">Administrator</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.toggle("Dark interface", value=dark_mode, key="dark_mode_toggle")

    return page, input_language_name, response_language_name


def _render_header(page: str, dark_mode: bool) -> None:
    subtitle = {
        "Dashboard": "Monitor assessments and patient risk distribution",
        "New Assessment": "Provide patient details and symptoms to analyze risk",
        "History": "Review recent preliminary assessments",
        "Patients": "Browse patient assessment history",
        "Reports": "Generate and download assessment reports",
        "Analytics": "Understand risk trends across assessments",
        "Settings": "Configure language and local AI preferences",
    }[page]
    st.markdown(
        f"""
        <div class="topbar">
          <div>
            <div class="page-title">☰ &nbsp;{page}</div>
            <div class="page-subtitle">{subtitle}</div>
          </div>
          <div class="online"><span class="dot"></span>System Online</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_overview(patients: list[dict]) -> None:
    counts = {
        "Total": len(patients),
        "Critical": sum("CRITICAL" in p["risk_level"] for p in patients),
        "High": sum(p["risk_level"] == "HIGH RISK" for p in patients),
        "Moderate": sum(p["risk_level"] == "MODERATE RISK" for p in patients),
        "Low": sum(p["risk_level"] == "LOW RISK" for p in patients),
    }

    st.markdown("<div class='section-title'>Assessment Overview</div>", unsafe_allow_html=True)
    cols = st.columns(5)
    cards = [
        ("👥", "Total Assessments", counts["Total"]),
        ("🔴", "Critical", counts["Critical"]),
        ("🟠", "High Risk", counts["High"]),
        ("🟡", "Moderate", counts["Moderate"]),
        ("🟢", "Low Risk", counts["Low"]),
    ]
    for col, (icon, label, value) in zip(cols, cards):
        with col:
            st.markdown(
                f"<div class='metric-card'><span class='metric-icon'>{icon}</span><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
    left, right = st.columns([1.5, 1])
    with left:
        st.markdown("<div class='hero-card'><div class='section-title'>Risk Distribution</div>", unsafe_allow_html=True)
        if patients:
            chart = pd.DataFrame(
                {"Assessments": [counts["Critical"], counts["High"], counts["Moderate"], counts["Low"]]},
                index=["Critical", "High", "Moderate", "Low"],
            )
            st.bar_chart(chart, height=220)
        else:
            st.markdown("<div class='small-muted'>No assessments yet. Complete a patient assessment to populate analytics.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='hero-card'><div class='section-title'>Recent Assessments</div>", unsafe_allow_html=True)
        if patients:
            for patient in patients[:5]:
                icon = _risk_icon(patient["risk_level"])
                st.markdown(
                    f"<div class='summary-row'><span><b>{icon} {patient['patient_name'] or 'Unnamed Patient'}</b><br><span class='small-muted'>{patient['created_at']}</span></span><span class='summary-value'>{patient['risk_level']}</span></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown("<div class='small-muted'>No patient history available.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


def _render_assessment(
    input_language_name: str,
    response_language_name: str,
) -> None:
    st.markdown(
        """
        <div class="safety">⚠️ <b>Safety notice:</b> ASHA AI Guardian provides preliminary decision support only. It is not a diagnostic system. Emergency classifications require professional medical evaluation.</div>
        """,
        unsafe_allow_html=True,
    )

    patient_col, language_col = st.columns(2)
    with patient_col:
        st.markdown("<div class='input-card'><div class='section-title'>👤 Patient Information</div>", unsafe_allow_html=True)
        patient_name = st.text_input("Patient Name", placeholder="Enter patient name", label_visibility="visible")
        age = st.number_input("Age (Years)", min_value=0, max_value=120, value=25, step=1)
        st.markdown("</div>", unsafe_allow_html=True)

    with language_col:
        st.markdown("<div class='input-card'><div class='section-title'>🌐 Language Settings</div>", unsafe_allow_html=True)
        st.selectbox("Input Language", list(LANGUAGES), index=list(LANGUAGES).index(input_language_name), key="assessment_input_language")
        st.selectbox("Response Language", list(LANGUAGES), index=list(LANGUAGES).index(response_language_name), key="assessment_response_language")
        st.markdown("</div>", unsafe_allow_html=True)

    input_language_name = st.session_state.get("assessment_input_language", input_language_name)
    response_language_name = st.session_state.get("assessment_response_language", response_language_name)

    st.markdown("<div style='height:.9rem'></div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-card'><div class='section-title'>How would you like to provide symptoms?</div>", unsafe_allow_html=True)
    mode_cols = st.columns(4)
    modes = [("📝", "Text Input", "Type symptoms"), ("🎙", "Voice Input", "Speak symptoms"), ("🖼", "Upload Report", "Upload image"), ("📋", "Sample Cases", "Try examples")]
    for col, (icon, title, text) in zip(mode_cols, modes):
        with col:
            st.markdown(f"<div class='mode-card'><div class='mode-icon'>{icon}</div><div class='mode-title'>{title}</div><div class='mode-text'>{text}</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:.9rem'></div>", unsafe_allow_html=True)
    st.markdown("<div class='input-card'><div class='section-title'>✎ Symptoms</div>", unsafe_allow_html=True)
    quick = st.pills(
        "Quick symptom selection",
        QUICK_SYMPTOMS,
        selection_mode="multi",
        label_visibility="collapsed",
    )
    typed = st.text_area(
        "Symptoms / clinical notes",
        height=125,
        placeholder="Describe symptoms (e.g., chest pain, high fever, breathing difficulty...)",
        label_visibility="visible",
    )

    voice_text = st.session_state.get("voice_text", "")
    voice_col, reset_col = st.columns([1, 1])
    with voice_col:
        if st.button("🎙 Speak symptoms"):
            voice_text, error = listen(LANGUAGES[input_language_name])
            if error:
                st.warning(error)
            elif voice_text:
                st.session_state["voice_text"] = voice_text
                st.rerun()
    with reset_col:
        if st.button("↻ Reset input"):
            st.session_state.pop("voice_text", None)
            st.session_state.pop("latest", None)
            st.rerun()
    if voice_text:
        st.info(f"Voice input: {voice_text}")

    report_text = ""
    uploaded = st.file_uploader("🖼 Upload report image for OCR", type=["png", "jpg", "jpeg"], label_visibility="visible")
    if uploaded:
        image = Image.open(uploaded)
        image_col, ocr_col = st.columns(2)
        with image_col:
            st.image(image, caption="Uploaded report", use_container_width=True)
        with ocr_col:
            with st.spinner("Extracting text from report..."):
                report_text = extract_text(image)
            st.text_area("OCR result (English)", report_text, height=150)

    analyze = st.button("🛡  Analyze Patient", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if analyze:
        raw_user_text = ", ".join(quick or []) + " " + typed + " " + voice_text
        english_user_text = translate_text(raw_user_text, LANGUAGES[input_language_name], "en")
        combined = " ".join(part for part in [english_user_text, report_text] if part.strip()).strip()
        if not combined:
            st.warning("Please enter symptoms, select a symptom, speak, or upload a report.")
            return

        with st.spinner("Running safety-first assessment..."):
            result = triage(combined, int(age))
            ai_text = ask_ai(combined, result)
            display_ai = translate_text(ai_text, "en", LANGUAGES[response_language_name])
            display_recommendation = translate_text(result.recommendation, "en", LANGUAGES[response_language_name])

        st.session_state["latest"] = {
            "result": result,
            "ai": display_ai,
            "recommendation": display_recommendation,
            "symptoms": combined,
            "name": patient_name,
            "age": int(age),
            "input_language": input_language_name,
            "response_language": response_language_name,
        }
        save_patient(patient_name, int(age), combined, result)
        st.rerun()

    _render_result()


def _render_result() -> None:
    latest = st.session_state.get("latest")
    if not latest:
        return

    result = latest["result"]
    max_score = 15
    marker = max(2, min(98, (result.risk_score / max_score) * 100))
    risk_class = _risk_class(result.risk_level)
    icon = _risk_icon(result.risk_level)

    st.markdown("<div style='height:1rem'></div><div class='section-title'>Assessment Result</div>", unsafe_allow_html=True)
    left, right = st.columns([1, 1.65])
    with left:
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='section-title'>Risk Level</div><div class='risk-pill {risk_class}'>{icon} {result.risk_level}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='score'>{result.risk_score} <span>/ 15 risk score</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='score-track'><div class='score-marker' style='left:{marker}%'></div></div><div class='scale'><span>0</span><span>5</span><span>10</span><span>15</span></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='result-card'><div class='section-title'>👥 Detected Symptoms (Matched)</div>", unsafe_allow_html=True)
        if result.detected_symptoms:
            st.markdown("".join(f"<span class='symptom-tag'>{s} &nbsp;!</span>" for s in result.detected_symptoms), unsafe_allow_html=True)
        else:
            st.markdown("<div class='small-muted'>No configured symptoms were matched.</div>", unsafe_allow_html=True)
        st.markdown("<div style='height:.7rem'></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='rec-card'><b>🩺 Recommendation</b><br><br><strong>{latest['recommendation']}</strong></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:.9rem'></div>", unsafe_allow_html=True)
    ai_col, summary_col = st.columns([1.65, 1])
    with ai_col:
        st.markdown("<div class='ai-card'><div class='ai-label'>🧠 AI Explanation <span class='small-muted'>(via Ollama)</span></div><div style='height:.65rem'></div>", unsafe_allow_html=True)
        st.write(latest["ai"])
        if st.button("🔊 Read explanation aloud"):
            ok, error = speak(latest["ai"])
            if not ok:
                st.warning(error)
        st.markdown("</div>", unsafe_allow_html=True)

    with summary_col:
        st.markdown("<div class='result-card'><div class='section-title'>Assessment Summary</div>", unsafe_allow_html=True)
        rows = [
            ("Patient Name", latest["name"] or "—"),
            ("Age", latest["age"]),
            ("Risk Level", result.risk_level),
            ("Risk Score", f"{result.risk_score} / 15"),
            ("Input Method", "Multi-input"),
            ("Language", latest.get("response_language", "English")),
        ]
        for key, value in rows:
            value_class = "color:#dc2626;" if key in {"Risk Level", "Risk Score"} and "LOW" not in str(value) else ""
            st.markdown(f"<div class='summary-row'><span class='summary-key'>{key}</span><span class='summary-value' style='{value_class}'>{value}</span></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    report = build_report(latest["name"], latest["age"], latest["symptoms"], result, latest["ai"])
    export_col, new_col = st.columns([1, 1])
    with export_col:
        st.download_button("⬇ Download Report", report, file_name="asha_ai_assessment.txt", mime="text/plain", use_container_width=True)
    with new_col:
        if st.button("＋ New Assessment", use_container_width=True):
            st.session_state.pop("latest", None)
            st.session_state.pop("voice_text", None)
            st.rerun()

    st.markdown("<div class='footer-note'>ⓘ <b>Disclaimer:</b> ASHA AI Guardian provides preliminary risk assessment for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment.</div>", unsafe_allow_html=True)


def _render_history(patients: list[dict]) -> None:
    st.markdown("<div class='hero-card'><div class='section-title'>Assessment History</div><div class='section-caption'>Local SQLite assessment records</div>", unsafe_allow_html=True)
    if patients:
        display = pd.DataFrame(patients)
        st.dataframe(display, use_container_width=True, hide_index=True)
    else:
        st.info("No assessment history yet.")
    st.markdown("</div>", unsafe_allow_html=True)


def _render_settings() -> None:
    st.markdown("<div class='hero-card'><div class='section-title'>⚙ Settings</div>", unsafe_allow_html=True)
    st.write(f"**Local LLM:** `{OLLAMA_MODEL}`")
    st.write("**Safety authority:** Deterministic triage engine")
    st.write("**Supported languages:** English, Kannada, Hindi")
    st.write("**Storage:** Local SQLite")
    st.caption("The LLM is restricted to explanation. Risk classification remains deterministic.")
    st.markdown("</div>", unsafe_allow_html=True)


def run_app() -> None:
    """Render the professional ASHA AI Guardian dashboard."""
    st.set_page_config(page_title="ASHA AI Guardian", page_icon="🩺", layout="wide", initial_sidebar_state="expanded")

    dark_mode = st.session_state.get("dark_mode_toggle", False)
    _inject_css(dark_mode)
    page, input_language_name, response_language_name = _render_sidebar(dark_mode)

    # The sidebar toggle is read on the next Streamlit rerun so CSS follows the selected theme.
    dark_mode = st.session_state.get("dark_mode_toggle", False)
    _inject_css(dark_mode)
    _render_header(page, dark_mode)

    patients = get_patients()

    if page == "Dashboard":
        _render_overview(patients)
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        _render_assessment(input_language_name, response_language_name)
    elif page == "New Assessment":
        _render_assessment(input_language_name, response_language_name)
    elif page in {"History", "Patients", "Reports", "Analytics"}:
        _render_history(patients)
    elif page == "Settings":
        _render_settings()

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.caption("Local-first prototype • Triage is rule-based and not clinically validated • OCR focuses on English • Translation and Google speech recognition may require internet access")
