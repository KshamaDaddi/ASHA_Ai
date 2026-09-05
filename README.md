# 🩺 ASHA AI Guardian

**Local-first healthcare decision-support prototype with deterministic triage, local LLM explanation, OCR, multilingual input, voice support, SQLite history, and Streamlit dashboard.**

> ⚠️ **Safety:** ASHA AI Guardian is a software prototype for preliminary decision support. It is **not a diagnostic system**, is not clinically validated, and must not replace a qualified healthcare professional or emergency services.

## 🎯 Problem

Community and frontline healthcare workflows may involve incomplete symptom descriptions, language barriers, handwritten/printed reports, and limited access to advanced software. ASHA AI Guardian demonstrates how these inputs can be combined into one explainable workflow while keeping safety-critical risk classification deterministic.

## 🧠 What I Built

The application accepts patient information through:

- Text-based symptom descriptions
- Quick symptom selection
- Voice input
- Uploaded report images using OCR
- English, Kannada, and Hindi input/output translation

The system then:

1. Normalizes the symptom information.
2. Translates non-English user input to English for the triage engine.
3. Runs a deterministic weighted triage algorithm.
4. Applies age-based and high-risk combination rules.
5. Produces **LOW, MODERATE, HIGH, or CRITICAL** risk classification.
6. Sends the triage result to a local Ollama LLM for a simple explanation.
7. Prevents the LLM from becoming the safety authority.
8. Stores assessment history in SQLite.
9. Generates a downloadable assessment report.

## 🏗️ Architecture

```text
                         USER
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
        Text             Voice            Image
          │                │                │
          │          Voice Service          OCR
          │                │                │
          └────────────────┼────────────────┘
                           ▼
                    Translation Layer
                           │
                           ▼
                  Deterministic Triage
                           │
                 ┌─────────┴─────────┐
                 ▼                   ▼
          Risk Classification    Risk Context
                 │                   │
                 │                Ollama
                 │             Explanation only
                 └─────────┬─────────┘
                           ▼
                    Streamlit Dashboard
                           │
                    ┌──────┴──────┐
                    ▼             ▼
                 SQLite        Report
                 History       Download
```

### Safety design

The most important architectural decision is the separation between **decision** and **explanation**:

> **Deterministic triage = safety authority**  
> **Ollama LLM = explanation layer**

The LLM receives the deterministic result as context and is explicitly instructed not to downgrade or contradict HIGH/CRITICAL cases. This reduces the risk of relying on generative output for a safety-critical classification.

## 📁 Project Structure

```text
ASHA_Ai/
├── main.py                         # Streamlit entry point
├── app/
│   ├── __init__.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── triage_engine.py        # Deterministic risk engine
│   │   ├── ollama_service.py       # Local LLM integration
│   │   ├── ocr_service.py          # EasyOCR adapter
│   │   ├── translation_service.py  # Translation adapter
│   │   ├── voice_service.py        # Speech-to-text / TTS
│   │   └── report_service.py       # Report generation
│   ├── database/
│   │   ├── __init__.py
│   │   └── database.py             # SQLite persistence
│   └── ui/
│       ├── __init__.py
│       └── dashboard.py             # Streamlit UI + orchestration
├── tests/
│   ├── __init__.py
│   └── test_triage.py              # Triage unit tests
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

This structure keeps `main.py` intentionally small and makes each technical responsibility easy to understand, test, and explain in an interview.

## 🔬 Triage Engine

The triage engine is rule-based and explainable. Symptoms have predefined weights, for example:

| Signal | Weight |
|---|---:|
| Chest pain | 5 |
| Breathing difficulty | 5 |
| Shortness of breath | 5 |
| Heavy bleeding | 5 |
| Unconscious | 5 |
| Seizure | 5 |
| High fever | 3 |
| Dehydration | 3 |
| Vomiting | 2 |
| Weakness | 2 |
| Headache | 1 |
| Cough | 1 |

Additional rules account for age and symptom combinations. The engine also uses **longest-first phrase matching and span masking**, so `high fever` is not accidentally counted as both `high fever` and `fever`.

Basic negation patterns such as `no chest pain` and `I don't have chest pain` are also handled.

### Risk thresholds

```text
score >= 10  → CRITICAL EMERGENCY
score >= 6   → HIGH RISK
score >= 3   → MODERATE RISK
otherwise    → LOW RISK
```

These are prototype engineering rules, **not clinical guidelines**.

## 🤖 Generative AI

Ollama runs a local model such as `phi3`. The model is used for natural-language explanation, not diagnosis.

The prompt provides:

- Patient symptom text
- Deterministic risk level
- Deterministic risk score
- Detected symptoms
- Safety recommendation

The system prompt explicitly states that the deterministic result is the safety authority and that the model must not prescribe medicines or downgrade HIGH/CRITICAL risk.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| Streamlit | Interactive dashboard |
| Ollama | Local LLM inference |
| EasyOCR | Text extraction from report images |
| deep-translator | Multilingual translation |
| SpeechRecognition | Voice-to-text |
| pyttsx3 | Local text-to-speech |
| SQLite | Local assessment history |
| Pandas | Dashboard data handling |
| Pytest | Automated tests |

## 🚀 Setup

### 1. Clone the repository

```bash
git clone https://github.com/KshamaDaddi/ASHA_Ai.git
cd ASHA_Ai
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama

Install Ollama separately and make sure it is running. Then pull the model configured by the application:

```bash
ollama pull phi3
```

To use another installed model:

```bash
set ASHA_AI_MODEL=<model-name>
```

### 5. Run the application

```bash
streamlit run main.py
```

## 🧪 Run Tests

```bash
pytest -q
```

The tests focus on the safety-critical deterministic layer, including emergency combinations, negation, age adjustments, overlapping symptom phrases, and low-risk cases.

## 🔐 Data and Privacy

Assessment history is stored locally in `asha_ai.db`. The database file is ignored by Git and should not be committed to the repository.

The application is **local-first**, not fully offline: translation and Google speech recognition may require internet access. OCR and text-to-speech are designed to run locally.

## ⚠️ Current Limitations

- Rule-based triage is not a medical diagnostic model.
- Thresholds and weights are prototype engineering choices, not clinical recommendations.
- No clinical validation or prospective evaluation has been performed.
- OCR currently targets English text.
- Voice recognition depends on the configured speech-recognition provider.
- Translation uses an external translation service and may require internet access.
- The local LLM's explanation quality depends on the installed Ollama model.
- Symptom keyword matching cannot replace clinical reasoning or examination.

## 🔮 Future Improvements

1. Replace keyword rules with a validated clinical risk model after obtaining appropriate datasets and clinical oversight.
2. Add structured symptom entities and better negation/context detection.
3. Add multilingual OCR.
4. Add model evaluation with precision, recall, F1, sensitivity, and specificity.
5. Add audit logs and stronger privacy controls.
6. Add automated CI testing with GitHub Actions.
7. Add model/version tracking for reproducibility.
8. Add role-based access if deployed in a real organization.

## 💬 Interview Explanation

> “ASHA AI Guardian is a local-first healthcare decision-support prototype. I designed it around a safety-first architecture where deterministic triage performs the risk classification and a local Ollama LLM is used only to explain the result. The application accepts text, voice, and report images, supports multilingual input, stores assessment history in SQLite, and provides a Streamlit dashboard. I separated the system into services for triage, LLM integration, OCR, translation, voice, reporting, and persistence so each component can be tested and maintained independently.”

### Key engineering point

If asked **“Why not let the LLM decide the risk?”**, explain:

> “LLMs are probabilistic. For a safety-sensitive workflow, I wanted a deterministic and explainable layer to own the classification. The LLM is therefore constrained to explanation, while the application retains control over the safety decision.”

## 📌 Resume-Ready Project Summary

**ASHA AI Guardian — Healthcare Decision-Support Application**

- Built a local-first Streamlit application combining deterministic symptom triage, Ollama LLM explanation, EasyOCR, multilingual translation, voice interaction, and SQLite persistence.
- Designed a safety-first architecture where a deterministic weighted triage engine owns LOW/MODERATE/HIGH/CRITICAL classification while the local LLM provides explanation only.
- Implemented phrase-overlap prevention, basic symptom negation handling, age-based escalation, combination rules, automated triage tests, and downloadable assessment reports.
