# 🩺 ASHA AI Guardian

**A local-first healthcare decision-support prototype combining deterministic preliminary triage, a local Ollama LLM, OCR, multilingual interaction, voice support, SQLite history, and a Streamlit dashboard.**

> ⚠️ **Safety:** This is a software prototype for preliminary decision support. It is **not a diagnostic system**, is not clinically validated, and must not replace a qualified healthcare professional or emergency services.

## 🎯 Problem

Frontline/community healthcare workflows can involve incomplete symptom descriptions, language barriers, voice input, and information contained in uploaded reports. ASHA AI Guardian demonstrates how these inputs can be combined into one explainable workflow while keeping the safety-sensitive risk decision deterministic.

## 🧠 What I Built

The application accepts:

- Typed symptom descriptions
- Quick symptom selection
- Voice input
- Uploaded report images using EasyOCR
- English, Kannada, and Hindi interaction

The workflow is:

```text
Patient Input
   │
   ├── Text ──────────────┐
   ├── Voice → STT ───────┤
   └── Report → OCR ──────┤
                          ▼
                   Translation
                          │
                          ▼
                 Deterministic Triage
                          │
                          ▼
                  Risk Classification
                          │
                ┌─────────┴─────────┐
                ▼                   ▼
           Safety Result       Ollama LLM
                                    │
                              Explanation only
                └─────────┬─────────┘
                          ▼
                   Streamlit UI
                    ┌─────┴─────┐
                    ▼           ▼
                 SQLite      Report
                 History     Download
```

### Core design principle

```text
Deterministic Triage  →  SAFETY AUTHORITY
Ollama LLM            →  EXPLANATION ONLY
```

The LLM receives the deterministic result as context and is instructed not to diagnose, prescribe medicines, or downgrade HIGH/CRITICAL cases.

---

## 🏗️ Architecture

The project uses a modular service-oriented structure instead of putting all logic into `main.py`.

```text
                         Streamlit UI
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
        Voice                 OCR                Text
          │                   │                   │
          └───────────────────┼───────────────────┘
                              ▼
                    Translation Service
                              │
                              ▼
                     Triage Engine
                              │
                              ▼
                       Risk Result
                         │         │
                         │         ▼
                         │    Ollama Service
                         │    Explanation
                         │         │
                         └────┬────┘
                              ▼
                       Report Service
                              │
                              ▼
                       SQLite Database
```

### Project structure

```text
ASHA_Ai/
├── main.py                         # Application entry point
├── app/
│   ├── services/
│   │   ├── triage_engine.py        # Deterministic risk engine
│   │   ├── ollama_service.py       # Local LLM integration
│   │   ├── ocr_service.py          # EasyOCR adapter
│   │   ├── translation_service.py  # Translation adapter
│   │   ├── voice_service.py        # STT / TTS
│   │   └── report_service.py       # Report generation
│   ├── database/
│   │   └── database.py             # SQLite persistence
│   └── ui/
│       └── dashboard.py             # Streamlit UI/orchestration
├── tests/
│   └── test_triage.py              # Triage unit tests
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

| Module | Responsibility |
|---|---|
| `main.py` | Starts the application |
| `triage_engine.py` | Risk scoring and classification |
| `ollama_service.py` | Local LLM explanation |
| `ocr_service.py` | Report image text extraction |
| `translation_service.py` | Language translation |
| `voice_service.py` | Speech-to-text and text-to-speech |
| `report_service.py` | Downloadable report creation |
| `database.py` | Local assessment history |
| `dashboard.py` | UI and workflow orchestration |
| `test_triage.py` | Automated tests |

---

## 🔬 Deterministic Triage Engine

The triage engine is rule-based, deterministic, and explainable. Prototype symptom weights include:

| Symptom / Signal | Weight |
|---|---:|
| Breathing difficulty | 5 |
| Shortness of breath | 5 |
| Heavy bleeding | 5 |
| Chest pain | 5 |
| Unconscious | 5 |
| Seizure | 5 |
| High fever | 3 |
| Dehydration | 3 |
| Vomiting | 2 |
| Dizziness | 2 |
| Weakness | 2 |
| Fever | 2 |
| Headache | 1 |
| Cough | 1 |

Additional escalation rules consider patient age and combinations such as chest pain + breathing difficulty, high fever + dehydration, and vomiting + weakness.

### Risk thresholds

```text
Score >= 10  → CRITICAL EMERGENCY
Score >= 6   → HIGH RISK
Score >= 3   → MODERATE RISK
Score < 3    → LOW RISK
```

These are **prototype engineering rules, not clinical guidelines**.

### Robustness improvements

- Longest-first phrase matching
- Span-overlap prevention so `high fever` is not double-counted as `high fever` + `fever`
- Basic negation detection such as `no chest pain`
- Age-based escalation
- High-risk symptom-combination rules
- Structured `TriageResult` returned by the engine

---

## 🤖 Generative AI

Ollama runs a local LLM such as `phi3`.

The model receives:

- Patient symptom text
- Deterministic risk level
- Risk score
- Detected symptoms
- Safety recommendation

It generates a concise explanation for the user.

### Why separate the LLM from triage?

LLMs are probabilistic. For a safety-sensitive workflow, the application needs a predictable and testable decision boundary. Therefore:

```text
Rules → Decide
LLM   → Explain
```

The LLM cannot become the safety authority.

---

## 🌐 Multilingual Support

Supported languages:

- English
- Kannada
- Hindi

The intended flow is:

```text
User Language
      ↓
Translation to English
      ↓
Triage Processing
      ↓
Ollama Explanation
      ↓
Translation to Response Language
```

Translation is isolated as a service so it can later be replaced by another model or provider.

---

## 🖼️ OCR

Uploaded report images are processed with EasyOCR:

```text
Report Image → EasyOCR → Extracted Text → Assessment Workflow
```

Current OCR processing focuses on English text. OCR output is extracted text and should not be treated as verified clinical information.

---

## 🎙️ Voice

The voice layer uses:

- `SpeechRecognition` for speech-to-text
- Google speech recognition for recognition
- `pyttsx3` for local text-to-speech

```text
Microphone → Speech Recognition → Text → Triage → Explanation → TTS
```

Speech recognition may require internet access.

---

## 💾 SQLite History

The application stores local assessment history containing:

- Patient name
- Age
- Symptoms
- Risk level
- Risk score
- Recommendation
- Timestamp

`asha_ai.db` is ignored by Git and should not be committed.

---

## 📊 Dashboard

The Streamlit dashboard provides:

- Patient information input
- Quick symptom selection
- Voice input
- Report upload and OCR
- Multilingual interaction
- Risk classification
- Risk distribution visualization
- Local LLM explanation
- Text-to-speech
- Assessment history
- Downloadable reports

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| Streamlit | UI and dashboard |
| Ollama | Local LLM inference |
| EasyOCR | OCR |
| deep-translator | Translation |
| SpeechRecognition | Speech-to-text |
| pyttsx3 | Text-to-speech |
| SQLite | Local persistence |
| Pandas | Dashboard data |
| NumPy | Image/OCR processing |
| Pytest | Unit testing |

---

## 🚀 Setup

### 1. Clone

```bash
git clone https://github.com/KshamaDaddi/ASHA_Ai.git
cd ASHA_Ai
```

### 2. Create environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Ollama

Install Ollama and pull the default model:

```bash
ollama pull phi3
```

To use another installed model:

Windows CMD:

```bash
set ASHA_AI_MODEL=<model-name>
```

PowerShell:

```powershell
$env:ASHA_AI_MODEL="<model-name>"
```

macOS/Linux:

```bash
export ASHA_AI_MODEL=<model-name>
```

### 5. Run

```bash
streamlit run main.py
```

---

## 🧪 Testing

```bash
pytest -q
```

Tests focus on the deterministic safety layer, including:

- Emergency combinations
- Negated symptoms
- Overlapping symptom phrases
- Age escalation
- Child-specific escalation
- Low-risk classification

Because the triage engine is separated from Streamlit, it can be tested independently.

---

## 🔐 Privacy and Data Handling

ASHA AI Guardian is **local-first, not fully offline**.

Local components include Ollama inference, SQLite storage, OCR, and text-to-speech. Translation and Google speech recognition may require internet access.

Do not use real patient data with this prototype unless appropriate privacy, security, consent, and organizational requirements have been addressed.

---

## ⚠️ Limitations

- The triage engine is not a medical diagnostic model.
- Weights and thresholds are prototype engineering choices.
- No clinical validation has been performed.
- Keyword matching cannot replace clinical reasoning or examination.
- Basic negation handling does not cover every linguistic context.
- OCR may produce incorrect text.
- Current OCR focuses on English.
- Translation and speech recognition may introduce errors.
- LLM explanations can still be imperfect.

---

## 🔮 Future Improvements

### AI / ML

- Replace prototype rules with a clinically validated risk model.
- Add structured symptom/entity extraction.
- Improve negation and context detection.
- Evaluate models using precision, recall, F1, sensitivity, and specificity.
- Add model/version tracking.

### Generative AI

- Structured LLM output.
- Hallucination evaluation.
- RAG using trusted medical sources.
- Stronger safety guardrails.
- Automated LLM evaluation.

### Engineering

- GitHub Actions CI.
- Structured logging.
- Integration tests.
- Docker support.
- Stronger privacy/security controls.
- Production monitoring.

### Product

- Multilingual OCR.
- Accessibility improvements.
- Authentication and role-based access.
- Human-in-the-loop review for high-risk workflows.

---

## 💬 Interview Explanation

### 30-second answer

> “ASHA AI Guardian is a local-first healthcare decision-support prototype. It accepts symptoms through text, voice, and report images, supports multilingual interaction, and performs preliminary risk classification using a deterministic triage engine. I integrated a local Ollama LLM to explain the result, but I deliberately did not let the LLM make the safety-critical decision. Assessment history is stored in SQLite and the application is presented through Streamlit.”

### Technical answer

> “I designed the application using a modular service-oriented architecture. The UI handles interaction while separate services handle triage, OCR, translation, voice, reporting, persistence, and LLM inference. The triage engine assigns symptom weights, applies age and combination rules, handles basic negation, and returns a structured TriageResult. That result is passed to Ollama as context for explanation. This keeps the probabilistic LLM separate from the safety authority.”

### Why not let the LLM decide?

> “An LLM is probabilistic and can produce inconsistent outputs. For a safety-sensitive workflow, I wanted the classification logic to be deterministic, explainable, and testable. Therefore, the triage engine owns the risk decision and the LLM is restricted to explanation.”

### Is this a medical AI model?

> “It is a healthcare decision-support prototype, not a clinically validated medical model. The current triage layer uses engineering rules. A production system would require appropriate datasets, clinical validation, regulatory review, privacy controls, and human oversight.”

---

## 📌 Resume-Ready Description

**ASHA AI Guardian — Healthcare Decision-Support Application**

- Built a local-first Streamlit healthcare decision-support application integrating deterministic symptom triage, local Ollama LLM explanation, EasyOCR, multilingual translation, voice interaction, SQLite persistence, and downloadable reports.
- Designed a safety-first architecture where deterministic weighted triage owns LOW/MODERATE/HIGH/CRITICAL classification while the local LLM is restricted to natural-language explanation.
- Implemented symptom phrase-overlap prevention, basic negation handling, age-based escalation, high-risk combination rules, modular services, and automated triage tests.

---

## 👩‍💻 Author

**Kshama Daddi**  
AI & Data Science | Python | Machine Learning | Generative AI

GitHub: https://github.com/KshamaDaddi

---

## 📄 License

See the `LICENSE` file included in the repository.

> **Project principle:** Build AI systems where the model is useful, but the application remains responsible for critical decision boundaries.
