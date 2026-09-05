# 🩺 ASHA AI Guardian

**Offline, multilingual healthcare support assistant built with Python, Streamlit, Ollama, OCR, voice interaction, and a deterministic safety-first triage engine.**

> ⚠️ **Safety notice:** ASHA AI is a preliminary screening and information-support tool. It is not a diagnostic system and must not replace a qualified healthcare professional or emergency service.

## What it does

ASHA AI accepts patient symptoms by text or voice, optionally reads a medical-report image with OCR, performs a deterministic preliminary risk assessment, stores a local patient history, and uses a locally hosted Ollama model to explain the result.

### Safety architecture

The LLM does **not** decide the emergency level. The deterministic triage engine makes the risk assessment first; Ollama is used only for explanation and conversational support.

```text
Patient
  │
  ├── Text symptoms
  ├── Voice input
  └── Medical report image
          │
          ▼
   Input normalization
          │
          ├── Translation
          └── OCR
          │
          ▼
  Deterministic Triage Engine
          │
          ├── Symptom weights
          ├── Age adjustment
          ├── Combination rules
          └── Negation handling
          │
          ▼
   Risk Classification
          │
          ├── Critical Emergency
          ├── High Risk
          ├── Moderate Risk
          └── Low Risk
          │
          ▼
       Ollama LLM
   explanation / guidance
          │
          ▼
     Streamlit Dashboard
          │
          ▼
      Local SQLite DB
```

## Key features

- 🔴 Safety-first deterministic triage
- 🧠 Local LLM through Ollama
- 🌐 English, Kannada, and Hindi support
- 🖼 Medical-report OCR with EasyOCR
- 🎤 Voice symptom input
- 🔊 Text-to-speech response
- 👤 Local patient history
- 📊 Risk analytics dashboard
- 🛡️ Negation-aware symptom detection
- 🧪 Reproducible Python dependencies
- 🔐 Local-first data storage

## Project structure

```text
ASHA_Ai/
├── main.py             # Complete application
├── requirements.txt    # Python dependencies
├── README.md
├── LICENSE
└── .gitignore
```

The application has intentionally been consolidated into **one `main.py`** so it is easy to understand, run, demonstrate, and submit as a portfolio project.

## Installation

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

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama

Install Ollama separately and pull a small local model, for example:

```bash
ollama pull phi3
```

If you use another installed model, set:

```text
ASHA_AI_MODEL=your-model-name
```

### 5. Run ASHA AI

```bash
streamlit run main.py
```

The Streamlit application will provide the local URL in the terminal.

## Example output

Input:

```text
Patient age: 45
Symptoms: chest pain and breathing difficulty
```

Triage output:

```text
🔴 CRITICAL EMERGENCY

Risk score: 15
Detected symptoms: chest pain, breathing difficulty
Recommendation: Seek emergency medical care immediately.
```

The application then sends the symptoms **together with the deterministic triage result** to Ollama for a concise explanation. The LLM cannot downgrade the emergency classification.

## Risk engine

The current preliminary scoring layer uses curated symptom weights and escalation rules.

| Risk | Score | Action |
|---|---:|---|
| Critical Emergency | ≥ 10 | Seek emergency medical care immediately |
| High Risk | 6–9 | Seek urgent medical evaluation |
| Moderate Risk | 3–5 | Monitor and seek medical advice |
| Low Risk | 0–2 | Basic precautions; seek advice if symptoms worsen |

This is a **screening heuristic**, not a clinically validated medical scoring system.

## Why use an LLM + rules?

A general-purpose LLM is useful for natural-language explanations, but it should not be the sole authority for safety-critical classification. ASHA AI therefore follows:

**Rules → Risk decision → LLM explanation**

This makes the critical decision path deterministic and easier to test.

## Privacy

Patient data is stored in a local SQLite database created at runtime. Database files are ignored by Git and should never be committed when they contain real patient information.

Do not use this prototype with real patient data without appropriate security, privacy, authentication, access control, audit logging, encryption, clinical validation, and regulatory review.

## Limitations

- Symptom coverage is intentionally limited.
- The triage engine is not clinically validated.
- OCR quality depends on image quality.
- Voice recognition depends on the local environment and speech-recognition service.
- Translation may fail or produce imperfect wording.
- Ollama must be installed and the selected model must be available locally.
- The application must not be treated as a medical diagnosis or emergency service.

## Future improvements

1. Add a validated clinical evaluation dataset.
2. Add structured symptom extraction with better medical NLP.
3. Add more robust negation and uncertainty detection.
4. Add automated API/UI tests.
5. Add encrypted storage and authentication for any controlled deployment.
6. Add model evaluation metrics and safety benchmarks.
7. Add Docker and CI/CD.
8. Add clinician-reviewed emergency protocols.

## License

See [LICENSE](LICENSE).
