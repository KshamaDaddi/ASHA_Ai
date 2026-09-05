# ASHA AI — Offline Healthcare Support Assistant

ASHA AI is a safety-first healthcare support application that combines deterministic preliminary triage with a locally hosted LLM through Ollama. It also provides multilingual input, medical-report OCR, voice interaction, and patient-history storage.

> **Safety notice:** ASHA AI is a preliminary screening and information-support tool. It is not a medical diagnostic system and must not replace a qualified healthcare professional or emergency service.

## Architecture

```text
Streamlit UI
    |
    v
FastAPI REST API
    |
    +--> Triage Service --------> deterministic risk score
    |
    +--> Ollama Service --------> local LLM explanation
    |
    +--> OCR / Translation / Voice services
    |
    v
SQLite (local application data)
```

## Core design

The application separates **risk assessment** from **LLM generation**. The deterministic triage service makes the preliminary risk decision; the local LLM is used for conversational explanation and general guidance. This prevents an LLM response from silently overriding an emergency rule.

## Features

- Safety-first rule-based preliminary triage
- Negation-aware symptom matching for common phrases such as "I don't have chest pain"
- Age-based risk adjustments
- Combination-symptom escalation rules
- Local Ollama LLM support
- FastAPI REST API with Pydantic validation
- Streamlit interface
- Multilingual input support
- Medical-report OCR
- Voice input and text-to-speech
- Local SQLite patient history
- Automated triage unit tests

## API

Start the backend:

```bash
cd backend
uvicorn app.main:app --reload
```

Then open FastAPI documentation at `http://127.0.0.1:8000/docs`.

### Health

`GET /health`

```json
{"status": "healthy"}
```

### Triage

`POST /api/v1/triage`

```json
{
  "symptoms": "chest pain and breathing difficulty",
  "age": 45
}
```

Example response:

```json
{
  "risk_level": "CRITICAL EMERGENCY",
  "risk_score": 15,
  "detected_symptoms": ["chest pain", "breathing difficulty"],
  "recommendation": "Seek emergency medical care immediately."
}
```

### Chat

`POST /api/v1/chat`

```json
{"message": "What should I do for a mild cough?"}
```

The chat endpoint uses the configured local Ollama model. Set `ASHA_AI_MODEL` to change the model; the default is `phi3`.

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install -r backend/requirements.txt
```

Install and start Ollama separately, then make sure the configured model is available locally.

Run tests:

```bash
pytest backend/tests -q
```

## Project structure

```text
ASHA_Ai/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── routes/
│   │   │   ├── chat.py
│   │   │   └── triage.py
│   │   └── services/
│   │       ├── ollama_service.py
│   │       └── triage_service.py
│   ├── tests/
│   │   └── test_triage.py
│   ├── streamlit_app.py
│   ├── translator.py
│   └── voice_assistant.py
├── .gitignore
├── README.md
└── LICENSE
```

## Triage limitations

The current engine is intentionally a lightweight preliminary screening layer based on a curated symptom-weight table. It does not diagnose conditions, understand every medical phrase, or replace clinical assessment. Emergency decisions should be confirmed by appropriate healthcare services.

## Roadmap

- Move all Streamlit business logic behind FastAPI
- Add OCR and translation service endpoints
- Add patient API with proper schemas and validation
- Add authentication and audit logging before any real patient deployment
- Expand automated safety and API tests
- Add structured logging and monitoring
- Add a proper evaluation dataset for triage behavior
- Add containerized deployment

## License

See [LICENSE](LICENSE).
