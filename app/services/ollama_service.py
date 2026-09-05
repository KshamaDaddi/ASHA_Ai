"""Local Ollama LLM integration. The LLM explains; it never owns triage safety."""
import os

try:
    import ollama
except ImportError:
    ollama = None

OLLAMA_MODEL = os.getenv("ASHA_AI_MODEL", "phi3")
SYSTEM_PROMPT = """You are ASHA AI Guardian, a healthcare decision-support assistant.
You are NOT a doctor and must not diagnose disease.
The deterministic triage result supplied by the application is the safety authority.
Never contradict, reduce, or downgrade a CRITICAL or HIGH risk result.
For emergency symptoms, advise immediate professional medical care.
Do not prescribe medicines or recommend unsafe treatment changes.
Explain the result in simple, calm language and keep the response concise.
"""

def ask_ai(prompt: str, triage_result=None) -> str:
    if not prompt.strip():
        return "Please describe the patient's symptoms."
    if ollama is None:
        return "Ollama is not installed. Install Ollama and the selected local model."

    context = ""
    if triage_result:
        context = (
            "\nDeterministic triage result (SAFETY AUTHORITY):\n"
            f"Risk level: {triage_result.risk_level}\n"
            f"Risk score: {triage_result.risk_score}\n"
            f"Detected symptoms: {', '.join(triage_result.detected_symptoms) or 'None'}\n"
            f"Recommendation: {triage_result.recommendation}\n"
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
