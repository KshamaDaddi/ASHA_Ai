"""Local Ollama service used only for explanation and conversational support."""

import os
from typing import Optional

import ollama

MODEL_NAME = os.getenv("ASHA_AI_MODEL", "phi3")

SYSTEM_PROMPT = """You are ASHA AI, an offline healthcare support assistant.

Your role is to explain information clearly and help users understand the
preliminary triage result produced by a deterministic safety engine.

Rules:
- Never claim to diagnose a disease.
- Never override a CRITICAL or HIGH risk triage result.
- For emergency symptoms, advise immediate professional/emergency care.
- Do not recommend prescription medicines or unsafe treatment changes.
- Keep responses concise, calm, and easy to understand.
- Clearly state that the assistant is not a substitute for a clinician.
"""


def ask_ai(prompt: str, model: Optional[str] = None) -> str:
    """Generate a local response through Ollama."""
    if not prompt or not prompt.strip():
        return "Please describe your symptoms or ask a healthcare question."

    response = ollama.chat(
        model=model or MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt.strip()},
        ],
    )
    return response["message"]["content"].strip()
