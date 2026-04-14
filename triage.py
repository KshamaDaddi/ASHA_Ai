# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ASHA-AI  |  triage.py
# End-to-end triage pipeline: translate → infer → classify → speak
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from __future__ import annotations

from .config import STRONG_EMERGENCY_KW, REFER_KW, HOME_KW
from .inference import asha_triage
from .translation import translate_en_to_indian, translate_indian_to_en
from .speech import speak

# ── Triage classification ─────────────────────────────────────────────────────

def classify_urgency(english_response: str) -> str:
    """
    Classify the urgency level of a triage response.

    Priority order:
        1. Strong emergency keywords  →  EMERGENCY
        2. General home-management keywords  →  HOME
        3. Any referral keywords  →  REFER
        4. Default  →  HOME (safe default)

    Returns one of:
        "🚨 EMERGENCY — Call 108 immediately"
        "⚠️ REFER to nearest health facility"
        "✅ Home management — Monitor closely"
    """
    r = english_response.lower()

    if any(k in r for k in STRONG_EMERGENCY_KW):
        return "🚨 EMERGENCY — Call 108 immediately"
    if any(k in r for k in HOME_KW):
        return "✅ Home management — Monitor closely"
    if any(k in r for k in REFER_KW):
        return "⚠️ REFER to nearest health facility"
    return "✅ Home management — Monitor closely"


# ── Full triage pipeline ──────────────────────────────────────────────────────

def full_triage(
    query: str,
    output_language: str = "English",
    generate_audio: bool = True,
) -> tuple[str, str, str | None]:
    """
    Complete triage pipeline for a single patient query.

    Steps:
        1. Translate input → English (if needed)
        2. RAG + Gemma inference (always in English for accuracy)
        3. Classify urgency from English response
        4. Translate response → requested output language
        5. Generate TTS audio

    Args:
        query:           Symptom description (any supported language).
        output_language: Display language for the response.
        generate_audio:  Whether to generate a TTS mp3 file.

    Returns:
        (translated_response, status_label, audio_path_or_None)
    """
    if not query.strip():
        return "Please describe the patient's symptoms.", "Waiting", None

    eng_query    = translate_indian_to_en(query, output_language)
    eng_response = asha_triage(eng_query, language="en")
    status       = classify_urgency(eng_response)
    translated   = translate_en_to_indian(eng_response, output_language)
    audio        = speak(translated[:400], output_language) if generate_audio else None

    return translated, status, audio


def safe_full_triage(
    query: str,
    output_language: str = "English",
) -> tuple[str, str, str | None]:
    """
    Wrapper around full_triage with graceful error handling for the Gradio UI.
    """
    try:
        if not query or not str(query).strip():
            return "Please enter a valid query.", "❌ Empty input", None

        result = full_triage(query, output_language)

        if isinstance(result, tuple):
            if len(result) == 3:
                return result
            if len(result) == 2:
                return result[0], result[1], None
            if len(result) == 1:
                return result[0], "✅ Done", None

        if isinstance(result, str):
            return result, "✅ Done", None

        return "Unexpected output format.", "❌ Error", None

    except Exception as e:
        return f"Error: {e}", "❌ Failed", None
