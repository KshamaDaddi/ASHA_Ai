# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ASHA-AI  |  speech.py
# Text-to-speech via gTTS  +  Automatic speech recognition via Whisper (offline)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from __future__ import annotations

import os
import tempfile

from .config import LANG_GT, LANG_WHISPER, WHISPER_MODEL, TTS_MAX_CHARS

# Lazy-loaded Whisper model — loaded only on first transcription call
_whisper_model = None


# ── Text-to-Speech ────────────────────────────────────────────────────────────

def speak(text: str, language: str = "English") -> str | None:
    """
    Convert text to speech using gTTS.

    Args:
        text:     The text to speak (truncated to TTS_MAX_CHARS).
        language: Display language name (e.g. "Kannada", "Hindi", "English").

    Returns:
        Path to a temporary .mp3 file, or None on failure.
    """
    try:
        from gtts import gTTS

        lang_code = LANG_GT.get(language, "en")
        tts = gTTS(text=text[:TTS_MAX_CHARS], lang=lang_code, slow=False)
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tts.save(tmp.name)
        return tmp.name

    except Exception as e:
        print(f"[speak] Error: {e}")
        return None


# ── Automatic Speech Recognition ─────────────────────────────────────────────

def transcribe_audio(audio_path: str, language: str = "auto") -> dict:
    """
    Transcribe an audio file using Whisper (fully offline after first download).

    Args:
        audio_path: Path to the audio file (wav / mp3 / ogg / m4a …).
        language:   Display language name or "auto" for automatic detection.

    Returns:
        Dict with key "text" containing the transcription string.
    """
    global _whisper_model
    try:
        import whisper

        if _whisper_model is None:
            print(f"Loading Whisper '{WHISPER_MODEL}' model (first call — ~30 s)…")
            _whisper_model = whisper.load_model(WHISPER_MODEL)
            print("✅ Whisper loaded")

        whisper_lang = LANG_WHISPER.get(language)   # None → auto-detect
        result = (
            _whisper_model.transcribe(audio_path, language=whisper_lang)
            if whisper_lang
            else _whisper_model.transcribe(audio_path)
        )
        return {"text": result.get("text", "").strip()}

    except Exception as e:
        print(f"[transcribe_audio] Error: {e}")
        return {"text": ""}
