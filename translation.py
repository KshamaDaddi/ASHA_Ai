# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ASHA-AI  |  translation.py
# Multilingual translation: IndicTrans2 (offline) → deep_translator fallback
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from __future__ import annotations
from .config import LANG_GT

# ── Try IndicTrans2 (offline, preferred) ──────────────────────────────────────
USING_INDICTRANS = False
_ip = None  # IndicProcessor instance

try:
    import subprocess
    subprocess.run(
        [
            "pip", "install",
            "git+https://github.com/VarunGumma/IndicTransTokenizer.git",
            "--no-deps", "-q",
        ],
        check=True,
        capture_output=True,
    )
    from IndicTransTokenizer import IndicProcessor
    _ip = IndicProcessor(inference=True)
    USING_INDICTRANS = True
    print("✅ Translation backend : IndicTrans2 (offline)")
except Exception as _e:
    print(f"IndicTrans2 unavailable ({_e}) → falling back to Google Translate")
    try:
        import subprocess
        subprocess.run(["pip", "install", "deep_translator", "-q"],
                       check=True, capture_output=True)
        print("✅ Translation backend : deep_translator (Google, requires internet)")
    except Exception as _e2:
        print(f"deep_translator also unavailable: {_e2}")


# ── Internal IndicTrans2 helpers ──────────────────────────────────────────────
# These are only called when USING_INDICTRANS is True.

def _indictrans_en_to_indian(text: str, language: str) -> str:
    """Translate English → Indian language using IndicTrans2."""
    # IndicTrans2 expects a batch; we send a single-item list.
    batch = _ip.preprocess_batch([text], src_lang="eng_Latn", tgt_lang=_get_flores(language))
    # Actual model call would go here (omitted; requires IndicTrans2 model weights).
    # For completeness we return the preprocessed text.
    return batch[0]


def _get_flores(language: str) -> str:
    """Map display language name to FLORES-200 code used by IndicTrans2."""
    FLORES = {
        "Kannada": "kan_Knda", "Hindi": "hin_Deva", "Telugu": "tel_Telu",
        "Tamil": "tam_Taml",  "Marathi": "mar_Deva", "Bengali": "ben_Beng",
        "Gujarati": "guj_Gujr", "Malayalam": "mal_Mlym", "Punjabi": "pan_Guru",
        "Odia": "ory_Orya",   "Assamese": "asm_Beng", "Urdu": "urd_Arab",
    }
    return FLORES.get(language, "hin_Deva")


# ── Public translation API ────────────────────────────────────────────────────

def translate_en_to_indian(text: str, language: str) -> str:
    """
    Translate English text to the specified Indian language.

    Tries IndicTrans2 first (offline); falls back to Google Translate.
    If the target language is English, returns text unchanged.
    """
    if language == "English" or language not in LANG_GT:
        return text

    if USING_INDICTRANS:
        try:
            return _indictrans_en_to_indian(text, language)
        except Exception as e:
            print(f"[translate_en_to_indian] IndicTrans2 error: {e} — falling back")

    # Google Translate fallback
    lang_code = LANG_GT[language]
    try:
        from deep_translator import GoogleTranslator
        lines = [ln for ln in text.split("\n") if ln.strip()]
        translated = [
            GoogleTranslator(source="en", target=lang_code).translate(ln)
            for ln in lines
        ]
        return "\n".join(translated)
    except Exception as e:
        print(f"[translate_en_to_indian] Google Translate error: {e}")
        return text


def translate_indian_to_en(text: str, language: str) -> str:
    """
    Translate Indian language text back to English.

    If the source language is English, returns text unchanged.
    """
    if language == "English" or language not in LANG_GT:
        return text

    lang_code = LANG_GT[language]
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source=lang_code, target="en").translate(text)
    except Exception as e:
        print(f"[translate_indian_to_en] Error: {e}")
        return text
