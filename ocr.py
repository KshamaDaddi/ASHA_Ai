# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ASHA-AI  |  ocr.py
# Medicine label analysis: Tesseract OCR → Gemma explanation → translation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from __future__ import annotations

from .config import LANG_TESSERACT
from .inference import asha_triage
from .translation import translate_en_to_indian


def extract_text_from_image(image, language: str = "English") -> str:
    """
    Run Tesseract OCR on a PIL image, supporting English and major Indian scripts.

    Args:
        image:    PIL.Image object.
        language: Display language name used to select the Tesseract language pack.

    Returns:
        Extracted text string, or a fallback message if OCR fails.
    """
    try:
        import pytesseract

        tess_lang = "eng"
        extra     = LANG_TESSERACT.get(language)
        if extra:
            tess_lang = f"eng+{extra}"

        ocr_text = pytesseract.image_to_string(image, lang=tess_lang).strip()
        return ocr_text if ocr_text else "(No text detected on the label)"

    except Exception as e:
        print(f"[extract_text_from_image] OCR error: {e}")
        return "(OCR unavailable)"


def analyse_image(image, question: str, language: str = "English") -> str:
    """
    Full medicine-image analysis pipeline:
      1. OCR the label with Tesseract
      2. Build an ASHA-tailored prompt
      3. Run through fine-tuned Gemma for explanation
      4. Translate response to the requested language

    Args:
        image:    PIL.Image of a medicine label / packet.
        question: The ASHA worker's question about the medicine.
        language: Desired response language.

    Returns:
        Translated explanation string.
    """
    ocr_text = extract_text_from_image(image, language)

    user_q = question.strip() if (question and question.strip()) else "Explain this medicine."

    prompt = (
        f"Medicine label text (from OCR):\n{ocr_text}\n\n"
        f"ASHA worker question: {user_q}\n\n"
        f"Explain in simple terms suitable for a rural health worker. "
        f"Include: medicine name, uses, correct dose, side effects, and any safety warnings. "
        f"Respond in English first; translation will follow."
    )

    eng_response = asha_triage(prompt, language="en")
    return translate_en_to_indian(eng_response, language)
