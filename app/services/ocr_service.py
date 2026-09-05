"""Lazy English OCR service for uploaded report images."""
from functools import lru_cache
import numpy as np
from PIL import Image

@lru_cache(maxsize=1)
def get_ocr_reader():
    try:
        import easyocr
        return easyocr.Reader(["en"], gpu=False)
    except Exception:
        return None

def extract_text(image: Image.Image) -> str:
    reader = get_ocr_reader()
    if reader is None:
        return "OCR is unavailable because EasyOCR is not installed."
    try:
        results = reader.readtext(np.array(image))
        return " ".join(result[1] for result in results).strip()
    except Exception:
        return "OCR could not process this image."
