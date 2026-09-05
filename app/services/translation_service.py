"""Translation adapter used by the presentation layer."""
try:
    from deep_translator import GoogleTranslator
except ImportError:
    GoogleTranslator = None

def translate_text(text: str, source: str, target: str) -> str:
    if not text or source == target or GoogleTranslator is None:
        return text
    try:
        return GoogleTranslator(source=source, target=target).translate(text)
    except Exception:
        return text
