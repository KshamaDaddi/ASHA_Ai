from deep_translator import GoogleTranslator

# -----------------------------
# Translate User Input to English
# -----------------------------
def translate_to_english(text, source_lang):

    if source_lang == "English":
        return text

    lang_map = {
        "Kannada": "kn",
        "Hindi": "hi"
    }

    try:

        translated = GoogleTranslator(
            source=lang_map[source_lang],
            target='en'
        ).translate(text)

        return translated

    except Exception:

        return text


# -----------------------------
# Translate AI Response
# -----------------------------
def translate_from_english(text, target_lang):

    if target_lang == "English":
        return text

    lang_map = {
        "Kannada": "kn",
        "Hindi": "hi"
    }

    try:

        translated = GoogleTranslator(
            source='en',
            target=lang_map[target_lang]
        ).translate(text)

        return translated

    except Exception:

        return text