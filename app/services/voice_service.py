"""Speech-to-text and text-to-speech adapters."""
SPEECH_LOCALES = {"en": "en-IN", "kn": "kn-IN", "hi": "hi-IN"}

def listen(language: str = "en") -> tuple[str | None, str | None]:
    try:
        import speech_recognition as sr
    except ImportError:
        return None, "SpeechRecognition is not installed."
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=15)
        text = recognizer.recognize_google(audio, language=SPEECH_LOCALES.get(language, "en-IN"))
        return text, None
    except Exception as exc:
        return None, f"Voice input failed: {exc}"

def speak(text: str) -> tuple[bool, str | None]:
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 160)
        engine.say(text)
        engine.runAndWait()
        return True, None
    except Exception as exc:
        return False, f"Text-to-speech failed: {exc}"
