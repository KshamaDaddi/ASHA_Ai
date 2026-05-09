import speech_recognition as sr
import pyttsx3

# -----------------------------
# SPEECH TO TEXT
# -----------------------------
def listen():

    recognizer = sr.Recognizer()

    with sr.Microphone() as source:

        print("Listening...")

        recognizer.adjust_for_ambient_noise(source)

        audio = recognizer.listen(source)

    try:

        text = recognizer.recognize_google(audio)

        return text

    except:

        return None


# -----------------------------
# TEXT TO SPEECH
# -----------------------------
def speak(text):

    engine = pyttsx3.init()

    engine.setProperty('rate', 160)

    engine.say(text)

    engine.runAndWait()