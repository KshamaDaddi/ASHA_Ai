# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ASHA-AI  |  app.py
# Gradio UI — three tabs: Text Triage · Voice Triage · Medicine Image Analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import gradio as gr
from PIL import Image

from .config import ALL_LANGS
from .triage import safe_full_triage
from .speech import transcribe_audio
from .ocr import analyse_image

# ── Pre-fill examples ─────────────────────────────────────────────────────────

TEXT_EXAMPLES = [
    ["A 2-year-old has fever for 3 days, very fast breathing, chest in-drawing.", "Kannada"],
    ["Pregnant woman at 8 months has severe headache and swollen face.",          "Hindi"],
    ["Newborn baby not feeding at 2 days old.",                                   "Telugu"],
    ["Child MUAC is 10.5 cm with swollen feet on both sides.",                   "Tamil"],
    ["Elderly man has chest pain going to left arm with sweating.",               "Marathi"],
    ["Child has mild cold, no fever, eating well.",                               "English"],
    ["Which vaccines for a 6-week-old baby?",                                     "Bengali"],
    ["How do I identify a malaria case in my village?",                           "Gujarati"],
]

MED_EXAMPLES = [
    ["What is this medicine and the correct dose for a child?", "Kannada"],
    ["Is this safe for a pregnant woman?",                      "Hindi"],
    ["How many times a day should this be taken?",              "Telugu"],
    ["What are the side effects?",                              "Tamil"],
]

# ── Tab handlers ──────────────────────────────────────────────────────────────

def tab_text(query: str, lang: str):
    """Handle text triage tab."""
    return safe_full_triage(query, lang)


def tab_voice(audio_path: str, in_lang: str, out_lang: str):
    """Handle voice triage tab: ASR → triage → TTS."""
    if audio_path is None:
        return "", "Please record audio first.", "Waiting", None

    try:
        transcribed = transcribe_audio(audio_path, language=in_lang).get("text", "")
        if not transcribed:
            return "", "Could not understand. Please try again.", "Failed", None

        resp, status, audio = safe_full_triage(transcribed, out_lang)
        return transcribed, resp, status, audio

    except Exception as e:
        return "", f"Error: {e}", "Failed", None


def tab_image(image, question: str, lang: str):
    """Handle medicine image tab: OCR → Gemma → translation → TTS."""
    if image is None:
        return "Please upload a medicine image.", "Waiting", None

    try:
        resp = analyse_image(image, question, lang)
        r    = str(resp).lower()

        if any(k in r for k in ["emergency", "poison", "toxic", "overdose"]):
            status = "🚨 EMERGENCY"
        elif any(k in r for k in ["doctor", "refer", "prescription"]):
            status = "⚠️ Consult doctor"
        else:
            status = "✅ General info"

        from .speech import speak
        audio = speak(str(resp)[:400], lang)
        return resp, status, audio

    except Exception as e:
        return f"Error: {e}", "❌ Failed", None


# ── Gradio layout ─────────────────────────────────────────────────────────────

CSS = (
    "footer{display:none!important} "
    ".status textarea{font-size:16px!important;font-weight:700!important}"
)


def build_demo() -> gr.Blocks:
    with gr.Blocks(css=CSS) as demo:

        gr.Markdown(
            "# 🏥 ASHA-AI — Offline Multilingual Health Triage Assistant\n"
            "_Empowering ASHA workers with AI-powered triage guidance in 11 Indian languages_"
        )

        with gr.Tabs():

            # ── TAB 1: TEXT TRIAGE ──────────────────────────────────────────
            with gr.TabItem("📝 Text Triage"):
                gr.Markdown("Enter the patient's symptoms. Receive step-by-step triage guidance.")
                with gr.Row():
                    with gr.Column(scale=1):
                        t1_lang   = gr.Dropdown(choices=ALL_LANGS, value="English",
                                                label="Response Language")
                        t1_status = gr.Textbox(label="Triage Decision",
                                               value="Waiting…", interactive=False,
                                               elem_classes=["status"])
                        t1_audio  = gr.Audio(label="Audio Response", type="filepath")

                    with gr.Column(scale=2):
                        t1_query = gr.Textbox(label="Patient Symptoms", lines=4,
                                              placeholder="Describe symptoms here…")
                        t1_btn   = gr.Button("🔍 Get Guidance", variant="primary")
                        t1_resp  = gr.Textbox(label="ASHA-AI Response", lines=12)

                t1_btn.click(
                    tab_text,
                    inputs=[t1_query, t1_lang],
                    outputs=[t1_resp, t1_status, t1_audio],
                )
                gr.Examples(TEXT_EXAMPLES, inputs=[t1_query, t1_lang])

            # ── TAB 2: VOICE TRIAGE ─────────────────────────────────────────
            with gr.TabItem("🎙️ Voice Triage"):
                gr.Markdown("Record symptoms in any language — receive guidance back in your chosen language.")
                with gr.Row():
                    with gr.Column(scale=1):
                        t2_in_lang   = gr.Dropdown(["auto"] + ALL_LANGS, value="auto",
                                                   label="Spoken Language (auto-detect)")
                        t2_out_lang  = gr.Dropdown(ALL_LANGS, value="English",
                                                   label="Response Language")
                        t2_status    = gr.Textbox(label="Status", value="Waiting…",
                                                  elem_classes=["status"])
                        t2_audio_out = gr.Audio(label="Audio Response", type="filepath")

                    with gr.Column(scale=2):
                        t2_audio_in  = gr.Audio(type="filepath", label="Record Symptoms")
                        t2_btn       = gr.Button("🎙️ Process Voice", variant="primary")
                        t2_transcrib = gr.Textbox(label="Transcription")
                        t2_resp      = gr.Textbox(label="ASHA-AI Response", lines=10)

                t2_btn.click(
                    tab_voice,
                    inputs=[t2_audio_in, t2_in_lang, t2_out_lang],
                    outputs=[t2_transcrib, t2_resp, t2_status, t2_audio_out],
                )

            # ── TAB 3: MEDICINE IMAGE ───────────────────────────────────────
            with gr.TabItem("💊 Medicine Image"):
                gr.Markdown("Upload a photo of a medicine label — ASHA-AI will explain it in simple terms.")
                with gr.Row():
                    with gr.Column(scale=1):
                        t3_lang    = gr.Dropdown(ALL_LANGS, value="English",
                                                 label="Response Language")
                        t3_question = gr.Textbox(
                            label="Your Question",
                            placeholder="e.g. Is this safe for a pregnant woman?",
                        )
                        t3_btn     = gr.Button("🔬 Analyse", variant="primary")
                        t3_status  = gr.Textbox(label="Status", value="Waiting…",
                                                elem_classes=["status"])
                        t3_audio   = gr.Audio(type="filepath", label="Audio Explanation")

                    with gr.Column(scale=2):
                        t3_img     = gr.Image(type="pil", label="Medicine Label Photo")
                        t3_resp    = gr.Textbox(label="Explanation", lines=12)

                t3_btn.click(
                    tab_image,
                    inputs=[t3_img, t3_question, t3_lang],
                    outputs=[t3_resp, t3_status, t3_audio],
                )
                gr.Examples(MED_EXAMPLES, inputs=[t3_question, t3_lang])

        gr.Markdown(
            "---\n"
            "⚠️ **Disclaimer**: ASHA-AI is a decision-support tool only. "
            "Always follow official protocols and refer when in doubt."
        )

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

def launch(share: bool = True, debug: bool = False):
    """Build and launch the Gradio app."""
    demo = build_demo()
    print("🚀 Launching ASHA-AI…")
    demo.launch(share=share, debug=debug)


if __name__ == "__main__":
    launch()
