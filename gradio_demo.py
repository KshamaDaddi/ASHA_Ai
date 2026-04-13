"""
gradio_demo.py
==============
Standalone Gradio demo for ASHA-AI.
Run this file to launch the live demo UI.

    python 3_app/gradio_demo.py

Requires Ollama running locally with Gemma 4:
    ollama pull gemma3:4b
    (or your fine-tuned model: ollama create asha-ai -f Modelfile)
"""

import gradio as gr
import httpx
import json
import chromadb
from chromadb.utils import embedding_functions

# ── Setup ─────────────────────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434"
OLLAMA_MODEL = "gemma3:4b"   # replace with "asha-ai" after fine-tuning

SYSTEM_PROMPT = (
    "You are ASHA-AI, a trusted health assistant for ASHA workers in rural India. "
    "Provide clear, numbered, step-by-step triage guidance based on IMNCI protocols. "
    "Always prioritise safety. When uncertain, advise referral. "
    "Use simple language. If it is an emergency, say EMERGENCY clearly first."
)

embed_fn   = embedding_functions.SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")
client     = chromadb.PersistentClient(path="data/chroma_db")
collection = client.get_or_create_collection("asha_knowledge", embedding_function=embed_fn)

EMERGENCY_KW = ["emergency", "call 108", "call 102", "refer immediately",
                "without delay", "do not delay", "at once"]
REFER_KW     = ["refer", "phc", "chc", "hospital", "nrc", "ambulance"]

EXAMPLES = [
    ["A 2-year-old has fever for 3 days, very fast breathing and chest in-drawing.", "English"],
    ["Pregnant woman at 8 months has severe headache, swollen face and blurred vision.", "English"],
    ["A newborn baby (2 days old) is not feeding and crying abnormally.", "English"],
    ["A child's MUAC is 10.5 cm with swollen feet on both sides.", "English"],
    ["An elderly man has chest pain spreading to left arm with heavy sweating.", "English"],
    ["Child has mild cold and runny nose for 1 day. No fever. Eating well.", "English"],
    ["Which vaccines should a 6-week-old baby receive at the sub-centre?", "English"],
    ["How do I identify and report a malaria case in my village?", "English"],
]


def get_status(response: str) -> tuple[str, str]:
    r = response.lower()
    if any(k in r for k in EMERGENCY_KW):
        return "🚨 EMERGENCY — Call 108 Now", "#dc2626"
    if any(k in r for k in REFER_KW):
        return "⚠️ REFER to health facility", "#d97706"
    return "🏠 Home management — Monitor closely", "#16a34a"


def triage(query: str, language: str, history: list) -> tuple[list, str, str]:
    if not query.strip():
        return history, "❓ Enter a question above", ""

    # Retrieve context
    results    = collection.query(query_texts=[query], n_results=2)
    docs       = results["documents"][0]
    metas      = results["metadatas"][0]
    sources    = [m.get("title", "ASHA Manual") for m in metas]
    context    = "\n\n".join(f"[{s}]: {d}" for s, d in zip(sources, docs))

    lang_note  = "Respond in Kannada (ಕನ್ನಡ)." if language == "Kannada" \
                 else "Respond in simple English."

    prompt = (
        f"Relevant ASHA guidelines:\n{context}\n\n"
        f"ASHA worker question: {query}\n\n"
        f"{lang_note} Numbered steps. Most urgent action first."
    )

    # Call Ollama
    try:
        resp = httpx.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "stream": False,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                "options": {"temperature": 0.2, "num_predict": 400},
            },
            timeout=60.0,
        )
        answer = resp.json()["message"]["content"]
    except Exception as e:
        answer = f"⚠️ Could not reach local model. Make sure Ollama is running.\nError: {e}"

    status_text, _ = get_status(answer)
    sources_text   = "📚 Sources: " + " · ".join(sources)
    full_answer    = f"{answer}\n\n{sources_text}"

    history.append((query, full_answer))
    return history, status_text, ""


def clear_chat():
    return [], "❓ Waiting for input", ""


# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="ASHA-AI",
    theme=gr.themes.Soft(primary_hue="emerald", neutral_hue="slate"),
    css="""
    .status-box textarea { font-size: 16px !important; font-weight: 600 !important; }
    footer { display: none !important; }
    """
) as demo:

    gr.HTML("""
    <div style='text-align:center; padding:24px 0 8px;'>
        <h1 style='font-size:2rem; color:#064e3b; margin:0;'>🏥 ASHA-AI</h1>
        <p style='color:#6b7280; margin:8px 0 0; font-size:15px;'>
            Offline-first multilingual health triage assistant · Powered by Gemma 4 E4B<br>
            <small>Fine-tuned on ASHA training manuals · Works without internet · English + Kannada</small>
        </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            language    = gr.Radio(
                choices=["English", "Kannada"],
                value="English",
                label="Response language",
            )
            status_out  = gr.Textbox(
                value="❓ Waiting for input",
                label="Triage decision",
                interactive=False,
                elem_classes=["status-box"],
            )
            gr.HTML("""
            <div style='background:#fef9c3; border:1px solid #eab308;
                        border-radius:8px; padding:10px; font-size:12px; color:#713f12;'>
                ⚠️ <strong>For trained ASHA workers only.</strong>
                This tool supports — not replaces — clinical judgment.
                For emergencies, always call 108.
            </div>
            """)

        with gr.Column(scale=2):
            chatbot     = gr.Chatbot(
                label="ASHA-AI Conversation",
                height=420,
                bubble_full_width=False,
            )
            with gr.Row():
                query_in    = gr.Textbox(
                    placeholder="Describe the patient's symptoms...",
                    show_label=False,
                    scale=5,
                    container=False,
                )
                submit_btn  = gr.Button("Ask →", variant="primary", scale=1)
                clear_btn   = gr.Button("Clear", scale=1)

    gr.Examples(
        examples=EXAMPLES,
        inputs=[query_in, language],
        label="Try these clinical scenarios:",
        examples_per_page=4,
    )

    # Events
    submit_btn.click(triage, [query_in, language, chatbot], [chatbot, status_out, query_in])
    query_in.submit(triage, [query_in, language, chatbot], [chatbot, status_out, query_in])
    clear_btn.click(clear_chat, [], [chatbot, status_out, query_in])


if __name__ == "__main__":
    print("🚀 Starting ASHA-AI Gradio demo...")
    print(f"   Model : {OLLAMA_MODEL} via Ollama at {OLLAMA_URL}")
    print(f"   KB    : {collection.count()} knowledge chunks loaded")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,        # generates public URL for hackathon submission
    )
