# 🏥 ASHA-AI — Offline Multilingual Health Triage Assistant for Rural India

<p align="center">
  <img src="https://img.shields.io/badge/Model-Gemma%203%204B-blue?style=for-the-badge&logo=google" />
  <img src="https://img.shields.io/badge/Fine--tuned%20with-Unsloth%20%2B%20LoRA-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/RAG-ChromaDB%20%2B%20SentenceTransformers-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Languages-12%20Indian%20Languages-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Interface-Gradio-yellow?style=for-the-badge" />
</p>

<p align="center">
  <b>AI-powered health triage for ASHA workers — works offline, speaks your language, and knows when to refer.</b>
</p>

---

## 📖 Overview

**ASHA-AI** is an offline-first, multilingual health triage assistant purpose-built for **ASHA (Accredited Social Health Activist)** workers in rural India. These frontline health workers serve in areas with poor connectivity, limited training resources, and high patient load — often making critical triage decisions alone.

ASHA-AI gives them an intelligent second opinion — in their own language — powered by a fine-tuned **Gemma 3 4B** model augmented with a **Retrieval-Augmented Generation (RAG)** pipeline seeded with IMNCI protocols and ASHA training manuals. The system classifies cases as `🚨 EMERGENCY`, `⚠️ REFER`, or `✅ Home Management` with step-by-step numbered guidance, voice output, and medicine-label analysis.

---

## 🎯 Problem Statement

India's 1 million+ ASHA workers are the last line of healthcare in rural communities. They face:

- **No internet connectivity** in many villages
- **Language barriers** — most AI tools are English-only
- **Life-critical decisions** with minimal on-call support
- **Lack of accessible** IMNCI/clinical protocol lookup tools

ASHA-AI addresses all four with a single, deployable application.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🗣️ **12 Indian Languages** | Kannada, Hindi, Telugu, Tamil, Marathi, Bengali, Gujarati, Malayalam, Punjabi, Odia, Assamese, Urdu |
| 🔇 **Offline-First** | IndicTrans2 for offline translation; Whisper for on-device speech recognition |
| 🎙️ **Voice Triage** | Speak symptoms → receive spoken triage guidance |
| 💊 **Medicine Image Analysis** | OCR a medicine label → get dosage, usage, and safety info |
| 📋 **RAG-Augmented Responses** | ChromaDB + all-MiniLM-L6-v2 retrieves relevant ASHA manual sections |
| ⚡ **4-bit Quantisation** | Runs on a single T4 GPU via Unsloth + bitsandbytes |
| 🔊 **Text-to-Speech** | gTTS outputs audio responses in the selected language |

---

## 🧠 Model Architecture

```
User Input (text / voice / image)
        │
        ▼
 Language Detection & Translation (IndicTrans2 / Google Translate fallback)
        │
        ▼
 RAG Retrieval ──── ChromaDB (all-MiniLM-L6-v2 embeddings)
        │               └── ASHA manual chunks (ANC, IMNCI, immunisation, etc.)
        ▼
 Fine-tuned Gemma 3 4B (Unsloth + LoRA, 4-bit)
        │
        ▼
 Triage Classification  →  🚨 EMERGENCY / ⚠️ REFER / ✅ Home Management
        │
        ▼
 Translation (→ requested Indian language) + gTTS Audio Output
```

---

## 🗂️ Clinical Coverage

The knowledge base and training data cover the following health domains based on official IMNCI protocols and ASHA training manuals:

**Paediatric Emergencies**
- Severe pneumonia, fast breathing, chest in-drawing
- Febrile convulsions, general danger signs
- Severe and moderate dehydration
- Malnutrition (SAM/MAM via MUAC classification)

**Newborn Care**
- Danger signs in neonates (feeding refusal, jaundice, temperature)
- Pathological jaundice (< 24 hours)
- Routine newborn care protocol

**Maternal Health**
- Pre-eclampsia and eclampsia danger signs
- Antepartum haemorrhage
- ANC registration and missed visits
- Postnatal depression with suicidal ideation

**Adult Emergencies**
- Suspected myocardial infarction (heart attack)
- Stroke recognition and time-critical referral
- Snake bite first aid

**Prevention & Immunisation**
- Universal Immunisation Programme (UIP) schedule
- Malaria RDT testing and ACT treatment
- Fever differentiation (malaria, dengue, typhoid)
- JSY/JSSK scheme guidance

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Base Model | `unsloth/gemma-3-4b-it-unsloth-bnb-4bit` |
| Fine-tuning | Unsloth + PEFT (LoRA, r=16) + TRL SFTTrainer |
| Quantisation | bitsandbytes 4-bit |
| Vector Store | ChromaDB (persistent, local) |
| Embeddings | `all-MiniLM-L6-v2` (SentenceTransformers) |
| ASR | OpenAI Whisper (`base`, offline) |
| OCR | Tesseract (multi-script: eng + Indian scripts) |
| Translation | IndicTrans2 (offline) / `deep_translator` (fallback) |
| TTS | gTTS |
| UI | Gradio (`gr.Blocks`) |
| Evaluation | scikit-learn (precision, recall, F1) |

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10+
- CUDA-enabled GPU (T4 or better recommended)
- Google Colab (recommended for first run)

### Quick Start (Google Colab)

**Step 1 — Install dependencies**
```bash
pip install -q unsloth transformers accelerate peft datasets trl bitsandbytes
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" -q
pip install chromadb sentence-transformers langchain langchain-community -q
pip install gradio scikit-learn pymupdf gtts openai-whisper pytesseract -q
```

**Step 2 — Run the notebook**

Open `asha_ai.py` in Google Colab and run cells sequentially:

1. **Cell 1** — Install packages
2. **Cell 2** — Build training dataset (IMNCI Q&A pairs)
3. **Cell 3** — Load Gemma 3 4B + add LoRA adapters
4. **Cell 4** — Fine-tune with SFTTrainer (~60–90 min on T4)
5. **Cell 5** — Build ChromaDB knowledge base
6. **Cell 6** — RAG inference tests
7. **Cell 7** — Run evaluation on 15 triage scenarios
8. **Cell 8** — Setup TTS, ASR, OCR, translation
9. **Cell 10** — Launch Gradio demo

> 💡 **After fine-tuning completes**, the LoRA adapter is saved to `./asha_ai_adapter/`. Subsequent runs can load directly from this path.

---

## 🖥️ Gradio Interface

The app launches with three tabs:

### Tab 1 — Text Triage
Enter symptoms in any language → receive numbered step-by-step triage guidance + audio playback.

### Tab 2 — Voice Triage
Record symptoms in any of 12 languages → auto-transcribed by Whisper → response returned in chosen output language with audio.

### Tab 3 — Medicine Image
Upload a photo of a medicine label → OCR extracts text → Gemma explains dosage, uses, and safety warnings in the worker's language.

---

## 📊 Evaluation Results

Evaluated on **15 real-world triage scenarios** (10 REFER, 5 HOME) using keyword-based classification:

| Metric | Score |
|---|---|
| Overall Accuracy | ≥ 86% |
| REFER Precision | High |
| **REFER Recall** | **~100%** ← Primary safety metric |
| REFER F1 | High |

> **Why REFER recall is the key metric:** A missed referral (false HOME) can cost a patient's life. The system is intentionally conservative — it is better to over-refer than to miss a critical case.

Results are saved to `data/eval_results.json` for use in the hackathon submission writeup.

---

## 🗺️ Project Structure

```
asha-ai/
├── asha_ai.py               # Main notebook (all cells)
├── data/
│   ├── train.jsonl          # Fine-tuning dataset (20 × 10 examples)
│   ├── eval.jsonl           # Evaluation split
│   ├── chroma_db/           # Persistent ChromaDB vector store
│   └── eval_results.json    # Triage evaluation metrics
├── asha_ai_adapter/         # Saved LoRA adapter (post fine-tuning)
│   ├── adapter_config.json
│   └── adapter_model.safetensors
└── README.md
```

---

## 🏆 Hackathon Tracks Targeted

This project is submitted to the **Gemma 4 Impact Challenge** targeting:

- 🏥 **Health & Sciences** — clinical AI for underserved populations
- 🌍 **Digital Equity** — offline-first, multilingual access for rural India
- ⚡ **Unsloth Special Track** — efficient fine-tuning with Unsloth + 4-bit quantisation
- 🦙 **Ollama Special Track** — deployable via Ollama for fully local inference

---

## 🔒 Ethical Considerations

- ASHA-AI is a **decision-support tool**, not a replacement for medical professionals.
- All responses instruct workers to refer when uncertain — the system defaults to patient safety.
- No patient data is stored. All inference runs locally.
- Training data is based on publicly available IMNCI protocols and ASHA training manuals from the Government of India.

---

## 🚀 Future Roadmap

- [ ] Quantise and package as an Ollama model for fully local deployment on Android devices
- [ ] Expand training dataset with more disease conditions and rare presentations
- [ ] Add ASHA dashboard for logging case referrals
- [ ] Integrate with ABHA health ID for longitudinal patient tracking
- [ ] Community fine-tuning with anonymised real ASHA worker queries

---

## 👩‍💻 About the Author

**Kshama** — Final-year B.E. student in AI & Data Science, Karnataka  
Oracle OCI GenAI & DevOps Certified | Data Science Intern @ Take It Smart Pvt. Ltd.  
Passionate about building AI systems that solve real problems for real people.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://linkedin.com)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com)

---

## 📄 License

This project is released under the **MIT License**. See `LICENSE` for details.

The Gemma model is subject to [Google's Gemma Terms of Use](https://ai.google.dev/gemma/terms).

---

<p align="center">
  Built with ❤️ for India's frontline health heroes — the ASHA workers.
</p>
