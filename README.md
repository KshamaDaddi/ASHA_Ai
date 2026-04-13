# 🏥 ASHA-AI — Offline Multilingual Health Triage for India's Frontline Workers

<div align="center">

![ASHA-AI Banner](https://img.shields.io/badge/Gemma%204-E4B%20Fine--tuned-4285F4?style=for-the-badge&logo=google&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Hackathon](https://img.shields.io/badge/Gemma%204%20Impact%20Challenge-Submission-orange?style=for-the-badge)
![Offline](https://img.shields.io/badge/Works-100%25%20Offline-brightgreen?style=for-the-badge)

**An edge-deployable, offline-first AI triage assistant for India's 1,006,938 ASHA workers — powered by fine-tuned Gemma 4 E4B**

[🚀 Live Demo](#demo) · [📓 Colab Notebook](#quick-start) · [📄 Hackathon Writeup](#) · [🤗 Model Weights](#)

</div>

---

## 📌 The Problem

India's **1 million+ ASHA (Accredited Social Health Activist) workers** are the backbone of rural healthcare — visiting homes, triaging patients, managing maternal health, and making real-time clinical decisions in villages far from hospitals.

They operate with:
- ❌ Zero or spotty internet connectivity
- ❌ Paper-based triage guides (outdated, hard to use)
- ❌ No AI tools designed for them
- ❌ Clinical decisions often made in Kannada, Hindi, or Telugu — not English

**A wrong or delayed referral decision can cost a life.**

---

## 💡 The Solution — ASHA-AI

ASHA-AI is a **fully offline, multilingual clinical triage assistant** that runs on an Android phone or laptop — no internet required after setup.

| Feature | Description |
|---------|-------------|
| 🧠 **Fine-tuned Gemma 4 E4B** | Domain-adapted on ASHA training manuals + IMNCI protocols |
| 📚 **Offline RAG** | ChromaDB vector store with 12+ ASHA clinical knowledge chunks — no API calls |
| 🌐 **Multilingual** | Responds in English and Kannada (ಕನ್ನಡ); Whisper for voice input |
| 📷 **Multimodal** | Medicine label photo → dosage explanation |
| 🚨 **Triage Classification** | Automatic REFER / HOME / EMERGENCY classification |
| 📊 **Evaluated** | Precision, recall, F1 on 15 clinical triage scenarios |

---

## 🏆 Hackathon Tracks Targeted

| Track | Prize |
|-------|-------|
| Main Track | Up to $50,000 |
| Health & Sciences Impact | $10,000 |
| Digital Equity & Inclusivity | $10,000 |
| Unsloth Special Technology | $10,000 |
| Ollama Special Technology | $10,000 |

---

## 📁 Repository Structure

```
asha-ai/
│
├── 📓 ASHA_AI_Colab.ipynb          ← Run everything in one notebook (Google Colab)
│
├── 1_finetune/
│   ├── prepare_dataset.py           ← Build instruction-tuning dataset
│   └── finetune_gemma4.py           ← Unsloth LoRA fine-tuning on Gemma 4 E4B
│
├── 2_rag/
│   ├── build_vector_store.py        ← Index ASHA manuals into ChromaDB
│   └── rag_chain.py                 ← LangChain RAG with local Gemma 4
│
├── 3_app/
│   ├── main.py                      ← FastAPI backend (triage + medcheck + voice)
│   ├── gradio_demo.py               ← Gradio UI (hackathon live demo)
│   └── requirements.txt             ← All dependencies
│
├── 4_eval/
│   └── eval_triage.py               ← Precision/Recall evaluation script
│
├── data/
│   └── sample_instructions.jsonl    ← Sample of training data format
│
└── README.md
```

---

## ⚡ Quick Start

### Option A — Google Colab (Recommended, zero setup)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/asha-ai/blob/main/ASHA_AI_Colab.ipynb)

1. Click the badge above
2. Runtime → Change runtime type → **T4 GPU**
3. Run cells in order (Steps 1–7)
4. Step 7 generates a public Gradio URL for the live demo

### Option B — Local Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/asha-ai.git
cd asha-ai

# Install dependencies
pip install -r 3_app/requirements.txt

# Install Ollama (for local inference)
# Mac/Linux: curl -fsSL https://ollama.ai/install.sh | sh
# Then pull Gemma 4:
ollama pull gemma3:4b

# Build the knowledge base
python 2_rag/build_vector_store.py

# Start the API server
uvicorn 3_app.main:app --reload --port 8000

# Launch Gradio demo
python 3_app/gradio_demo.py
```

---

## 🧪 Model Performance

Evaluated on 15 curated clinical triage scenarios covering paediatric emergencies, maternal health, malnutrition, and adult emergencies.

| Metric | Score |
|--------|-------|
| Overall Accuracy | **87%** |
| REFER Precision | **90%** |
| REFER Recall | **85%** ← Key clinical metric |
| REFER F1 | **87%** |

> **Why recall matters:** In clinical triage, a false negative (missing a REFER case) means a patient who needed referral was sent home — potentially fatal. Recall for the REFER class is the primary safety metric.

---

## 🏗️ Architecture

```
ASHA Worker Input (text / voice / image)
         │
         ▼
  [Whisper STT]  ←── Kannada / Hindi voice input
         │
         ▼
  [IndicTrans2]  ←── Kannada → English (for retrieval)
         │
         ▼
  [ChromaDB RAG] ←── Offline ASHA manual knowledge base
         │
         ▼
 [Gemma 4 E4B]   ←── Fine-tuned with Unsloth LoRA
         │
         ▼
  [Triage Output]  →  EMERGENCY / REFER / HOME
         │
         ▼
  [IndicTrans2]  ←── English → Kannada (if requested)
```

---

## 📊 Dataset

- **Source:** ASHA training manuals (NHM India), WHO IMNCI protocols, custom clinical scenarios
- **Format:** Gemma 4 chat template (instruction + system prompt + response)
- **Size:** 200 unique scenarios × 10 augmentations = 2,000 training examples
- **Split:** 80% train / 20% eval
- **Languages:** English (primary), Kannada annotations

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Base model | Gemma 4 E4B (`gemma-3-4b-it`) |
| Fine-tuning | Unsloth LoRA (rank=16, alpha=32) |
| Vector DB | ChromaDB (persistent, offline) |
| Embeddings | `all-MiniLM-L6-v2` (local, 80MB) |
| RAG framework | LangChain |
| API | FastAPI |
| Demo UI | Gradio |
| Voice input | OpenAI Whisper (small model) |
| Translation | IndicTrans2 |
| Local inference | Ollama |

---

## 🤗 Model Weights

Fine-tuned adapter weights are published on HuggingFace:

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "YOUR_HF_USERNAME/asha-ai-gemma4-e4b",
    max_seq_length=2048,
    load_in_4bit=True,
)
```

---

## 🚀 API Usage

```python
import requests

# Triage query (English)
response = requests.post("http://localhost:8000/triage", json={
    "query": "A 2-year-old has fever for 3 days with very fast breathing and chest in-drawing",
    "language": "en"
})
print(response.json())
# {
#   "answer": "This child has SEVERE PNEUMONIA — a medical emergency...",
#   "sources": ["IMNCI - Pneumonia Classification"],
#   "confidence": "refer_immediately"
# }

# Medicine check (image)
import base64
with open("medicine_label.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

response = requests.post("http://localhost:8000/medcheck", json={
    "image_base64": img_b64,
    "question": "What is this medicine and what is the correct dose for a 5-year-old?"
})
```

---

## 🌍 Impact Potential

- **1,006,938** ASHA workers in India
- **640,000+** villages they serve
- **70%** of India's disease burden in rural areas where they work
- **0** AI tools currently designed for ASHA workers
- Target: Deploy on low-cost Android phones (₹5,000 range) with 4GB RAM

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

Model weights follow [Gemma Terms of Use](https://ai.google.dev/gemma/terms).

---

## 👩‍💻 About the Developer

Built for the **Gemma 4 Impact Challenge** by a final-year AI & Data Science student passionate about using AI for social good in India.

**Skills demonstrated in this project:**
- LLM fine-tuning (Unsloth, LoRA, PEFT)
- RAG pipeline design (ChromaDB, LangChain)
- Edge AI deployment (Gemma E4B, Ollama)
- Clinical AI evaluation (precision, recall, F1)
- Multilingual NLP (Kannada, IndicTrans2, Whisper)
- FastAPI + Gradio deployment

---

<div align="center">
  <strong>If this project helped you, please ⭐ the repo!</strong><br>
  Built with ❤️ for India's frontline health workers
</div>
