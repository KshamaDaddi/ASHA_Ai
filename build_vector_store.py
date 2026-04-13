"""
build_vector_store.py
=====================
Ingests ASHA training manual PDFs and builds a local ChromaDB vector store
for offline retrieval-augmented generation (RAG).

This works 100% offline after first setup — no internet needed during app use.

pip install chromadb langchain-community pymupdf sentence-transformers
"""

import os
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ─── Config ───────────────────────────────────────────────────────────────────
PDF_DIR       = Path("data/manuals")          # Drop your ASHA PDFs here
CHROMA_DIR    = Path("data/chroma_db")        # Offline vector store
COLLECTION    = "asha_knowledge"
CHUNK_SIZE    = 512
CHUNK_OVERLAP = 64
EMBED_MODEL   = "all-MiniLM-L6-v2"           # ~80MB, works offline after download

# ─── 1. Embed function (local Sentence Transformers) ─────────────────────────
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL
)

# ─── 2. ChromaDB client ───────────────────────────────────────────────────────
client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = client.get_or_create_collection(
    name=COLLECTION,
    embedding_function=embed_fn,
    metadata={"hnsw:space": "cosine"},
)

# ─── 3. Ingest PDFs ───────────────────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " "],
)

# If you don't have PDFs yet, we seed with embedded text snippets
SEED_TEXT_CHUNKS = [
    ("ANC Protocol", 
     "Every pregnant woman should have at least 4 ANC visits. "
     "First visit before 12 weeks. Measure weight, blood pressure, haemoglobin. "
     "Give iron-folic acid tablets (IFA) from 12 weeks onwards. "
     "TT vaccination at 16 and 20 weeks. Register for JSY scheme."),
    ("Danger Signs in Pregnancy",
     "Refer immediately for: severe headache, blurred vision, swelling of face/hands, "
     "vaginal bleeding, severe abdominal pain, fever above 38°C, reduced fetal movements, "
     "fits or convulsions. These may indicate pre-eclampsia, eclampsia, or placenta praevia."),
    ("IMNCI Triage — Pneumonia",
     "Fast breathing in children: under 2 months >60/min, 2-12 months >50/min, "
     "1-5 years >40/min. Chest in-drawing = severe pneumonia. "
     "Stridor at rest = very severe. Refer all severe/very severe cases immediately. "
     "Give amoxicillin 40mg/kg/day in 2 doses for 5 days for non-severe pneumonia."),
    ("Dehydration Assessment",
     "Some dehydration: restless/irritable, sunken eyes, thirsty drinks eagerly. "
     "Give ORS 75ml/kg over 4 hours. Severe dehydration: lethargic, sunken eyes, "
     "skin pinch goes back very slowly, unable to drink. Refer immediately to hospital "
     "for IV rehydration. Continue breastfeeding throughout."),
    ("Newborn Care — First Week",
     "Keep newborn warm. Breastfeed within 1 hour of birth. Skin-to-skin contact. "
     "Do not bathe for 24 hours. Danger signs: not feeding, fast breathing, "
     "temperature below 35.5°C or above 37.5°C, yellow skin before 24 hours, "
     "bleeding, fits. Any danger sign = refer immediately."),
    ("Malnutrition — MUAC",
     "MUAC (Mid Upper Arm Circumference): Green ≥ 12.5cm (normal), "
     "Yellow 11.5-12.5cm (moderate acute malnutrition — MAM), "
     "Red < 11.5cm (severe acute malnutrition — SAM). "
     "SAM with complications = immediate NRC referral. "
     "SAM without complications = RUTF (Ready-to-Use Therapeutic Food) at home."),
    ("Immunisation Schedule",
     "At birth: BCG, OPV-0, Hepatitis B. 6 weeks: DPT-1, OPV-1, Hib, Rotavirus, PCV. "
     "10 weeks: DPT-2, OPV-2. 14 weeks: DPT-3, OPV-3, IPV. "
     "9 months: Measles/MR-1, VitA-1. 16-24 months: DPT-B1, OPV-B, MR-2, VitA-2. "
     "All vaccines are free at government facilities under UIP."),
    ("Vector-Borne Disease Prevention",
     "Malaria: use insecticide-treated bed nets (ITNs), "
     "drain stagnant water, report fever + chills + sweating to ASHA. "
     "RDT for malaria — treat with ACT if positive. "
     "Dengue: fever + rash + severe headache + pain behind eyes. "
     "No aspirin for dengue. Refer if platelet count drops or bleeding occurs."),
]

# Seed the vector store
docs_to_add = []
ids_to_add  = []
metas_to_add = []

for i, (title, text) in enumerate(SEED_TEXT_CHUNKS):
    docs_to_add.append(text)
    ids_to_add.append(f"seed_{i}")
    metas_to_add.append({"source": "ASHA Manual", "title": title})

# Also load PDFs if present
if PDF_DIR.exists():
    for pdf_path in PDF_DIR.glob("*.pdf"):
        print(f"Loading: {pdf_path.name}")
        loader = PyMuPDFLoader(str(pdf_path))
        pages  = loader.load()
        chunks = splitter.split_documents(pages)
        for j, chunk in enumerate(chunks):
            docs_to_add.append(chunk.page_content)
            ids_to_add.append(f"{pdf_path.stem}_{j}")
            metas_to_add.append({
                "source": pdf_path.name,
                "page": chunk.metadata.get("page", 0),
            })

# Batch upsert (ChromaDB max 5461 per batch)
BATCH = 500
for start in range(0, len(docs_to_add), BATCH):
    collection.upsert(
        documents=ids_to_add[start:start+BATCH],
        ids=ids_to_add[start:start+BATCH],
        metadatas=metas_to_add[start:start+BATCH],
    )
    # ChromaDB upsert takes documents separately
    # Fix: use correct API
    pass

# Correct upsert call
collection.upsert(
    documents=docs_to_add,
    ids=ids_to_add,
    metadatas=metas_to_add,
)

print(f"\nVector store ready: {collection.count()} chunks indexed at {CHROMA_DIR}")


# ─── 4. Quick retrieval test ──────────────────────────────────────────────────
results = collection.query(
    query_texts=["child with fever and fast breathing"],
    n_results=3,
)
print("\n--- Top 3 retrieved chunks for test query ---")
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"[{meta['title']}] {doc[:120]}...")
