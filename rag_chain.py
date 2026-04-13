"""
rag_chain.py
============
LangChain RAG chain wrapping local Gemma 4 via Ollama.
Used by the FastAPI backend for grounded, offline triage responses.

Usage:
    from rag_chain import ASHARagChain
    chain = ASHARagChain()
    result = chain.query("Child has fever and fast breathing")
    print(result["answer"])
    print(result["sources"])
"""

from pathlib import Path
from typing import Optional
import chromadb
from chromadb.utils import embedding_functions
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser


SYSTEM_INSTRUCTION = (
    "You are ASHA-AI, a trusted clinical decision-support assistant for ASHA "
    "(Accredited Social Health Activist) workers in rural India. "
    "Your guidance is based strictly on IMNCI protocols and ASHA training manuals. "
    "Always prioritise patient safety. When uncertain, advise referral. "
    "Use clear, numbered, step-by-step language."
)

TRIAGE_PROMPT = PromptTemplate(
    input_variables=["context", "question", "language_note"],
    template=(
        "{system}\n\n"
        "Relevant ASHA clinical guidelines:\n{context}\n\n"
        "ASHA worker's question: {question}\n\n"
        "{language_note}\n"
        "Give a numbered step-by-step response. Start with the most urgent action first.\n"
        "If this is an emergency, say EMERGENCY clearly in the first line.\n\n"
        "Response:"
    ).replace("{system}", SYSTEM_INSTRUCTION),
)

EMERGENCY_KEYWORDS = [
    "emergency", "call 108", "call 102", "refer immediately",
    "without delay", "do not delay", "at once", "life-threatening",
    "ತಕ್ಷಣ", "ತುರ್ತು",
]

REFER_KEYWORDS = [
    "refer", "phc", "chc", "hospital", "clinic", "health centre",
    "nrc", "fru", "sub-centre",
]


class ASHARagChain:
    def __init__(
        self,
        chroma_dir: str = "data/chroma_db",
        ollama_model: str = "gemma3:4b",
        ollama_url: str = "http://localhost:11434",
        n_retrieve: int = 3,
    ):
        # ── Vector store ─────────────────────────────────────────────────────
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        client = chromadb.PersistentClient(path=chroma_dir)
        self.collection = client.get_or_create_collection(
            "asha_knowledge", embedding_function=embed_fn
        )
        self.n_retrieve = n_retrieve

        # ── LLM (local Gemma 4 via Ollama) ───────────────────────────────────
        self.llm = Ollama(
            model=ollama_model,
            base_url=ollama_url,
            temperature=0.2,
            num_predict=512,
        )

        # ── Chain ────────────────────────────────────────────────────────────
        self.chain = TRIAGE_PROMPT | self.llm | StrOutputParser()

    def retrieve(self, query: str) -> tuple[str, list[str]]:
        results = self.collection.query(
            query_texts=[query], n_results=self.n_retrieve
        )
        docs    = results["documents"][0]
        sources = [m.get("title", m.get("source", "ASHA Manual"))
                   for m in results["metadatas"][0]]
        context = "\n\n".join(
            f"[{src}]: {doc}" for src, doc in zip(sources, docs)
        )
        return context, sources

    def classify(self, response: str) -> str:
        r = response.lower()
        if any(k in r for k in EMERGENCY_KEYWORDS):
            return "refer_immediately"
        if any(k in r for k in REFER_KEYWORDS):
            return "refer"
        return "home_management"

    def query(self, question: str, language: str = "en") -> dict:
        context, sources = self.retrieve(question)
        lang_note = (
            "Respond in Kannada (ಕನ್ನಡ)." if language == "kn"
            else "Respond in clear, simple English."
        )
        answer = self.chain.invoke({
            "context": context,
            "question": question,
            "language_note": lang_note,
        })
        return {
            "answer": answer,
            "sources": sources,
            "confidence": self.classify(answer),
            "language": language,
        }


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    chain = ASHARagChain()

    test_cases = [
        ("A 2-year-old has fever for 3 days with very fast breathing and chest in-drawing.", "en"),
        ("Pregnant woman has severe headache and swollen face at 8 months.", "en"),
        ("Child has mild cold, no fever, eating and playing normally.", "en"),
    ]

    for question, lang in test_cases:
        print(f"\n{'='*60}")
        print(f"Q: {question}")
        result = chain.query(question, language=lang)
        print(f"[{result['confidence'].upper()}]")
        print(result["answer"])
        print(f"Sources: {', '.join(result['sources'])}")
