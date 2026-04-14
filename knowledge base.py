# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ASHA-AI  |  knowledge_base.py
# Build and query the ChromaDB vector store from ASHA protocol chunks
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from __future__ import annotations

import chromadb
from chromadb.utils import embedding_functions

from .config import CHROMA_PATH, COLLECTION, EMBED_MODEL, KNOWLEDGE_BASE, RAG_TOP_K

# Module-level singleton so the collection is loaded only once per process
_collection = None


def get_collection(chroma_path: str = CHROMA_PATH, collection_name: str = COLLECTION):
    """Return (and lazily initialise) the persistent ChromaDB collection."""
    global _collection
    if _collection is not None:
        return _collection

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    client = chromadb.PersistentClient(path=chroma_path)

    # Create collection if it doesn't exist; otherwise open existing
    try:
        col = client.get_collection(
            name=collection_name,
            embedding_function=embed_fn,
        )
        print(f"✅ Existing knowledge base loaded: {col.count()} chunks")
    except Exception:
        col = client.create_collection(
            name=collection_name,
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
        _index_knowledge_base(col)

    _collection = col
    return _collection


def _index_knowledge_base(collection) -> None:
    """Upsert all KNOWLEDGE_BASE chunks into the collection."""
    docs   = [text  for _, text  in KNOWLEDGE_BASE]
    ids    = [f"chunk_{i}" for i in range(len(KNOWLEDGE_BASE))]
    metas  = [{"title": title} for title, _ in KNOWLEDGE_BASE]

    collection.upsert(documents=docs, ids=ids, metadatas=metas)
    print(f"✅ Knowledge base indexed: {collection.count()} chunks")


def rebuild_knowledge_base(
    chroma_path: str = CHROMA_PATH,
    collection_name: str = COLLECTION,
) -> None:
    """
    Drop and re-create the collection from scratch.
    Useful when KNOWLEDGE_BASE content is updated.
    """
    global _collection

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    client = chromadb.PersistentClient(path=chroma_path)

    try:
        client.delete_collection(collection_name)
        print(f"🗑  Old collection '{collection_name}' deleted.")
    except Exception:
        pass

    col = client.create_collection(
        name=collection_name,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )
    _index_knowledge_base(col)
    _collection = col


def retrieve_context(query: str, n: int = RAG_TOP_K) -> str:
    """
    Retrieve the top-n most relevant ASHA manual chunks for a query.

    Returns a formatted string ready to inject into the LLM prompt.
    """
    col = get_collection()
    results = col.query(query_texts=[query], n_results=n)
    parts = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        parts.append(f"[{meta['title']}]: {doc}")
    return "\n\n".join(parts)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rebuild_knowledge_base()
    test_query = "child with fever and fast breathing"
    ctx = retrieve_context(test_query)
    print(f"\nTest retrieval for: '{test_query}'")
    print(ctx[:400], "...")
