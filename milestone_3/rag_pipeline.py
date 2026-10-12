"""
rag_pipeline.py — Semantic Search & RAG Pipeline (Milestone 3)

Implements a TRUE Retrieval-Augmented Generation pipeline:
  1. embed_documents()  — encode a list of text chunks into a FAISS index
  2. semantic_search()  — retrieve top-k relevant chunks for a query
  3. generate_answer()  — pass query + context to Gemini LLM → real answer
  4. full_rag_query()   — orchestrates all three steps end-to-end

No string concatenation masquerading as generation.
The answer comes from a real language model (Gemini 1.5 Flash).
"""

import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# ── Load environment variables ────────────────────────────────────────────────
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ── Embedding model (shared across calls) ────────────────────────────────────
_EMBED_MODEL = None


def _get_embed_model() -> SentenceTransformer:
    """Lazy-load the embedding model (cached after first call)."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBED_MODEL


# ── Gemini LLM setup ──────────────────────────────────────────────────────────
def _get_gemini_model():
    """
    Configure and return a Gemini GenerativeModel instance.
    Raises a clear error if the API key is missing.
    """
    if not GEMINI_API_KEY:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. "
            "Add it to your .env file: GEMINI_API_KEY=your-key-here"
        )
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel("gemini-1.5-flash")


# ════════════════════════════════════════════════════════════════════════════════
# STEP 1 — Embedding & Index Construction
# ════════════════════════════════════════════════════════════════════════════════

def embed_documents(documents: list) -> tuple:
    """
    Encode a list of text strings into dense vectors and build a FAISS index.

    Args:
        documents: List of text strings to index.

    Returns:
        (faiss_index, embeddings_array, model)
        Keep the returned index in memory or save it with faiss.write_index().
    """
    model = _get_embed_model()
    embeddings = model.encode(documents, show_progress_bar=False)
    embeddings_np = np.array(embeddings).astype("float32")

    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)

    return index, embeddings_np, model


def save_index(index, path: str = "kg_faiss.index"):
    """Persist the FAISS index to disk so it survives restarts."""
    faiss.write_index(index, path)
    print(f"[rag_pipeline] ✅ FAISS index saved to {path}")


def load_index(path: str = "kg_faiss.index"):
    """Load a previously saved FAISS index from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"FAISS index not found at '{path}'. Run embed_documents() first.")
    return faiss.read_index(path)


# ════════════════════════════════════════════════════════════════════════════════
# STEP 2 — Semantic Search (Retrieval)
# ════════════════════════════════════════════════════════════════════════════════

def semantic_search(query: str, documents: list, index, model, top_k: int = 3) -> list:
    """
    Find the top_k most semantically similar documents to the query.

    Args:
        query:     The user's natural language question.
        documents: The original list of text strings (same order as index).
        index:     FAISS index built by embed_documents().
        model:     SentenceTransformer model used to encode the query.
        top_k:     Number of results to retrieve.

    Returns:
        List of (document_text, distance_score) tuples, best match first.
    """
    q_embedding = model.encode([query])
    q_np = np.array(q_embedding).astype("float32")

    distances, indices = index.search(q_np, k=min(top_k, len(documents)))

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(documents):
            results.append((documents[idx], float(dist)))

    return results


# ════════════════════════════════════════════════════════════════════════════════
# STEP 3 — LLM Answer Generation (the G in RAG)
# ════════════════════════════════════════════════════════════════════════════════

def generate_answer(query: str, context_chunks: list) -> str:
    """
    Generate a grounded answer using the Gemini 1.5 Flash model.

    The model is instructed to answer ONLY from the provided context,
    preventing hallucination of information not in the enterprise data.

    Args:
        query:          The user's natural language question.
        context_chunks: List of retrieved text chunks from semantic_search().

    Returns:
        The LLM-generated answer string.
        On failure, returns a safe error message (never raises).
    """
    context_str = "\n".join(f"- {chunk}" for chunk in context_chunks)

    prompt = f"""You are an enterprise intelligence assistant analyzing financial transaction data.
Use ONLY the following context to answer the question.
If the answer is not in the context, say exactly: "I don't have enough information to answer that."

Context:
{context_str}

Question: {query}

Answer:"""

    try:
        model = _get_gemini_model()
        response = model.generate_content(prompt)
        return response.text.strip()
    except EnvironmentError as e:
        return f"LLM unavailable: {e}"
    except Exception as e:
        return f"LLM unavailable: {e}"


# ════════════════════════════════════════════════════════════════════════════════
# STEP 4 — Full RAG Orchestrator
# ════════════════════════════════════════════════════════════════════════════════

def full_rag_query(query: str, documents: list, top_k: int = 3) -> dict:
    """
    End-to-end RAG: embed corpus → retrieve context → generate answer.

    Args:
        query:     User's question.
        documents: Corpus of text strings to search over.
        top_k:     Number of context chunks to retrieve and feed to the LLM.

    Returns:
        dict with:
          "answer"  → str  — the LLM-generated answer
          "sources" → list — the retrieved context chunks (text only)
    """
    # Build or rebuild the index
    index, _, model = embed_documents(documents)

    # Retrieve
    search_results = semantic_search(query, documents, index, model, top_k=top_k)
    context_chunks = [text for text, _ in search_results]

    # Generate
    answer = generate_answer(query, context_chunks)

    return {
        "answer":  answer,
        "sources": context_chunks,
    }


# ── Self-test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_docs = [
        "TX_001 is a Fraud transaction of $4500.00. Risk level is High Risk. Merchant category is Electronics. Cardholder is USER_0.",
        "TX_002 is a Normal transaction of $120.50. Risk level is Low Risk. Merchant category is Grocery. Cardholder is USER_1.",
        "TX_003 is a Fraud transaction of $9800.00. Risk level is High Risk. Merchant category is Travel. Cardholder is USER_2.",
        "TX_004 is a Normal transaction of $55.00. Risk level is Low Risk. Merchant category is Food. Cardholder is USER_3.",
        "TX_005 is a Fraud transaction of $3000.00. Risk level is High Risk. Merchant category is Electronics. Cardholder is USER_4.",
    ]

    test_queries = [
        "Which transactions were flagged as fraud?",
        "What is the highest risk transaction amount?",
        "Which cardholder made a grocery purchase?",
    ]

    print("\n" + "=" * 60)
    print("  rag_pipeline.py — Self-Test")
    print("=" * 60)

    for query in test_queries:
        print(f"\n❓ Query: {query}")
        result = full_rag_query(query, sample_docs, top_k=3)
        print(f"💡 Answer: {result['answer']}")
        print("📚 Sources used:")
        for i, src in enumerate(result["sources"], 1):
            print(f"   [{i}] {src[:80]}...")