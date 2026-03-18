from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

documents = [
    "Infosys partnered with Microsoft.",
    "Microsoft operates cloud services in Bengaluru.",
    "Infosys provides consulting services."
]

# Step 1 — Create embeddings
embeddings = model.encode(documents)

# Step 2 — Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))


# 🔹 STEP 1 — Semantic Search
def semantic_search(query):
    q_embed = model.encode([query])
    distances, indices = index.search(np.array(q_embed), k=2)

    results = [documents[i] for i in indices[0]]
    return results


# 🔥 STEP 2 — ADD RAG FUNCTION HERE
def generate_answer(query):
    context = semantic_search(query)

    answer = f"Based on enterprise data: {' '.join(context)}"
    return answer


# 🔹 STEP 3 — TEST RAG
print(generate_answer("Who partnered with Microsoft?"))