from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

docs = [
    "Infosys partnered with Microsoft",
    "Microsoft operates in Bengaluru",
    "Infosys provides consulting"
]

embeddings = model.encode(docs)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Query
query = "Who partnered with Microsoft?"
q_embed = model.encode([query])

distances, indices = index.search(np.array(q_embed), k=2)

print("Top matches:")
for i in indices[0]:
    print(docs[i])