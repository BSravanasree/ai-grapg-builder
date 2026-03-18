import streamlit as st
from neo4j import GraphDatabase
from pyvis.network import Network
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


# ---------- RAG SETUP ----------
model = SentenceTransformer('all-MiniLM-L6-v2')

documents = [
    "Infosys partnered with Microsoft.",
    "Microsoft operates cloud services in Bengaluru.",
    "Infosys provides consulting services."
]

embeddings = model.encode(documents)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))


def semantic_search(query):
    q_embed = model.encode([query])
    distances, indices = index.search(np.array(q_embed), k=2)
    return [documents[i] for i in indices[0]]


def generate_answer(query):
    context = semantic_search(query)
    return "Based on enterprise data: " + " ".join(context)


# ---------- NEO4J SETUP ----------
uri = "neo4j://127.0.0.1:7687"
username = "neo4j"
password = "neo4j123"  # CHANGE if needed

driver = GraphDatabase.driver(uri, auth=(username, password))


def fetch_graph():
    try:
        with driver.session() as session:
            result = session.run("""
                MATCH (a)-[r]->(b)
                RETURN a.name AS source, type(r) AS rel, b.name AS target
            """)
            return result.data()
    except Exception as e:
        st.error("Neo4j connection failed. Is database running?")
        return []


# ---------- STREAMLIT UI ----------

st.title("🧠 AI Knowledge Graph Builder Dashboard")

# 🔍 RAG Search
st.header("Ask Enterprise Question")

query = st.text_input("Enter your question:")

if st.button("Get Answer"):
    answer = generate_answer(query)
    st.success(answer)


# 🌐 Graph Visualization
st.header("Knowledge Graph Visualization")

if st.button("Load Graph"):
    data = fetch_graph()

    net = Network(height="500px", width="100%", bgcolor="#222", font_color="white")

    for row in data:
        net.add_node(row["source"], label=row["source"])
        net.add_node(row["target"], label=row["target"])
        net.add_edge(row["source"], row["target"], label=row["rel"])

    net.save_graph("graph.html")

    HtmlFile = open("graph.html", "r", encoding="utf-8")
    st.components.v1.html(HtmlFile.read(), height=500)