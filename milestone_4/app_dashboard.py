"""
app_dashboard.py — Interactive Graph Dashboard (Milestone 4)

Security fixes applied in this version:
  - All Neo4j credentials loaded from .env via python-dotenv
  - No hardcoded passwords anywhere in this file
  - fetch_graph() imported from neo4j_loader (DRY — not re-implemented here)
"""

import os
import tempfile

import streamlit as st
import numpy as np
from pyvis.network import Network
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv

# ── Load .env ──────────────────────────────────────────────────────────────────
load_dotenv()

# ── Import project modules ─────────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "milestone_2"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "milestone_3"))

try:
    from neo4j_loader import fetch_graph, get_graph_stats, test_connection
    NEO4J_AVAILABLE = test_connection()
except Exception as e:
    NEO4J_AVAILABLE = False
    st.warning(f"Neo4j module import failed: {e}")

try:
    from rag_pipeline import full_rag_query, embed_documents
    RAG_AVAILABLE = True
except Exception as e:
    RAG_AVAILABLE = False


# ── RAG SETUP (inline fallback if rag_pipeline import fails) ──────────────────
@st.cache_resource
def load_rag_index(documents):
    """Build and cache the FAISS index so it isn't rebuilt on every interaction."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(documents)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return model, index


def semantic_search_local(query, documents, model, index, k=3):
    q_embed = model.encode([query])
    distances, indices = index.search(np.array(q_embed), k=k)
    return [documents[i] for i in indices[0] if i < len(documents)]


def generate_answer_local(query, context_chunks):
    """Fallback answer generation used if rag_pipeline.py is unavailable."""
    context = "\n".join(context_chunks)
    return f"**Retrieved context (LLM unavailable):**\n\n{context}"


# ── Sample documents (replaced by real data when pipeline is connected) ────────
SAMPLE_DOCUMENTS = [
    "Infosys partnered with Microsoft to deliver cloud analytics services.",
    "Microsoft operates cloud infrastructure services in Bengaluru, India.",
    "Infosys provides enterprise consulting and digital transformation services.",
    "A high-risk transaction of $4500 was flagged as potentially fraudulent.",
    "Cardholder USER_42 made 6 transactions within 24 hours — elevated velocity.",
]

# ── Streamlit Page Config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Knowledge Graph Builder",
    page_icon="🧠",
    layout="wide",
)

# ── Sidebar Navigation ─────────────────────────────────────────────────────────
st.sidebar.title("🧠 KG Builder")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["M1 — Data Preview", "M2 — Graph Builder", "M3 — Semantic Search", "M4 — Graph Explorer"],
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: M1 — Data Preview
# ══════════════════════════════════════════════════════════════════════════════
if page == "M1 — Data Preview":
    st.title("📊 Milestone 1 — Data Preview")
    st.markdown("Preview of the processed credit card transaction dataset.")

    csv_candidates = [
        os.path.join(os.path.dirname(__file__), "..", "data", "processed_creditcard.csv"),
        os.path.join(os.path.dirname(__file__), "..", "milestone_1", "processed_creditcard.csv"),
    ]
    csv_path = next((p for p in csv_candidates if os.path.exists(p)), None)

    if csv_path:
        import pandas as pd
        df = pd.read_csv(csv_path)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Rows", f"{len(df):,}")
        col2.metric("Total Columns", len(df.columns))
        if "is_fraud" in df.columns:
            fraud_count = int(df["is_fraud"].sum())
            col3.metric("Fraud Transactions", f"{fraud_count:,}")
        elif "fraud_label" in df.columns:
            fraud_count = int((df["fraud_label"] == "Fraud").sum())
            col3.metric("Fraud Transactions", f"{fraud_count:,}")

        st.markdown("### 🔍 Dataset Preview (first 100 rows)")
        st.dataframe(df.head(100), use_container_width=True)

        st.markdown("### 📈 Column Statistics")
        st.dataframe(df.describe(), use_container_width=True)

        st.markdown("### 🕳️ Null Values per Column")
        null_df = df.isnull().sum().reset_index()
        null_df.columns = ["Column", "Null Count"]
        st.dataframe(null_df, use_container_width=True)
    else:
        st.warning(
            "processed_creditcard.csv not found. "
            "Run `milestone_1/notebook.ipynb` first to generate it."
        )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: M2 — Graph Builder
# ══════════════════════════════════════════════════════════════════════════════
elif page == "M2 — Graph Builder":
    st.title("🔗 Milestone 2 — Graph Builder")
    st.markdown("Build the Neo4j knowledge graph from the processed dataset.")

    if not NEO4J_AVAILABLE:
        st.warning(
            "⚠️ Neo4j not connected. Start Neo4j Desktop, ensure it is running "
            "on bolt://localhost:7687, then reload this page."
        )
    else:
        st.success("✅ Neo4j connected.")

    limit = st.number_input(
        "Row limit (use 0 for all rows)", min_value=0, max_value=10000, value=100, step=50
    )

    if st.button("🚀 Build Knowledge Graph"):
        if not NEO4J_AVAILABLE:
            st.error("Cannot build graph — Neo4j is not connected.")
        else:
            try:
                sys.path.insert(
                    0, os.path.join(os.path.dirname(__file__), "..")
                )
                from data_bridge import run_pipeline
                with st.spinner("Running pipeline..."):
                    run_pipeline(limit=int(limit) if limit > 0 else None)
                st.success("✅ Pipeline complete!")

                stats = get_graph_stats()
                c1, c2, c3 = st.columns(3)
                c1.metric("Nodes", stats["node_count"])
                c2.metric("Relationships", stats["relationship_count"])
                c3.metric("Fraud Nodes", stats["fraud_node_count"])
            except ImportError:
                st.error(
                    "data_bridge.py not found. "
                    "Ensure it exists in the project root directory."
                )
            except Exception as e:
                st.error(f"Pipeline failed: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: M3 — Semantic Search
# ══════════════════════════════════════════════════════════════════════════════
elif page == "M3 — Semantic Search":
    st.title("🔍 Milestone 3 — Semantic Search & RAG")
    st.markdown("Ask natural language questions about the enterprise data.")

    query = st.text_input("Enter your question:", placeholder="Which transactions were flagged as fraud?")

    if st.button("🔎 Search & Answer") and query.strip():
        with st.spinner("Searching and generating answer..."):
            if RAG_AVAILABLE:
                result = full_rag_query(query, SAMPLE_DOCUMENTS)
                answer = result.get("answer", "No answer generated.")
                sources = result.get("sources", [])
            else:
                model, index = load_rag_index(SAMPLE_DOCUMENTS)
                sources = semantic_search_local(query, SAMPLE_DOCUMENTS, model, index, k=3)
                answer = generate_answer_local(query, sources)

        st.markdown("### 💡 Answer")
        st.success(answer)

        st.markdown("### 📚 Top Retrieved Context Chunks")
        for i, chunk in enumerate(sources, 1):
            st.info(f"**[{i}]** {chunk}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: M4 — Graph Explorer
# ══════════════════════════════════════════════════════════════════════════════
elif page == "M4 — Graph Explorer":
    st.title("🌐 Milestone 4 — Knowledge Graph Explorer")

    if not NEO4J_AVAILABLE:
        st.warning(
            "⚠️ Neo4j not connected. Start Neo4j Desktop and reload this page. "
            "The graph visualisation will appear here once connected."
        )
    else:
        # ── Graph Stats Panel ──────────────────────────────────────────────
        st.markdown("### 📊 Graph Statistics")
        stats = get_graph_stats()
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Nodes", stats["node_count"])
        c2.metric("Total Relationships", stats["relationship_count"])
        c3.metric("Fraud Nodes", stats["fraud_node_count"])

        st.markdown("---")

        # ── Node Search ────────────────────────────────────────────────────
        st.markdown("### 🔍 Search Nodes")
        search_term = st.text_input(
            "Filter graph by node name (leave blank to show all):",
            placeholder="e.g. Infosys, Fraud, Microsoft"
        )

        if st.button("📡 Load / Refresh Graph"):
            data = fetch_graph(search_term=search_term)

            if not data:
                st.info(
                    "No graph data found. "
                    "Run the pipeline in M2 — Graph Builder first."
                )
            else:
                # ── Build pyvis network ────────────────────────────────────
                net = Network(
                    height="600px",
                    width="100%",
                    bgcolor="#1a1a2e",
                    font_color="white",
                    directed=True,
                )
                net.set_options("""
                {
                  "nodes": {"borderWidth": 2, "shadow": true},
                  "edges": {"color": {"inherit": "from"}, "smooth": {"type": "dynamic"}},
                  "physics": {"stabilization": {"iterations": 300}}
                }
                """)

                added_nodes = set()
                for row in data:
                    src, rel, tgt = row["source"], row["rel"], row["target"]

                    # Colour fraud nodes red for visibility
                    src_color = "#e74c3c" if "fraud" in str(src).lower() else "#3498db"
                    tgt_color = "#e74c3c" if "fraud" in str(tgt).lower() else "#2ecc71"

                    if src not in added_nodes:
                        net.add_node(src, label=src, color=src_color)
                        added_nodes.add(src)
                    if tgt not in added_nodes:
                        net.add_node(tgt, label=tgt, color=tgt_color)
                        added_nodes.add(tgt)
                    net.add_edge(src, tgt, label=rel)

                # ── Render via temp file (safe for deployed environments) ───
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".html", delete=False, encoding="utf-8"
                ) as tmp:
                    net.save_graph(tmp.name)
                    html_content = open(tmp.name, "r", encoding="utf-8").read()

                st.components.v1.html(html_content, height=620, scrolling=True)
                st.caption(
                    f"Showing {len(added_nodes)} nodes | "
                    f"{len(data)} relationships"
                    + (f' | Filter: "{search_term}"' if search_term else "")
                )