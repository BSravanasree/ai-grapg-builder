# 🧠 AI-Based Knowledge Graph Builder for Enterprise Intelligence

> An end-to-end pipeline that ingests enterprise financial transaction data, extracts entities and relationships using NLP, builds a queryable Neo4j knowledge graph, and exposes a Retrieval-Augmented Generation (RAG) Q&A system through an interactive Streamlit dashboard.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DATA FLOW (M1 → M2 → M3 → M4)                   │
└─────────────────────────────────────────────────────────────────────────┘

  [creditcard.csv]
        │
        ▼
┌───────────────────┐
│  MILESTONE 1      │  notebook.ipynb
│  Data Ingestion   │  • Load & validate CSV
│  & Processing     │  • Normalize (MinMaxScaler)
│                   │  • Feature engineer risk_flag
└────────┬──────────┘
         │  processed_creditcard.csv
         ▼
┌───────────────────┐
│  MILESTONE 2      │  data_bridge.py + entity_extractor.py + neo4j_loader.py
│  Entity &         │  • Generate text sentences per row
│  Relationship     │  • spaCy NER → entities
│  Extraction       │  • Dependency parse → (subj, verb, obj) triples
│                   │  • Load triples into Neo4j via MERGE
└────────┬──────────┘
         │  Neo4j Graph + triples list
         ▼
┌───────────────────┐
│  MILESTONE 3      │  rag_pipeline.py
│  RAG + Semantic   │  • SentenceTransformer embeddings
│  Search           │  • FAISS vector index
│                   │  • Gemini 1.5 Flash LLM answer generation
└────────┬──────────┘
         │  Q&A answers + retrieved sources
         ▼
┌───────────────────┐
│  MILESTONE 4      │  app_dashboard.py (launched via app.py)
│  Interactive      │  • Multi-page Streamlit dashboard
│  Dashboard        │  • pyvis graph visualisation
│                   │  • Node search, graph stats, fraud highlighting
└───────────────────┘
```

---

## ✅ Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.9 – 3.12 | [python.org](https://python.org) |
| Neo4j Desktop | 5.x | [neo4j.com/download](https://neo4j.com/download) — create a local database |
| Google Gemini API Key | — | [aistudio.google.com](https://aistudio.google.com) — free tier available |

---

## 🚀 Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/AI-Based-Knowledge-Graph-Builder-for-Enterprise-Intelligence.git
cd AI-Based-Knowledge-Graph-Builder-for-Enterprise-Intelligence
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Configure environment variables
```bash
# Copy the example file
copy .env.example .env        # Windows
# OR
cp .env.example .env          # Mac/Linux

# Open .env and fill in your real values:
# NEO4J_URI=bolt://localhost:7687
# NEO4J_USER=neo4j
# NEO4J_PASSWORD=your-actual-password
# GEMINI_API_KEY=your-actual-gemini-key
```

### 4. Start Neo4j
- Open **Neo4j Desktop**
- Start your local database (default port: 7687)

### 5. Run Milestone 1 — Data Preprocessing
```bash
cd milestone_1
jupyter notebook notebook.ipynb
# Run all cells — this generates data/processed_creditcard.csv
```

### 6. Launch the full application
```bash
# From the project root:
python app.py
# OR directly:
streamlit run app.py
```

> The dashboard will open at **http://localhost:8501**

---

## 📋 Milestone Summary

| Milestone | Weeks | What It Does | Status |
|---|---|---|---|
| M1 — Data Ingestion & Processing | Week 1–2 | Loads creditcard.csv, cleans/normalises data, engineers features, outputs processed CSV | ✅ Complete |
| M2 — Entity Extraction & Graph | Week 3–4 | Runs spaCy NER + dep-parse on transaction sentences, loads (subject, relation, object) triples into Neo4j | ✅ Complete |
| M3 — Semantic Search & RAG | Week 5–6 | Embeds documents with SentenceTransformers, builds FAISS index, generates answers via Gemini 1.5 Flash | ✅ Complete |
| M4 — Dashboard & Deployment | Week 7–8 | Multi-page Streamlit app with pyvis graph explorer, node search, fraud highlighting, stats panel | ✅ Complete |

---

## 🛠️ Tech Stack

| Tool / Library | Purpose |
|---|---|
| `pandas` + `numpy` | Data ingestion, cleaning, and transformation |
| `scikit-learn` | Feature normalisation (MinMaxScaler) |
| `spaCy (en_core_web_sm)` | Named Entity Recognition + dependency parsing |
| `Neo4j` (+ Python driver) | Knowledge graph storage and querying |
| `sentence-transformers` | Semantic embedding generation |
| `FAISS` | High-speed vector similarity search |
| `google-generativeai` | Gemini LLM for answer generation (RAG) |
| `LangChain` | RAG orchestration framework |
| `Streamlit` | Interactive web dashboard |
| `pyvis` | Interactive graph visualisation |
| `python-dotenv` | Secure credential management from `.env` |

---

## 📁 Project Structure

```
AI-Based-Knowledge-Graph-Builder-for-Enterprise-Intelligence/
├── app.py                        ← Main entry point (streamlit run app.py)
├── data_bridge.py                ← M1 → M2 pipeline connector
├── requirements.txt
├── README.md
├── .env                          ← Your secrets (NOT committed to git)
├── .env.example                  ← Safe template committed to git
├── .gitignore
│
├── data/
│   ├── creditcard.csv            ← Raw input data
│   └── processed_creditcard.csv  ← M1 output → M2 input
│
├── milestone_1/
│   └── notebook.ipynb            ← Data ingestion & preprocessing
│
├── milestone_2/
│   ├── entity_extractor.py       ← spaCy NER + relation extraction
│   ├── neo4j_loader.py           ← Neo4j graph writer (secure)
│   └── main_m2.py                ← M2 runner (calls data_bridge)
│
├── milestone_3/
│   ├── rag_pipeline.py           ← FAISS + Gemini RAG
│   └── test_embeddings.py        ← Embedding tests
│
└── milestone_4/
    └── app_dashboard.py          ← Streamlit dashboard (all 4 pages)
```

---

## 🔐 Security Notes

- **Never commit `.env`** — it is in `.gitignore`
- All Neo4j credentials are loaded via `os.getenv()` + `python-dotenv`
- Cypher relationship types are validated against an allowlist — no injection possible
- Node names are fully parameterised in all queries

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.
